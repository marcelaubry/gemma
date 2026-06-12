"""Tests for ``POST /analyze`` SSE streaming (mandated checks 1, 2, 3, 6, 7).

Validates the ``POST /analyze`` Server-Sent Events (SSE) stream and the
``powermetrics`` GPU-metric fallback of the Gemma Compute Monitor backend
(``app.main`` / ``app.sse`` / ``app.metrics``). This module exercises **five**
of the seven mandated checks from the Agent Action Plan:

* **Check 1** -- the stream emits **exactly ``config.num_layers`` layer events**
  (== ``34`` for Gemma 3 4B), asserted **dynamically** against the loaded /
  mocked configuration and **never** a hardcoded ``18`` (AAP §0.7.2: ``18`` is
  Gemma 2B / Gemma 3 270M, not Gemma 3 4B).
* **Check 2** -- each **layer event matches the immutable seven-field schema**,
  validated against the real :class:`app.schemas.LayerEvent`.
* **Check 3** -- the stream is **terminated by the final event**
  ``{ per_token_ms: float[], done: true }`` (:class:`app.schemas.FinalEvent`).
* **Check 6** -- a **second concurrent ``/analyze``** while one analysis is in
  flight is rejected with **HTTP 429** (the single-flight guard in
  ``app.main``).
* **Check 7** -- the **``powermetrics``-unavailable ``0.0`` fallback**:
  :func:`app.metrics.gpu_pct` returns ``0.0`` when ``powermetrics`` is absent
  or times out (200 ms), and never raises.

Mocking strategy
----------------
The heavy gemma/JAX work is mocked at its leaves -- ``instrumentation.run_layers``
(the eager per-layer harness), the sampler ``sse._time_tokens``, and
``model_loader``'s readiness/accessors -- and the ``powermetrics`` subprocess is
stubbed, so the checks are deterministic, fast, and require neither JAX, Metal,
a checkpoint, nor root. Crucially the tests still exercise the **real**
``app.sse`` async generator, the **real** ``app.main`` ``503``/``429`` route
guards, and the **real** ``app.schemas`` serialization
(``model_dump_json()`` -> ``"done": true``), so the immutable wire contracts
are genuinely asserted.

The ``run_layers`` mock loops ``range(config.num_layers)`` exactly like the real
eager harness, so the layer-event count is genuinely **parameterized** to the
loaded configuration -- the "34 not 18" resolution is enforced structurally
rather than by a magic constant.

Run from inside ``backend/``::

    python -m pytest tests/test_analyze.py -q

Constraints (AAP §0.7): target runtime Python >= 3.12; no CUDA / non-Metal GPU
backend is referenced; no root is required; no ``gemma/**`` file is modified.
"""

from __future__ import annotations

import json
import subprocess

import pytest

# ``app.schemas`` is pure Pydantic (it imports no jax/gemma), so a direct
# module-scope import is safe and side-effect free; it lets every test validate
# payloads against the REAL immutable contracts. The heavier ``app`` modules
# (``metrics``, ``model_loader``, ``instrumentation``, ``sse``, ``main``) are
# imported INSIDE the fixtures/tests below so the conftest top-level
# JAX/gemma import safety net is guaranteed to have run first.
from app import schemas


# -----------------------------------------------------------------------------
# SSE parsing helper
# -----------------------------------------------------------------------------
def _parse_sse_events(text: str) -> list[dict]:
  """Parse an SSE response body into the list of JSON ``data:`` payloads.

  Tolerant of both ``\\r\\n`` and ``\\n`` line endings and of any
  ``event:`` / ``id:`` / ``:comment`` / ping lines (which are ignored): only
  ``data:`` lines are collected, joined per SSE event block, and decoded as
  JSON. A block that fails to decode is skipped rather than raising, so a stray
  comment or keep-alive frame can never break parsing.

  Args:
    text: The raw SSE response body (the full ``EventSourceResponse`` stream).

  Returns:
    A list of decoded JSON event payloads, in stream order.
  """
  events: list[dict] = []
  for block in text.replace("\r\n", "\n").split("\n\n"):
    data_parts = [
        line[len("data:"):].lstrip()
        for line in block.split("\n")
        if line.startswith("data:")
    ]
    if data_parts:
      try:
        events.append(json.loads("\n".join(data_parts)))
      except json.JSONDecodeError:
        # A non-JSON data block (e.g. a stray keep-alive) is ignored rather
        # than aborting the parse of the remaining, well-formed events.
        pass
  return events


# The exact, immutable seven-key set of the SSE layer-event contract
# (AAP §0.1.2). Declared once so Check 1/2 can assert "exactly these keys, no
# drift" without re-listing them inline.
EXPECTED_LAYER_KEYS = {
    "layer",
    "gpu_pct",
    "cpu_pct",
    "memory_used_gb",
    "kv_cache_gb",
    "activation_gb",
    "elapsed_ms",
}


# -----------------------------------------------------------------------------
# sse_client fixture: a ready client streaming a finite, deterministic SSE body
# -----------------------------------------------------------------------------
@pytest.fixture
def sse_client(make_client, fake_config, make_layer_event, monkeypatch):
  """Build a ready ``TestClient`` whose ``/analyze`` yields a finite SSE stream.

  Patches every heavy path so a ``POST /analyze`` produces a fast, finite,
  deterministic stream of exactly ``fake_config.num_layers`` layer events
  followed by one final per-token event, while leaving the REAL ``app.sse``
  generator, ``app.schemas`` serialization, and ``app.main`` guards in force:

  * ``model_loader.get_model_and_params`` -> opaque placeholders (``run_layers``
    is mocked, so the real model/params are never needed).
  * ``model_loader.get_config`` -> the :func:`fake_config` stand-in
    (``num_layers == 34``), so ``app.sse`` reads the layer count dynamically.
  * ``instrumentation.run_layers`` -> a generator looping
    ``range(config.num_layers)``, mirroring the real eager harness so the event
    count is genuinely parameterized (34, never 18).
  * ``sse._time_tokens`` -> a fixed list, so the final event is deterministic
    and the sampler is never actually run.

  Patching ``instrumentation.run_layers`` and ``sse._time_tokens`` on their
  own modules works because ``app.sse`` resolves both through the module
  namespace at call time (``instrumentation.run_layers`` is module-qualified and
  ``_time_tokens`` is a module global), so the real generator picks up the
  patched callables.

  Args:
    make_client: Conftest factory building a readiness-controlled client.
    fake_config: Conftest stand-in for the loaded Gemma 3 4B config
      (``num_layers == 34``).
    make_layer_event: Conftest factory producing a valid seven-field layer
      event dict.
    monkeypatch: Pytest fixture used to apply and auto-restore the patches.

  Returns:
    A ``fastapi.testclient.TestClient`` whose ``/analyze`` streams
    ``fake_config.num_layers`` layer events plus one final event.
  """
  from app import instrumentation, model_loader, sse

  client = make_client("ready")  # is_ready() -> True; fresh analyze_lock

  # Model/params are opaque placeholders -- ``run_layers`` is mocked below, so
  # they are never dereferenced; this avoids loading the real ~8.5 GB model.
  monkeypatch.setattr(
      model_loader,
      "get_model_and_params",
      lambda: (object(), object()),
      raising=True,
  )
  monkeypatch.setattr(
      model_loader, "get_config", lambda: fake_config, raising=True
  )

  # Mirror the REAL eager harness: yield EXACTLY ``config.num_layers`` events
  # (34 -- parameterized to the loaded config, never a hardcoded 18).
  def _fake_run_layers(model, params, config, prompt):
    for i in range(config.num_layers):
      yield make_layer_event(i, config)

  monkeypatch.setattr(
      instrumentation, "run_layers", _fake_run_layers, raising=True
  )

  # Mock the sampler timing so the final per-token event is deterministic and
  # fast (the real ``_time_tokens`` would drive gemma's sampler/JAX).
  monkeypatch.setattr(
      sse, "_time_tokens", lambda *a, **k: [10.0, 8.5, 9.25], raising=True
  )

  return client


def _collect_analyze_events(client, prompt: str) -> tuple[int, list[dict]]:
  """POST ``/analyze`` and return ``(status_code, parsed_events)``.

  Drains the SSE response via the streaming API so a finite telemetry stream is
  always fully read without any risk of blocking the test: ``client.stream``
  exposes the response before the body is consumed and ``resp.read()`` collects
  the complete body in one shot. Because the :func:`sse_client` fixture makes
  the stream finite (``num_layers`` layer events + exactly one final event),
  this always completes promptly. The body is parsed with
  :func:`_parse_sse_events`.

  Args:
    client: The ready ``TestClient`` from :func:`sse_client`.
    prompt: The prompt to send in the immutable ``{ "prompt": string }`` body.

  Returns:
    A ``(status_code, events)`` tuple where ``events`` is the list of decoded
    SSE JSON payloads in stream order.
  """
  with client.stream("POST", "/analyze", json={"prompt": prompt}) as resp:
    raw = resp.read()
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return resp.status_code, _parse_sse_events(text)


# -----------------------------------------------------------------------------
# Check 1: the stream emits EXACTLY ``config.num_layers`` layer events (34, not 18)
# -----------------------------------------------------------------------------
def test_analyze_emits_exactly_num_layers_layer_events(sse_client, fake_config):
  """``/analyze`` emits exactly ``config.num_layers`` layer events.

  The count is asserted **dynamically** against ``fake_config.num_layers`` --
  the value the (mocked) loaded configuration reports -- exactly as ``app.sse``
  drives the real eager harness, so the test is genuinely parameterized to the
  model rather than to a magic literal. A second assertion pins that value to
  ``34`` to lock in the AAP §0.7.2 resolution: Gemma 3 4B has
  ``_NUM_LAYERS_GEMMA3_4B == 34`` layers; ``18`` belongs to Gemma 2B /
  Gemma 3 270M and must NEVER appear here.

  Args:
    sse_client: Ready client whose ``/analyze`` streams a finite, deterministic
      body of ``fake_config.num_layers`` layer events plus one final event.
    fake_config: The mocked loaded configuration (``num_layers == 34``).
  """
  status_code, events = _collect_analyze_events(sse_client, "hello world")
  assert status_code == 200
  layer_events = [e for e in events if "layer" in e]
  # PARAMETERIZED to the loaded configuration -- NEVER a hardcoded 18.
  assert len(layer_events) == fake_config.num_layers
  # Lock in the §0.7.2 resolution: 34 == _NUM_LAYERS_GEMMA3_4B (18 is 2B/270M).
  assert fake_config.num_layers == 34


# -----------------------------------------------------------------------------
# Check 2: each layer event matches the immutable seven-field schema
# -----------------------------------------------------------------------------
def test_layer_events_match_immutable_schema(sse_client, fake_config):
  """Every layer event has EXACTLY the immutable seven LayerEvent fields.

  Each emitted layer event must carry precisely the seven keys of the immutable
  SSE layer-event contract (AAP §0.1.2) -- no missing keys, no extra drift --
  with ``layer`` an ``int`` and the remaining six numeric fields. Validating
  each event through the real :class:`app.schemas.LayerEvent` enforces the
  declared types and fails if the contract ever changes.

  Args:
    sse_client: Ready client streaming the deterministic SSE body.
    fake_config: The mocked configuration (used to assert the event count too).
  """
  _status_code, events = _collect_analyze_events(sse_client, "hi")
  layer_events = [e for e in events if "layer" in e]
  assert len(layer_events) == fake_config.num_layers
  for event in layer_events:
    # EXACTLY the seven contract keys -- no more, no less.
    assert set(event.keys()) == EXPECTED_LAYER_KEYS
    # ``layer`` MUST be an int per the contract (a zero-based layer index).
    assert isinstance(event["layer"], int)
    # Validate types through the REAL Pydantic model (raises on any drift).
    schemas.LayerEvent(**event)


# -----------------------------------------------------------------------------
# Check 3: the final per-token event terminates the stream
# -----------------------------------------------------------------------------
def test_stream_terminated_by_final_event(sse_client, fake_config):
  """The stream ends with exactly one ``{ per_token_ms, done: true }`` event.

  Asserts the terminal-event contract (AAP §0.1.2): the LAST event carries
  ``done == True`` and a ``list`` of numeric ``per_token_ms`` timings, validates
  it through the real :class:`app.schemas.FinalEvent`, and confirms there is
  exactly ONE final event with every preceding event being a layer event. The
  ``done: true`` literal is produced by the real ``FinalEvent.model_dump_json``
  serialization in ``app.sse``, so this also pins that wire form.

  Args:
    sse_client: Ready client streaming the deterministic SSE body.
    fake_config: The mocked configuration (``num_layers == 34``).
  """
  _status_code, events = _collect_analyze_events(sse_client, "hi")
  assert events, "expected at least one SSE event"
  # The LAST event is the terminal per-token event.
  final = events[-1]
  assert final.get("done") is True
  assert isinstance(final.get("per_token_ms"), list)
  assert all(isinstance(x, (int, float)) for x in final["per_token_ms"])
  # Validate against the REAL contract model: { per_token_ms: float[], done }.
  schemas.FinalEvent(**final)
  # Exactly ONE final event, and every preceding event is a layer event.
  assert sum(1 for e in events if e.get("done") is True) == 1
  assert all("layer" in e for e in events[:-1])
  assert len(events[:-1]) == fake_config.num_layers


# -----------------------------------------------------------------------------
# Check 6: a concurrent ``/analyze`` while one is in flight returns HTTP 429
# -----------------------------------------------------------------------------
class _InFlightLock:
  """Stand-in for an ``asyncio.Lock`` that is already held (analysis in flight).

  ``app.main``'s ``/analyze`` handler rejects a concurrent request by checking
  ``analyze_lock.locked()`` BEFORE awaiting/acquiring, so a lock whose
  ``locked()`` reports ``True`` deterministically reproduces the "an analysis is
  already running" condition without any real concurrency race. ``acquire`` /
  ``release`` are provided for interface completeness but are never reached on
  the ``429`` path (the handler raises before acquiring).
  """

  def locked(self) -> bool:
    """Report the lock as held so the single-flight guard fires."""
    return True

  async def acquire(self) -> bool:
    """Async-acquire shim (unused on the ``429`` path); always 'succeeds'."""
    return True

  def release(self) -> None:
    """Release shim (unused on the ``429`` path); a no-op."""
    return None


def test_second_concurrent_analyze_returns_429(make_client, app_main, monkeypatch):
  """A second ``/analyze`` while one is in flight is rejected with ``429``.

  Deterministically reproduces the single-flight contract (AAP §0.7.1): with
  the model ready (so the ``503`` readiness gate passes) and the module-level
  ``analyze_lock`` reporting ``locked() is True`` (an analysis already running),
  a ``POST /analyze`` with a VALID body must return ``429`` -- it must NOT queue
  behind the in-flight analysis. The status must be ``429`` specifically: not
  ``503`` (readiness passed), not ``422`` (the body is valid), and not ``200``
  (a second stream must never start).

  ``make_client("ready")`` installs a fresh ``asyncio.Lock`` on ``app.main``;
  this test then overrides it with an :class:`_InFlightLock`, so the handler's
  ``analyze_lock.locked()`` check -- resolved against the module global at call
  time -- observes the override and fires the guard.

  Args:
    make_client: Conftest factory; ``make_client("ready")`` forces
      ``is_ready()`` to ``True`` so the readiness gate passes.
    app_main: The ``app.main`` module, whose module-level ``analyze_lock`` is
      overridden to simulate an in-flight analysis.
    monkeypatch: Pytest fixture used to apply and auto-restore the override.
  """
  client = make_client("ready")  # is_ready() -> True, so the 503 gate passes

  # Override the fresh lock ``make_client`` installed with an 'in flight' one.
  # Applied AFTER ``make_client`` so this override is the live value the
  # handler observes when it evaluates ``analyze_lock.locked()``.
  monkeypatch.setattr(app_main, "analyze_lock", _InFlightLock(), raising=True)

  resp = client.post("/analyze", json={"prompt": "second request"})
  assert resp.status_code == 429


# -----------------------------------------------------------------------------
# Check 7: the ``powermetrics``-unavailable ``0.0`` GPU fallback (app.metrics)
# -----------------------------------------------------------------------------
def test_gpu_pct_zero_when_powermetrics_absent(monkeypatch):
  """``gpu_pct()`` returns ``0.0`` immediately when ``powermetrics`` is absent.

  When ``shutil.which("powermetrics")`` is ``None`` (e.g. a non-macOS host such
  as the Linux CI sandbox), :func:`app.metrics.gpu_pct` must return the
  documented ``0.0`` fallback (AAP §0.7.1) WITHOUT spawning any subprocess and
  without raising.

  Args:
    monkeypatch: Pytest fixture; patches ``metrics.shutil.which`` to report the
      tool as not installed.
  """
  from app import metrics

  # Tool not on PATH -> immediate 0.0 fallback, no subprocess spawned.
  monkeypatch.setattr(metrics.shutil, "which", lambda name: None, raising=True)
  result = metrics.gpu_pct()
  assert result == 0.0
  assert isinstance(result, float)


def test_gpu_pct_zero_on_timeout(monkeypatch):
  """``gpu_pct()`` returns ``0.0`` when the ``powermetrics`` subprocess times out.

  With the tool reported as present but its subprocess raising
  :class:`subprocess.TimeoutExpired` at the 200 ms timeout, :func:`app.metrics.
  gpu_pct` must catch the failure and return the ``0.0`` fallback -- never
  raising into the SSE stream and always returning a ``float``.

  Args:
    monkeypatch: Pytest fixture; patches ``metrics.shutil.which`` to report the
      tool present and ``metrics.subprocess.run`` to raise ``TimeoutExpired``.
  """
  from app import metrics

  # Tool 'present' but the subprocess times out at 200 ms -> 0.0 fallback.
  monkeypatch.setattr(
      metrics.shutil,
      "which",
      lambda name: "/usr/bin/powermetrics",
      raising=True,
  )

  def _raise_timeout(*args, **kwargs):
    raise subprocess.TimeoutExpired(
        cmd="powermetrics", timeout=metrics.POWERMETRICS_TIMEOUT_S
    )

  monkeypatch.setattr(metrics.subprocess, "run", _raise_timeout, raising=True)
  result = metrics.gpu_pct()
  assert result == 0.0  # never raises; always returns a float
  assert isinstance(result, float)


def test_powermetrics_timeout_and_fallback_constants():
  """The ``powermetrics`` timeout and GPU fallback constants match the contract.

  Pins the documented resilience knobs (AAP §0.5.2 / §0.7.1): the subprocess
  timeout is ``0.2`` s (200 ms) and the GPU fallback value is ``0.0``.
  """
  from app import metrics

  assert metrics.POWERMETRICS_TIMEOUT_S == 0.2  # 200 ms
  assert metrics.GPU_FALLBACK == 0.0
