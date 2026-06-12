"""Server-Sent Events bridge for ``POST /analyze`` (layer + final events).

This module turns the **synchronous, blocking** per-layer instrumentation
harness (:func:`app.instrumentation.run_layers`) and the gemma sampler into a
single **asynchronous** Server-Sent Events (SSE) stream, returned to FastAPI
through ``sse-starlette``'s :class:`~sse_starlette.sse.EventSourceResponse`.

Stream shape (immutable contracts -- AAP 0.1.2 / 0.7.1)
------------------------------------------------------
For one ``/analyze`` request the stream emits, in order:

* **One layer event per transformer layer** -- exactly ``config.num_layers``
  events (== 34 for Gemma 3 4B), each carrying the seven
  :class:`app.schemas.LayerEvent` keys ``{layer, gpu_pct, cpu_pct,
  memory_used_gb, kv_cache_gb, activation_gb, elapsed_ms}``.
* **One final event** -- :class:`app.schemas.FinalEvent`, serialized as
  ``{"per_token_ms": [...], "done": true}``.

Each event is emitted as an SSE ``data:`` payload (the Pydantic model
serialized with ``model_dump_json()``) tagged with an optional ``event:`` name
(``layer`` / ``final``). The frontend classifies events by payload content, so
the name is advisory only -- but emitting it keeps the wire standards-clean.

Non-blocking + disconnect contract (AAP 0.5.2 / Phase 6)
-------------------------------------------------------
The harness and the sampler perform **blocking JAX work**, which must never run
on the event loop:

* ``run_layers`` (a synchronous generator) is drained on a **worker thread**
  that pushes each layer dict into an :class:`asyncio.Queue` via
  ``loop.call_soon_threadsafe``; the async generator awaits ``queue.get()`` so
  per-layer events flow to the client in real time without stalling the loop.
* The per-token timing pass runs through ``anyio.to_thread.run_sync`` for the
  same reason.

Client disconnects are honored two ways: ``EventSourceResponse`` closes the
generator when the HTTP connection drops, and the generator additionally polls
``await request.is_disconnected()`` between layers, signalling the worker thread
to stop (via a :class:`threading.Event`) so no work is orphaned.

Instrument, never reimplement (AAP 0.7.1)
-----------------------------------------
Per-token timing reuses gemma's own sampler as-is -- ``gm.text.Sampler`` with
``stream=True`` (genuine token-by-token streaming) -- and ``gemma`` is imported
lazily inside the helper so importing this module never triggers a JAX/Metal
init. No model is loaded or reloaded here: the loaded singleton is obtained from
:mod:`app.model_loader`. No CUDA / NVIDIA backend is referenced anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, AsyncIterator

import anyio
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from app import instrumentation, model_loader, schemas

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tunables and constants
# -----------------------------------------------------------------------------
DEFAULT_MAX_NEW_TOKENS: int = 32
"""Default number of tokens decoded for the per-token timing pass.

Kept modest (32) so an ``/analyze`` stream stays short while still producing a
meaningful ``per_token_ms`` array for the frontend "Per-Token Compute Cost"
strip. It is comfortably below the sampler's ``max_out_length`` (2048), so the
sampler never raises for exceeding its output buffer."""

SSE_PING_INTERVAL_S: int = 15
"""Interval (seconds) between SSE keep-alive comments emitted by
``EventSourceResponse``. Pings are SSE comment lines (``: ...``) that the
frontend parser ignores, so they keep the connection alive through proxies
(e.g. the ngrok tunnel) without ever being mistaken for telemetry events."""

# SSE ``event:`` names. Advisory only -- the frontend classifies events by
# payload content -- but emitted for standards-clean, debuggable output.
_LAYER_EVENT_NAME: str = "layer"
_FINAL_EVENT_NAME: str = "final"

# Name for the layer-pump worker thread (aids debugging / thread dumps).
_WORKER_THREAD_NAME: str = "gemma-layer-pump"

# Sentinel pushed onto the queue by the worker thread to mark "no more layers".
# A unique object so it can never collide with a real layer dict or exception.
_QUEUE_SENTINEL: object = object()


# -----------------------------------------------------------------------------
# Per-token timing helpers (synchronous; always run inside a worker thread)
# -----------------------------------------------------------------------------
def _count_tokens(sampler_output: Any) -> int:
  """Best-effort count of generated tokens from a ``SamplerOutput``.

  Reads the ``tokens`` attribute exposed by ``gm.text.SamplerOutput`` (the
  predicted-token array) and returns its length. Returns ``0`` when the
  attribute is missing or has no usable length, so the bulk-timing fallback in
  :func:`_time_tokens` can decide there is nothing to time rather than raising.

  Args:
    sampler_output: The object returned by ``Sampler.sample(...,
      return_state=True)``; expected to expose a ``tokens`` array.

  Returns:
    The number of generated tokens, or ``0`` if it cannot be determined.
  """
  tokens = getattr(sampler_output, "tokens", None)
  if tokens is None:
    return 0
  try:
    return int(len(tokens))
  except TypeError:
    # A 0-d array (or otherwise length-less object) -- nothing to count.
    return 0


def _time_tokens(
    model: Any,
    params: Any,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[float]:
  """Time gemma per-token generation, returning per-token latencies (ms).

  This is **synchronous and blocking** (it drives JAX through the gemma
  sampler) and is therefore always invoked from a worker thread via
  ``anyio.to_thread.run_sync`` -- never directly on the event loop.

  Instrument, never reimplement (AAP 0.7.1): generation uses gemma's own
  ``gm.text.Sampler`` as-is. ``gemma`` (and the heavy JAX/Flax stack it pulls
  in) is imported **lazily** here so merely importing :mod:`app.sse` never
  triggers a JAX/Metal initialization.

  Timing strategy:
    * **Primary -- genuine per-token streaming.** ``sampler.sample(prompt,
      max_new_tokens=N, stream=True)`` yields decoded tokens one at a time
      (``gemma/gm/text/_sampler.py`` -> ``_stream_decode_state``). The
      wall-clock delta between successive yields is recorded as that token's
      latency, so the result reflects real decode cadence.
    * **Fallback -- bulk timing.** If ``stream=True`` is not iterable in the
      running environment, the whole ``sampler.sample(..., return_state=True)``
      call is timed once and divided evenly across the number of generated
      tokens (decoded via ``len(output.tokens)``). True streaming is preferred;
      this only runs when streaming yields nothing.

  Resilience: any failure is swallowed and whatever timings were collected (
  possibly an empty list) are returned, so the SSE final event is always able
  to emit -- this helper never raises into the stream.

  Args:
    model: The loaded ``gm.nn.Gemma3_4B`` instance from
      :func:`app.model_loader.get_model_and_params`.
    params: The restored checkpoint parameter tree (paired with ``model``).
    prompt: The user prompt to generate from.
    max_new_tokens: Maximum number of tokens to decode (default
      :data:`DEFAULT_MAX_NEW_TOKENS`).

  Returns:
    A list of per-token latencies in milliseconds, in generation order. May be
    empty if generation produced no tokens or failed.
  """
  per_token_ms: list[float] = []
  try:
    # Lazy heavy import -- keep gemma/JAX out of module-import time.
    from gemma import gm  # pylint: disable=import-outside-toplevel

    # Compose the gemma sampler as-is. The tokenizer is auto-resolved from
    # ``model.INFO.tokenizer_version`` inside ``Sampler.__post_init__`` -- no
    # explicit wiring required.
    sampler = gm.text.Sampler(model=model, params=params)

    # PRIMARY: stream=True yields one decoded token per iteration. Record the
    # delta between successive yields as each token's latency.
    try:
      token_stream = sampler.sample(
          prompt,
          max_new_tokens=max_new_tokens,
          stream=True,
      )
      last = time.perf_counter()
      # The token value is irrelevant -- only the decode cadence is timed.
      for _ in token_stream:
        now = time.perf_counter()
        per_token_ms.append((now - last) * 1000.0)
        last = now
    except TypeError:
      # ``stream=True`` not iterable in this environment; fall through to the
      # bulk-timing fallback below.
      logger.debug(
          "Sampler stream=True not iterable; using bulk-timing fallback."
      )

    if per_token_ms:
      return per_token_ms

    # FALLBACK: time the whole generation once and divide evenly by the number
    # of generated tokens. Preferred path is streaming above; this runs only
    # when streaming yielded nothing.
    start = time.perf_counter()
    output = sampler.sample(
        prompt,
        max_new_tokens=max_new_tokens,
        return_state=True,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    num_tokens = _count_tokens(output)
    if num_tokens > 0:
      per_token_ms = [total_ms / num_tokens] * num_tokens
  except Exception:  # pylint: disable=broad-exception-caught
    # Never let a timing failure abort the SSE stream. Return whatever was
    # collected (possibly empty) so the final event still emits.
    logger.exception(
        "Per-token timing failed; returning %d collected sample(s).",
        len(per_token_ms),
    )
  return per_token_ms


# -----------------------------------------------------------------------------
# The async SSE event generator
# -----------------------------------------------------------------------------
async def analyze_event_stream(
    request: Request,
    prompt: str,
) -> AsyncIterator[dict]:
  """Yield the ``/analyze`` SSE events: one per layer, then one final event.

  Bridges the blocking, synchronous :func:`app.instrumentation.run_layers`
  generator and the gemma sampler into an async stream of SSE-ready ``dict``
  payloads (``{"event": ..., "data": ...}``) consumed by
  :class:`~sse_starlette.sse.EventSourceResponse`.

  Flow:
    1. Obtain the loaded ``model``, ``params`` and ``config`` from
       :mod:`app.model_loader` (readiness is guaranteed by ``main.py`` before
       this is called -- it returns ``503`` while loading).
    2. **Layer phase.** Drain ``run_layers`` on a daemon worker thread that
       enqueues each layer dict onto an :class:`asyncio.Queue` via
       ``loop.call_soon_threadsafe``. For each dequeued dict, poll
       ``request.is_disconnected()`` first (stop cleanly if the client left),
       then validate it through :class:`app.schemas.LayerEvent` and yield it as
       a ``data:`` JSON payload. Exactly ``config.num_layers`` layer events are
       produced on the happy path.
    3. **Final phase.** Time per-token generation off the event loop via
       ``anyio.to_thread.run_sync(_time_tokens, ...)`` and yield exactly one
       :class:`app.schemas.FinalEvent` (``{"per_token_ms": [...], "done":
       true}``).

  The event loop is never blocked by JAX: the layer harness runs on the worker
  thread and the sampler runs in a thread executor. On client disconnect the
  worker thread is signalled to stop (``stop_event``) at the next layer
  boundary, so no work is orphaned. If the layer harness raises, the error is
  logged and the stream still terminates gracefully by emitting the final event
  (unless the client has already disconnected).

  Args:
    request: The Starlette/FastAPI request, polled via
      ``await request.is_disconnected()`` to detect a closed ``EventSource``.
    prompt: The user prompt forwarded to the instrumentation harness and the
      sampler.

  Yields:
    SSE event ``dict`` objects with a ``data`` JSON string (and an advisory
    ``event`` name) -- one per transformer layer, then one final event.
  """
  # Reuse the loaded singleton; never load/reload a model here (AAP Phase 6).
  model, params = model_loader.get_model_and_params()
  config = model_loader.get_config()

  loop = asyncio.get_running_loop()
  queue: asyncio.Queue = asyncio.Queue()
  stop_event = threading.Event()

  def _safe_enqueue(value: Any) -> None:
    """Enqueue ``value`` from the worker thread, tolerating a closed loop.

    Uses ``loop.call_soon_threadsafe`` so the async side wakes on
    ``queue.get()``. If the event loop has already been closed (e.g. during
    server shutdown) the call is dropped rather than raising on the worker.
    """
    try:
      loop.call_soon_threadsafe(queue.put_nowait, value)
    except RuntimeError:
      # Event loop is closed -- there is no consumer left to deliver to.
      logger.debug("Event loop closed; dropping a queued SSE item.")

  def _drain_layers() -> None:
    """Run the blocking layer harness, enqueueing each event (worker thread).

    Iterates the synchronous ``run_layers`` generator and forwards every layer
    dict to the async side. Any exception is captured and forwarded so the
    async generator can log it and stop. A sentinel is always enqueued last so
    the consumer's loop terminates even on early stop or error.
    """
    try:
      for layer_dict in instrumentation.run_layers(
          model, params, config, prompt
      ):
        if stop_event.is_set():
          logger.debug("Stop signalled; ending layer worker early.")
          break
        _safe_enqueue(layer_dict)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Forward the failure to the async side, which logs and stops the stream.
      logger.debug("Layer harness raised in worker thread: %s", exc)
      _safe_enqueue(exc)
    finally:
      _safe_enqueue(_QUEUE_SENTINEL)

  worker = threading.Thread(
      target=_drain_layers,
      name=_WORKER_THREAD_NAME,
      daemon=True,
  )
  worker.start()

  client_disconnected = False
  layer_error: Exception | None = None
  emitted = 0
  try:
    while True:
      item = await queue.get()
      if item is _QUEUE_SENTINEL:
        break
      if isinstance(item, Exception):
        # The harness failed; stop streaming layers but still close the stream
        # gracefully with a final event below.
        layer_error = item
        logger.error("Layer harness raised; stopping layers: %s", item)
        break
      # ``item`` is a layer dict. Honor client disconnect BEFORE emitting so a
      # closed EventSource yields no further events.
      if await request.is_disconnected():
        client_disconnected = True
        logger.info(
            "Client disconnected after %d layer event(s); stopping.",
            emitted,
        )
        break
      # Defensive validation guarantees the immutable wire shape.
      layer_event = schemas.LayerEvent(**item)
      emitted += 1
      yield {
          "event": _LAYER_EVENT_NAME,
          "data": layer_event.model_dump_json(),
      }
  finally:
    # Signal the worker to stop at the next layer boundary so no JAX work is
    # left running after the consumer goes away (also covers the disconnect /
    # cancellation paths, where this generator is closed mid-iteration).
    stop_event.set()

  # If the client is gone, do not run the (expensive) sampler or emit a final
  # event -- just stop cleanly.
  if client_disconnected:
    return

  # Final phase: per-token timing. Always attempt unless the client left.
  if await request.is_disconnected():
    logger.info("Client disconnected before final event; skipping it.")
    return

  # Run the blocking sampler off the event loop. ``_time_tokens`` never raises
  # (it returns whatever it collected), so the final event always emits.
  per_token_ms = await anyio.to_thread.run_sync(
      _time_tokens, model, params, prompt
  )
  final_event = schemas.FinalEvent(per_token_ms=per_token_ms)
  yield {
      "event": _FINAL_EVENT_NAME,
      "data": final_event.model_dump_json(),
  }
  if layer_error is not None:
    logger.warning(
        "Emitted final event despite an earlier layer error: %s",
        layer_error,
    )


# -----------------------------------------------------------------------------
# Response factory (called by app.main after the 503 / 429 guards)
# -----------------------------------------------------------------------------
def analyze_response(request: Request, prompt: str) -> EventSourceResponse:
  """Build the ``EventSourceResponse`` for ``POST /analyze``.

  Wraps :func:`analyze_event_stream` in
  :class:`~sse_starlette.sse.EventSourceResponse`, which renders each yielded
  ``dict`` as an SSE frame and provides built-in client-disconnect handling on
  top of the generator's own ``is_disconnected()`` polling. ``app.main`` calls
  this only after the readiness (``503``) and single-flight (``429``) guards
  have passed.

  Args:
    request: The Starlette/FastAPI request (forwarded for disconnect polling).
    prompt: The validated user prompt to analyze.

  Returns:
    An :class:`~sse_starlette.sse.EventSourceResponse` streaming the layer
    events followed by the final per-token event.
  """
  return EventSourceResponse(
      analyze_event_stream(request, prompt),
      ping=SSE_PING_INTERVAL_S,
  )


# Explicit public API consumed by ``app.main``. The per-token timing helpers
# (``_time_tokens`` / ``_count_tokens``) are private and intentionally omitted.
__all__ = [
    "analyze_event_stream",
    "analyze_response",
    "DEFAULT_MAX_NEW_TOKENS",
]
