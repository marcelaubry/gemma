"""Shared pytest fixtures and a heavy-dependency import safety net.

This is the central pytest support module (``conftest.py``) for the Gemma
Compute Monitor backend test suite. pytest imports a directory's
``conftest.py`` *before* it collects and imports the ``test_*.py`` modules in
that directory, so the module-level code here is guaranteed to run first. That
ordering lets this module discharge two responsibilities that must both take
effect before any test code is imported.

1. Heavy-dependency import safety net
-------------------------------------
The backend's ``app.instrumentation`` performs a top-level ``import jax`` and
``app.sse`` imports ``app.instrumentation`` (so importing ``app.sse`` pulls in
the JAX stack transitively, and ``app.main``'s ``/analyze`` handler imports
``app.sse`` lazily). On the Apple-Silicon deployment target JAX is present via
the Metal backend, but a generic CI/Linux runner that only exercises the HTTP
contracts need not -- and often cannot -- install the Metal-only JAX stack.
Without intervention, ``import app.sse`` (and therefore the lazy import inside
``/analyze``) would fail with ``ModuleNotFoundError: jax`` during collection.

To decouple the contract/endpoint tests from that Metal-only stack, the
module-level :func:`_stub_module_if_unimportable` call below inserts a
``MagicMock`` into ``sys.modules`` for ``jax`` / ``gemma`` *only when the real
module cannot be imported*. A real, importable ``jax``/``gemma`` on a dev
machine is therefore never clobbered. Because the test suite mocks the heavy
code paths (``app.instrumentation.run_layers`` and the sampler
``app.sse._time_tokens``) plus the ``powermetrics`` subprocess, the stubbed
modules are never actually executed -- they exist purely so that
``import app.main`` / ``import app.sse`` succeed and the real FastAPI routes,
CORS handling, and ``app.schemas`` response validation can still be exercised.

2. Shared fixtures and helpers
------------------------------
The fixtures here are intentionally thin: they only set up controlled state so
that the seven mandated checks in ``test_health.py`` and ``test_analyze.py``
run deterministically and fast. They never load the real ~8.5 GB Metal-only
model. Provided here:

* :data:`GEMMA3_4B_NUM_LAYERS` -- the verified layer count (34) for Gemma 3 4B.
* :func:`fake_config` -- a ``SimpleNamespace`` mirroring the loaded
  ``model.config`` attributes the backend reads.
* :func:`make_layer_event` -- a factory building valid seven-field
  ``LayerEvent`` payloads matching the immutable wire contract.
* :func:`model_loader_mod` / :func:`app_main` / :func:`schemas_mod` -- module
  accessors imported *after* the safety net has run, for patching.
* :func:`make_client` -- a factory building a ``fastapi.testclient.TestClient``
  whose readiness is controlled deterministically so the real model is never
  loaded.

Mocking strategy
----------------
The tests mock/stub the heavy gemma calls (``app.model_loader.load_model``,
``app.instrumentation.run_layers``, the sampler ``app.sse._time_tokens``) and
the ``powermetrics`` subprocess so the checks are deterministic and fast, while
still asserting the *real* immutable contracts from ``app.schemas``.

Constraints (AAP 0.7): the target runtime is Python >= 3.12; no non-Metal GPU
backend is referenced; the tests must not require root and must never modify
any ``gemma/**`` file.
``fastapi``, ``sse-starlette``, ``psutil`` and ``pydantic`` are lightweight,
cross-platform, and genuinely exercised, so they are *never* stubbed.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Every ``app.*`` and ``fastapi.testclient`` import in this module is performed
# INSIDE a fixture rather than at module scope. This is deliberate and
# load-bearing: the heavy-dependency import safety net immediately below must
# run before any ``import app.main`` / ``import app.sse`` is attempted, so the
# modules under test import cleanly even on a JAX-less runner. The lazy-import
# pylint check is therefore disabled file-wide rather than annotated per line.
# pylint: disable=import-outside-toplevel


# -----------------------------------------------------------------------------
# Heavy-dependency import safety net (TOP-LEVEL -- runs at conftest import time)
# -----------------------------------------------------------------------------
def _stub_module_if_unimportable(*dotted_names: str) -> None:
  """Insert a ``MagicMock`` into ``sys.modules`` for each unimportable module.

  For every fully-qualified module name in ``dotted_names`` that cannot be
  imported, a ``MagicMock`` is registered in ``sys.modules`` for the module
  *and* each of its parent packages, so a later ``import`` of that module (or
  ``from app.main import app``, which pulls it in transitively) succeeds on a
  host lacking the Apple-Silicon-only JAX Metal stack.

  The stubbing is strictly **conditional**: a module that imports successfully
  is left completely untouched, so a real ``jax``/``gemma`` on an Apple-Silicon
  dev machine is never clobbered. Because the suite mocks ``run_layers`` and
  ``_time_tokens``, a stubbed module is never actually executed.

  Args:
    *dotted_names: Fully-qualified module names to probe, e.g. ``"jax"`` or
      ``"jax.numpy"``. Parent packages are created as needed and the parent's
      attribute is wired to the stub so attribute access such as ``jax.numpy``
      also resolves to the stubbed submodule.
  """
  for dotted in dotted_names:
    try:
      __import__(dotted)
    except Exception:  # pylint: disable=broad-exception-caught
      # A plain ImportError (the package is absent) OR a jaxlib/Metal backend
      # initialization failure on a non-Apple-Silicon host both mean the module
      # is unusable here -- stub it and every ancestor package not yet present.
      parts = dotted.split(".")
      for i in range(len(parts)):
        mod_name = ".".join(parts[: i + 1])
        if mod_name not in sys.modules:
          sys.modules[mod_name] = MagicMock(name=mod_name)
      # Wire the parent's attribute to the stubbed submodule so attribute
      # access (e.g. ``jax.numpy``) resolves exactly as it would for a real
      # package, not just a bare ``import jax.numpy``.
      if "." in dotted:
        parent, _, child = dotted.rpartition(".")
        setattr(sys.modules[parent], child, sys.modules[dotted])


# Stub ONLY the Apple-Silicon-only heavy stack, and only when it is missing or
# its backend fails to initialize. WHY: ``app.instrumentation`` does a
# top-level ``import jax`` and ``app.sse`` imports ``app.instrumentation``, so
# without this net ``import app.main`` (whose ``/analyze`` handler lazily
# imports ``app.sse``) and any direct ``import app.sse`` in a test would raise
# ``ModuleNotFoundError: jax`` on a JAX-less CI runner. The heavy code paths
# (``run_layers`` and the sampler ``_time_tokens``) are mocked in the test
# modules, so the stubs are never actually called into.
_stub_module_if_unimportable("jax", "jax.numpy", "gemma", "gemma.gm")


# -----------------------------------------------------------------------------
# Verified layer count (AAP 0.7.2 conflict resolution: 34, never 18)
# -----------------------------------------------------------------------------
# Gemma 3 4B has 34 transformer layers (``_NUM_LAYERS_GEMMA3_4B = 34`` in
# gemma/gm/nn/_gemma.py:L34). It is NOT 18 -- 18 is Gemma 2B / Gemma 3 270M
# (``_NUM_LAYERS_GEMMA_2B`` / ``_NUM_LAYERS_GEMMA3_270M`` at L27 / L32). The
# tests assert against ``config.num_layers`` dynamically; this named constant
# documents the expected value and keeps the literal 34 in exactly one place.
GEMMA3_4B_NUM_LAYERS = 34


@pytest.fixture
def fake_config() -> SimpleNamespace:
  """Return a stand-in for the loaded Gemma 3 4B ``model.config``.

  Mirrors the attributes the backend reads from the real gemma
  ``TransformerConfig`` -- the Gemma3_4B dimensions (gemma/gm/nn/_gemma.py
  L221-249) and the ``num_layers`` ``cached_property`` (gemma/gm/nn/_config.py
  L86-88) -- so the instrumentation/metrics code paths and the tests can read
  ``num_layers`` / ``num_kv_heads`` / ``head_dim`` without loading the real
  model.

  Returns:
    A ``SimpleNamespace`` exposing the config attributes the backend reads:
    ``num_layers`` (34), ``num_kv_heads`` (4, grouped-query attention),
    ``head_dim`` (256), ``num_heads`` (8), ``embed_dim`` (2560) and
    ``hidden_dim`` (10240).
  """
  return SimpleNamespace(
      num_layers=GEMMA3_4B_NUM_LAYERS,  # 34 (derived dynamically in prod)
      num_kv_heads=4,  # grouped-query attention -- NOT num_heads (== 8)
      head_dim=256,
      num_heads=8,
      embed_dim=2560,
      hidden_dim=10240,  # 2560 * 8 // 2
  )


@pytest.fixture
def make_layer_event():
  """Return a factory that builds a valid seven-field ``LayerEvent`` payload.

  The produced dict reproduces the IMMUTABLE SSE layer-event contract exactly,
  so ``app.schemas.LayerEvent(**event)`` validates and the dict's key set is
  precisely the seven contract keys. ``layer`` is an ``int`` (the zero-based
  layer index) and every other field is a ``float``; ``gpu_pct`` defaults to
  ``0.0`` to mirror the documented ``powermetrics``-unavailable fallback.

  Returns:
    A callable ``_make(layer_index, config=None) -> dict``. The optional
    ``config`` argument is accepted for call-site flexibility (tests may pass a
    ``fake_config``); the payload values are fixed and deterministic.
  """

  def _make(layer_index: int, config=None) -> dict:
    # ``config`` is accepted for call-site flexibility (a caller may pass the
    # ``fake_config``) but the payload below uses fixed, deterministic values;
    # bind it to a throwaway so it is genuinely referenced (never "unused").
    _ = config
    # Keys and types reproduce the IMMUTABLE LayerEvent contract (AAP 0.1.2):
    #   { layer: int, gpu_pct: float, cpu_pct: float, memory_used_gb: float,
    #     kv_cache_gb: float, activation_gb: float, elapsed_ms: float }
    # ``layer`` MUST be int and every other field MUST be float so that
    # ``LayerEvent(**event)`` validates and ``set(event)`` equals the 7 keys.
    return {
        "layer": int(layer_index),
        "gpu_pct": 0.0,  # 0.0 mirrors the powermetrics-unavailable fallback
        "cpu_pct": 12.5,
        "memory_used_gb": 9.0,
        "kv_cache_gb": 0.5,
        "activation_gb": 0.25,
        "elapsed_ms": 3.0,
    }

  return _make


# -----------------------------------------------------------------------------
# Module accessor fixtures (import the app.* modules AFTER the safety net runs)
# -----------------------------------------------------------------------------
@pytest.fixture
def model_loader_mod():
  """Return the ``app.model_loader`` module for patching.

  Imported inside the fixture (not at module scope) so the import safety net
  above is guaranteed to have run first, letting ``app.model_loader`` import
  cleanly even on a JAX-less runner.

  Returns:
    The imported ``app.model_loader`` module object.
  """
  from app import model_loader

  return model_loader


@pytest.fixture
def app_main():
  """Return the ``app.main`` module (its ``app`` attribute is the FastAPI app).

  ``app_main.app`` is the ASGI application served by uvicorn and
  ``app_main.analyze_lock`` is the process-wide single-flight lock. Imported
  inside the fixture so the import safety net above has already run.

  Returns:
    The imported ``app.main`` module object.
  """
  from app import main

  return main


@pytest.fixture
def schemas_mod():
  """Return the ``app.schemas`` module (the four immutable contract models).

  Convenience accessor so tests can validate payloads against the real
  ``LayerEvent`` / ``FinalEvent`` / ``AnalyzeRequest`` / ``HealthResponse``
  models. ``app.schemas`` is itself JAX-free, but it is imported here too so
  every module accessor shares one consistent, post-safety-net import style.

  Returns:
    The imported ``app.schemas`` module object.
  """
  from app import schemas

  return schemas


# -----------------------------------------------------------------------------
# TestClient factory (deterministic readiness; NEVER loads the real model)
# -----------------------------------------------------------------------------
@pytest.fixture
def make_client(monkeypatch):
  """Return a factory building a ``TestClient`` with controlled readiness.

  The factory builds a ``fastapi.testclient.TestClient`` for the real
  ``app.main:app`` whose model-load readiness is driven entirely by patches, so
  the real ~8.5 GB Metal-only model is **never** loaded:

  * ``app.model_loader.load_model`` is stubbed to a no-op -- a
    belt-and-suspenders guard in case a ``TestClient`` version auto-runs the
    FastAPI lifespan.
  * ``app.model_loader.get_status`` / ``is_ready`` are patched so ``/health``
    and the ``/analyze`` readiness guard observe the requested state. The
    endpoints call these as *module attributes*
    (``model_loader.get_status()``), so patching the module is observed.
  * ``app.main.analyze_lock`` is replaced with a FRESH ``asyncio.Lock`` per
    client so a lock held or leaked by one test can never strand a later
    ``/analyze`` request in a permanent ``429`` (``asyncio.Lock`` binds to the
    running loop lazily on Python 3.12+).

  The client is built WITHOUT the ``with`` context manager: that means the
  FastAPI lifespan (which would otherwise call ``load_model``) does not run, so
  readiness is governed solely by the ``get_status``/``is_ready`` patches while
  the real routes, CORS, and ``app.schemas`` response validation are still
  exercised. ``monkeypatch`` automatically restores every patch at teardown.

  Args:
    monkeypatch: The pytest ``monkeypatch`` fixture, used to apply and
      auto-restore the readiness patches.

  Returns:
    A callable ``_make(status="loading", *, ready=None) -> TestClient`` where
    ``status`` is the value reported by ``/health`` and ``ready`` optionally
    overrides the boolean readiness flag (defaulting to ``status == "ready"``).
  """
  from fastapi.testclient import TestClient

  from app import main as main_mod
  from app import model_loader

  created: list[TestClient] = []

  def _make(
      status: str = "loading", *, ready: bool | None = None
  ) -> TestClient:
    # (1) Never load the real model: stub the loader to a no-op. This is a
    #     belt-and-suspenders guard (the no-``with`` build below already skips
    #     the lifespan that would call it).
    monkeypatch.setattr(model_loader, "load_model", lambda: None, raising=True)
    # (2) Deterministically control the readiness flag the endpoints read.
    ready_flag = (status == "ready") if ready is None else ready
    monkeypatch.setattr(
        model_loader, "get_status", lambda: status, raising=True
    )
    monkeypatch.setattr(
        model_loader, "is_ready", lambda: ready_flag, raising=True
    )
    # (3) Give each client a FRESH single-flight lock so a held/leaked lock
    #     never causes spurious cross-test 429s depending on test order.
    monkeypatch.setattr(main_mod, "analyze_lock", asyncio.Lock(), raising=True)
    # Build WITHOUT the ``with`` context manager so the FastAPI lifespan
    # (and therefore the model load) does NOT run; readiness is driven entirely
    # by the patches above while the real routes are still exercised.
    client = TestClient(main_mod.app)
    created.append(client)
    return client

  yield _make

  # Teardown: close every client this factory created. ``monkeypatch`` restores
  # the patched module attributes automatically, so only the clients need
  # explicit cleanup here.
  for client in created:
    try:
      client.close()
    except Exception:  # pylint: disable=broad-exception-caught
      # Closing is best-effort; a teardown error must never mask a test result.
      pass
