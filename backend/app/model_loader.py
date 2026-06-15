"""One-time Gemma 3 4B model load and ``loading -> ready`` readiness state.

This module owns the **single, process-wide load** of the Gemma 3 4B model
used by the Gemma Compute Monitor backend. It is driven exactly once by the
FastAPI lifespan/startup hook in :mod:`app.main`, builds the model and restores
its checkpoint parameters, caches the loaded ``TransformerConfig``, and exposes
a small readiness API consumed by the HTTP layer and the instrumentation/SSE
layers:

* ``GET /health`` reads :func:`get_status` (``"loading"`` or ``"ready"``).
* ``POST /analyze`` refuses with ``503`` until :func:`is_ready` is ``True``.
* ``app.instrumentation`` / ``app.sse`` obtain the loaded objects through
  :func:`get_model_and_params` and :func:`get_config`.

Instrument, never reimplement (AAP 0.7.1)
-----------------------------------------
The model and its checkpoint loader come **entirely** from the unmodified
sibling ``gemma`` library, composed through ``from gemma import gm``. This
module does NOT reimplement model construction, tokenization, the forward
pass, or checkpoint loading -- it only orchestrates the canonical gemma idiom
verified against the repository's ``README.md`` and source::

    from gemma import gm
    model = gm.nn.Gemma3_4B(text_only=True)
    params = gm.ckpts.load_params(GEMMA_WEIGHTS_PATH, text_only=True)

Verified gemma anchors (read-only; never edited here)
-----------------------------------------------------
* ``gm.nn.Gemma3_4B`` -- ``gemma/gm/nn/_gemma.py``. ``text_only`` is an
  attribute of its parent ``_Gemma3Base`` whose ``__post_init__`` sets
  ``config.vision_encoder = None`` when ``text_only=True``, stripping the
  SigLiP vision encoder (a text-only telemetry workload saves memory/compute).
* ``gm.ckpts.load_params(path, *, params=None, donate=True, text_only=False,
  sharding=None, quantize=False, restore_concurrent_gb=None) -> Params`` --
  ``gemma/gm/ckpts/_checkpoint.py`` (Orbax ``StandardCheckpointer``). Returns
  the raw params tree; downstream code binds it as ``{'params': params}``.
* ``config.num_layers`` is a ``functools.cached_property`` equal to
  ``len(config.attention_types)`` -- ``gemma/gm/nn/_config.py`` -- which is
  **34** for Gemma 3 4B (``_NUM_LAYERS_GEMMA3_4B = 34``). The layer count is
  therefore derived **dynamically** from the loaded configuration and is never
  hardcoded -- in particular it must not be confused with the smaller Gemma 2B
  / Gemma 3 270M models, whose layer count differs.

Design notes
------------
* **Lazy heavy import.** ``gemma`` (and the JAX stack it pulls in) is imported
  **inside** :func:`load_model`, never at module import time, so importing
  ``app.model_loader`` for tests and tooling stays cheap and never triggers a
  JAX/Metal initialization.
* **Atomic readiness flip.** :data:`state` starts at ``"loading"`` and is
  flipped to ``"ready"`` **last**, only after the model, params, config, and
  layer count are all populated. A :class:`threading.Lock` makes the
  guard/load/flip atomic because the lifespan hook runs this blocking call in a
  worker thread.
* **Fail closed, not crash-loop.** A missing ``GEMMA_WEIGHTS_PATH`` or a load
  failure records :attr:`ModelState.error`, keeps the status at ``"loading"``
  (so ``/analyze`` keeps returning ``503`` rather than ``500``-looping), and
  re-raises so the operator sees it.
* **No persistent state.** Nothing is written to disk or a database; the only
  state is the in-memory loaded model held by the :data:`state` singleton.
* **No GPU vendor flags.** This module sets no device flags; it relies entirely
  on the JAX backend configured by the environment (Metal on the Apple Silicon
  target) and never enables or imports any non-Metal compute backend.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from app import config

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Readiness status (must match ``app.schemas.HealthResponse.status``)
# -----------------------------------------------------------------------------
STATUS_LOADING: str = "loading"
"""Status while the one-time model load has not yet completed (or has failed).
``GET /health`` reports this and ``POST /analyze`` returns ``503`` while it is
in effect. Must match exactly the ``"loading"`` literal in the immutable
``HealthResponse`` contract (AAP 0.1.2)."""

STATUS_READY: str = "ready"
"""Status once the model and parameters are fully loaded. The flip to this
value is the contract that lets ``/analyze`` begin streaming. Must match
exactly the ``"ready"`` literal in the immutable ``HealthResponse`` contract."""

Status = Literal["loading", "ready"]
"""Type alias for the two readiness states, mirroring the
``Literal["ready", "loading"]`` constraint on ``app.schemas.HealthResponse``."""


# -----------------------------------------------------------------------------
# Mutable model-state singleton
# -----------------------------------------------------------------------------
@dataclass
class ModelState:
  """In-memory holder for the loaded model and its readiness flag.

  A single module-level instance, :data:`state`, is mutated in place by
  :func:`load_model`. The :attr:`status` field is the readiness flag; every
  other field is ``None`` until a successful load populates it. The object
  holds no persistent state and is never serialized.

  Attributes:
    status: Readiness flag, :data:`STATUS_LOADING` or :data:`STATUS_READY`.
      Starts at :data:`STATUS_LOADING` and is flipped to :data:`STATUS_READY`
      only after a fully successful load.
    model: The constructed ``gm.nn.Gemma3_4B`` Flax module (``None`` until
      ready).
    params: The raw params tree returned by ``gm.ckpts.load_params`` (``None``
      until ready). Downstream code binds it as ``{'params': params}``.
    config: The loaded ``model.config`` (``gemma`` ``TransformerConfig``)
      exposing ``num_layers``, ``num_kv_heads``, ``head_dim``, etc. (``None``
      until ready).
    num_layers: Convenience cache of ``config.num_layers`` (== 34 for Gemma 3
      4B). Derived dynamically from the loaded configuration -- never
      hardcoded.
    error: Human-readable description of the most recent load failure,
      surfaced in logs and by accessors. ``None`` when no failure has occurred.
  """

  status: str = STATUS_LOADING
  model: Any | None = None
  params: Any | None = None
  config: Any | None = None
  num_layers: int | None = None
  error: str | None = None


state = ModelState()
"""Module-level singleton holding the loaded model and readiness flag. Imported
by ``app.main`` (for ``/health`` and the lifespan hook) and read indirectly,
through the accessors below, by ``app.instrumentation`` and ``app.sse``."""

# Serializes :func:`load_model`: makes the double-load guard, the load, and the
# atomic flip to "ready" a single critical section. The lifespan hook runs the
# blocking load in a worker thread, so a lock (not an asyncio primitive) is the
# correct guard here.
_load_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Readiness helpers
# -----------------------------------------------------------------------------
def is_ready() -> bool:
  """Return ``True`` once the model and parameters are fully loaded.

  Returns:
    ``True`` if :attr:`ModelState.status` equals :data:`STATUS_READY`,
    otherwise ``False`` (including while loading and after a failed load).
  """
  return state.status == STATUS_READY


def get_status() -> Status:
  """Return the current readiness status for the ``GET /health`` response.

  Returns:
    Either :data:`STATUS_LOADING` or :data:`STATUS_READY`.
  """
  return state.status  # type: ignore[return-value]


def _require_ready(accessor: str) -> None:
  """Raise ``RuntimeError`` if the model is not loaded yet.

  Shared precondition check for the accessors below so they fail with a single,
  consistent, actionable message (including the last load error, when present)
  instead of returning ``None`` to callers that assume a loaded model.

  Args:
    accessor: Name of the calling accessor, embedded in the error message.

  Raises:
    RuntimeError: If :attr:`ModelState.status` is not :data:`STATUS_READY`.
  """
  if state.status != STATUS_READY:
    detail = f" Last load error: {state.error}" if state.error else ""
    raise RuntimeError(
        f"{accessor}() called before the Gemma 3 4B model finished loading"
        f" (status={state.status!r}); the FastAPI lifespan hook must run"
        f" load_model() to completion first.{detail}"
    )


# -----------------------------------------------------------------------------
# One-time model load (invoked once by the FastAPI lifespan hook)
# -----------------------------------------------------------------------------
def load_model() -> None:
  """Load Gemma 3 4B exactly once and flip the readiness flag to ``"ready"``.

  Builds ``gm.nn.Gemma3_4B(text_only=True)``, restores its parameters from the
  Orbax checkpoint at ``settings.gemma_weights_path`` via
  ``gm.ckpts.load_params(..., text_only=True)``, caches ``model.config`` and
  the dynamically derived ``config.num_layers`` (34 for Gemma 3 4B), and only
  then flips :attr:`ModelState.status` to :data:`STATUS_READY`.

  This call is **synchronous and blocking** (the first load can take roughly
  20-40 seconds), so the lifespan hook must run it off the event loop -- for
  example via ``anyio.to_thread.run_sync`` or ``loop.run_in_executor`` -- to
  keep the server responsive (so ``/health`` can report ``"loading"``
  meanwhile). The ``gemma``/JAX import is performed lazily inside this function
  so merely importing this module never triggers a heavy JAX/Metal init.

  The load is idempotent: a second call after a successful load is a no-op. On
  any failure the status stays :data:`STATUS_LOADING` (so ``/analyze`` returns
  ``503`` rather than ``500``), :attr:`ModelState.error` is recorded, and the
  exception is re-raised for the caller to log.

  Raises:
    RuntimeError: If ``GEMMA_WEIGHTS_PATH`` (``settings.gemma_weights_path``)
      is empty/unset.
    Exception: Any error raised by ``gemma`` while constructing the model or
      restoring the checkpoint is re-raised after being recorded.
  """
  with _load_lock:
    # Idempotency guard: the model must load EXACTLY once.
    if state.status == STATUS_READY:
      logger.info("Gemma 3 4B already loaded; skipping reload (idempotent).")
      return

    # Validate the checkpoint path BEFORE importing gemma/JAX so a misconfigured
    # environment fails fast and cheaply, without a heavy import. Keep the
    # status at "loading" so the server keeps returning 503 instead of
    # crash-looping.
    weights_path = config.settings.gemma_weights_path
    if not weights_path or not weights_path.strip():
      message = (
          "GEMMA_WEIGHTS_PATH is not set. Export it to the absolute path of"
          " the downloaded Gemma 3 4B Orbax checkpoint directory before"
          " starting the backend (see backend/.env.example). The server will"
          " keep reporting status='loading' until a valid checkpoint path is"
          " provided."
      )
      state.error = message
      logger.error("Cannot load model: %s", message)
      raise RuntimeError(message)

    started_at = time.perf_counter()
    try:
      # Lazy heavy import -- kept out of module scope (see module docstring).
      from gemma import gm  # pylint: disable=import-outside-toplevel

      logger.info(
          "Loading Gemma 3 4B (text_only=True) from checkpoint: %s",
          weights_path,
      )

      # Compose the unmodified gemma library as-is (instrument, never
      # reimplement). text_only=True on BOTH calls strips/skips the SigLiP
      # vision encoder and its parameters for this text-only telemetry
      # workload.
      model = gm.nn.Gemma3_4B(text_only=True)
      params = gm.ckpts.load_params(weights_path, text_only=True)

      # Derive the layer count DYNAMICALLY from the loaded configuration; never
      # hardcode it (config.num_layers == 34 for Gemma 3 4B).
      cfg = model.config
      num_layers = cfg.num_layers

      # Populate every field first, then flip readiness LAST so no consumer
      # ever observes status == "ready" with a half-populated state.
      state.model = model
      state.params = params
      state.config = cfg
      state.num_layers = num_layers
      state.error = None
      state.status = STATUS_READY

      elapsed_s = time.perf_counter() - started_at
      logger.info(
          "Gemma 3 4B ready (status='ready'): num_layers=%d, loaded in %.2fs.",
          num_layers,
          elapsed_s,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Record the failure for /health and logs, keep the status at "loading"
      # so the API fails with 503 (not ready) rather than 500, and re-raise so
      # the lifespan hook can log it. Do NOT flip to "ready" on failure.
      state.error = str(exc)
      logger.exception(
          "Failed to load Gemma 3 4B; backend remains in 'loading' state."
      )
      raise


# -----------------------------------------------------------------------------
# Accessors used by app.instrumentation and app.sse
# -----------------------------------------------------------------------------
def get_model_and_params() -> tuple[Any, Any]:
  """Return the loaded ``(model, params)`` pair.

  Used by ``app.instrumentation`` (the eager per-layer harness) and ``app.sse``
  (the gemma sampler) once readiness has been established.

  Returns:
    A ``(model, params)`` tuple where ``model`` is the ``gm.nn.Gemma3_4B``
    instance and ``params`` is the raw params tree from
    ``gm.ckpts.load_params``.

  Raises:
    RuntimeError: If called before :func:`load_model` has completed
      successfully.
  """
  _require_ready("get_model_and_params")
  return state.model, state.params


def get_config() -> Any:
  """Return the loaded model configuration (``gemma`` ``TransformerConfig``).

  Exposes ``num_layers`` (34), ``num_kv_heads`` (4), ``head_dim`` (256), and
  the other dimensions that ``app.instrumentation`` and ``app.metrics`` read to
  drive the dynamic layer count and the KV-cache memory estimate.

  Returns:
    The loaded ``model.config`` object.

  Raises:
    RuntimeError: If called before :func:`load_model` has completed
      successfully.
  """
  _require_ready("get_config")
  return state.config


__all__ = [
    "STATUS_LOADING",
    "STATUS_READY",
    "Status",
    "ModelState",
    "state",
    "is_ready",
    "get_status",
    "load_model",
    "get_model_and_params",
    "get_config",
]
