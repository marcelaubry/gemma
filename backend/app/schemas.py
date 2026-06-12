"""Pydantic v2 models for the four IMMUTABLE Gemma Compute Monitor contracts.

This module is the **single source of truth** for the wire format exchanged
between the FastAPI telemetry backend and the Railway-deployed Next.js
frontend. The four models below reproduce — *byte-for-byte* — the four JSON
contracts mandated by the Agent Action Plan (AAP §0.1.2 "Immutable interface
contracts" / §0.7.1). The TypeScript mirror in ``frontend/lib/types.ts`` is
written to match these field names exactly, and the backend test-suite under
``backend/tests/`` asserts directly against these models.

The four immutable contracts (reproduced verbatim from AAP §0.1.2):

    SSE layer event:
        { layer: int, gpu_pct: float, cpu_pct: float, memory_used_gb: float,
          kv_cache_gb: float, activation_gb: float, elapsed_ms: float }

    SSE final event:
        { per_token_ms: float[], done: true }

    POST /analyze request body:
        { prompt: string }

    GET /health response:
        { status: "ready" | "loading", model: "gemma-3-4b" }

ABSOLUTE RULE (AAP §0.7.1): the field names, their types, and the literal
values ``done: true`` and ``model: "gemma-3-4b"`` are **immutable**. Do NOT
rename, add, remove, or reorder fields, and do NOT introduce serialization
aliases that change the emitted JSON keys. The dict produced by
``model_dump()`` (and the JSON produced by ``model_dump_json()``) MUST contain
exactly the keys defined here. Any drift breaks both the frontend mirror
(``frontend/lib/types.ts``) and the mandated backend tests.

Dependency note: Pydantic v2 ships transitively with ``fastapi==0.111.0`` — no
additional dependency is introduced by this module. It intentionally imports
nothing from the rest of the backend package (the literal ``"gemma-3-4b"`` is
inlined rather than imported from ``app.config``) so the contracts stay
trivially importable by tests and tooling with zero side effects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LayerEvent(BaseModel):
  """Telemetry emitted over SSE once after each instrumented transformer layer.

  Exactly one ``LayerEvent`` is streamed after every transformer layer of the
  loaded Gemma 3 4B model finishes its device computation. The instrumentation
  harness calls ``jax.block_until_ready`` on the layer output *before* these
  metrics are sampled, so every value reflects work that has actually
  completed on-device — never speculative or pre-emitted state.

  Wire contract (AAP §0.1.2 — IMMUTABLE; do not alter field names/types/order)::

      { layer: int, gpu_pct: float, cpu_pct: float, memory_used_gb: float,
        kv_cache_gb: float, activation_gb: float, elapsed_ms: float }
  """

  layer: int
  """Zero-based index of the transformer layer that just completed. The total
    layer count is derived dynamically from the loaded model configuration
    (``config.num_layers`` == 34 for Gemma 3 4B), never hardcoded."""

  gpu_pct: float
  """Metal GPU active-residency percentage parsed from ``powermetrics``. Falls
    back to ``0.0`` when the subprocess is unavailable (e.g. missing root
    privileges or a non-macOS host)."""

  cpu_pct: float
  """System/process CPU utilization percentage sampled via ``psutil``."""

  memory_used_gb: float
  """Resident memory in gigabytes sampled via ``psutil``."""

  kv_cache_gb: float
  """Estimated key/value attention-cache footprint in gigabytes, derived from
    the loaded configuration's grouped-query-attention dimensions
    (``num_kv_heads``, ``head_dim``, ``num_layers``)."""

  activation_gb: float
  """Estimated activation-tensor footprint in gigabytes for the current
    layer, computed from its intermediate tensor shape."""

  elapsed_ms: float
  """Wall-clock time in milliseconds elapsed for this layer's computation,
    measured immediately after the per-layer synchronization barrier."""


class FinalEvent(BaseModel):
  """Terminal SSE event closing an ``/analyze`` stream with per-token timings.

  Emitted exactly once — after all :class:`LayerEvent` messages and after the
  gemma sampler decode loop completes — to deliver the per-token compute cost
  rendered by the frontend's "Per-Token Compute Cost" strip.

  Wire contract (AAP §0.1.2 — IMMUTABLE)::

      { per_token_ms: float[], done: true }
  """

  per_token_ms: list[float]
  """Per-token decode latencies in milliseconds: one entry per generated
    token, in generation order."""

  done: Literal[True] = True
  """Stream-completion sentinel. Declared as ``Literal[True]`` with a default
    of ``True`` so the field is always present and always serializes to the JSON
    boolean ``true`` (``model_dump_json()`` emits ``"done": true``). This both
    satisfies the immutable contract and documents the invariant that the value
    can never be ``false``."""


class AnalyzeRequest(BaseModel):
  """Request body for ``POST /analyze``.

  Wire contract (AAP §0.1.2 — IMMUTABLE)::

      { prompt: string }
  """

  prompt: str = Field(..., min_length=1)
  """The user prompt to analyze. Required and non-empty: ``min_length=1``
    rejects an empty string so the instrumented forward pass always receives
    real input. The serialized key remains exactly ``prompt`` and the value
    stays a required string."""


class HealthResponse(BaseModel):
  """Response body for ``GET /health``.

  Reports model-load readiness so the frontend can gate ``/analyze`` and show
  the "model warming up" resilience message while the one-time FastAPI
  startup load is still in progress. ``/analyze`` returns ``503`` while the
  status is ``"loading"``.

  Wire contract (AAP §0.1.2 — IMMUTABLE)::

      { status: "ready" | "loading", model: "gemma-3-4b" }
  """

  status: Literal["ready", "loading"]
  """Readiness flag: ``"ready"`` once the model and parameters are fully
    loaded, otherwise ``"loading"``. Constrained to exactly these two string
    values."""

  model: Literal["gemma-3-4b"] = "gemma-3-4b"
  """Static model identifier. Constrained to the immutable literal
    ``"gemma-3-4b"`` (mirrors ``config.MODEL_ID``) via ``Literal`` so Pydantic
    *rejects* any other value — e.g. ``HealthResponse(status="ready",
    model="wrong")`` raises ``ValidationError`` rather than silently accepting
    drift. The default keeps the field optional for callers while the type
    guarantees the serialized value is always exactly ``gemma-3-4b``, enforcing
    the immutable ``GET /health`` contract (AAP §0.1.2 / §0.7.1) byte-for-byte."""


# Explicit public API. Listing the four contract models keeps wildcard imports
# (and editor/tooling introspection) aligned with the immutable wire format.
# This does not affect serialization or field names.
__all__ = [
    "LayerEvent",
    "FinalEvent",
    "AnalyzeRequest",
    "HealthResponse",
]
