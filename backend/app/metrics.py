"""System and device telemetry sampling for the Gemma Compute Monitor.

This module supplies the host- and device-level numbers that fill every
**SSE layer event** streamed by ``POST /analyze`` (see the immutable
``LayerEvent`` contract in :mod:`app.schemas`): ``cpu_pct`` and
``memory_used_gb`` via :mod:`psutil`, ``gpu_pct`` via a short-lived macOS
``powermetrics`` subprocess, and the derived ``kv_cache_gb`` /
``activation_gb`` memory estimates. It is called per transformer layer by
``app.instrumentation`` and once more by ``app.sse`` while assembling each
event.

Design contract (AAP §0.5.2 / §0.7.1)
-------------------------------------
* **Dependency-light & gemma/JAX-free.** Only :mod:`psutil` and the standard
  library (``subprocess`` / ``re`` / ``shutil`` / ``logging``) are imported.
  The module never imports ``gemma`` or ``jax`` and never loads the model; it
  consumes plain integers and tensor shapes that callers read from the loaded
  configuration object. This keeps it trivially unit-testable in a sandbox
  with no model, no GPU, and no root privileges.
* **``powermetrics`` resilience.** GPU sampling uses a **200 ms timeout** and
  falls back to **``0.0``** on *any* failure. ``powermetrics`` is a macOS
  built-in that requires root, so when the backend runs without ``sudo`` (or
  on a non-macOS host such as the Linux sandbox) :func:`gpu_pct` correctly and
  quietly returns ``0.0`` — the frontend GPU waveform simply renders a flat
  line. This is the documented resilience default, not an error, so the
  failure path logs at ``DEBUG`` and :func:`gpu_pct` never raises.
* **Grouped-query attention (AAP §0.7.2).** The KV-cache estimate scales with
  ``num_kv_heads`` (== 4 for Gemma 3 4B), **not** ``num_heads`` (== 8); using
  ``num_heads`` would double the estimate. All model dimensions
  (``num_kv_heads``, ``head_dim``, ``num_layers``, ``embed_dim``) are read from
  the loaded configuration and are **never hardcoded**.

The dict produced by :func:`sample_layer_metrics` uses exactly the field names
of ``app.schemas.LayerEvent`` so callers can validate it with
``schemas.LayerEvent(**event)``; this module deliberately does **not** import
``app.schemas`` so it stays import-cheap and side-effect free.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import Any

import psutil

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
POWERMETRICS_TIMEOUT_S: float = 0.2
"""Wall-clock timeout (seconds) for the ``powermetrics`` subprocess — 200 ms.
Kept deliberately short so a single SSE layer event is never delayed waiting on
GPU telemetry; on timeout :func:`gpu_pct` returns :data:`GPU_FALLBACK`."""

GPU_FALLBACK: float = 0.0
"""GPU-utilization value returned whenever ``powermetrics`` is unavailable,
times out, errors, or produces unparseable output. ``0.0`` is the documented
resilience default (AAP §0.7.1)."""

MODEL_WEIGHTS_GB: float = 8.5
"""Static estimate (GB) of the resident Gemma 3 4B weight footprint. Exported
for the frontend "Unified Memory Breakdown" panel's fixed "Model Weights"
segment; the SSE ``LayerEvent`` itself carries only ``kv_cache_gb``,
``activation_gb`` and ``memory_used_gb`` — never the static weights."""

BYTES_PER_BF16: int = 2
"""Bytes per element for ``bfloat16`` activation and KV-cache tensors. The
gemma cache is initialized with ``dtype=jnp.bfloat16``
(``gemma/gm/nn/_config.py:init_cache``), so the memory math uses 2 bytes."""

_KV_TENSORS_PER_LAYER: int = 2
"""The KV cache stores two tensors per layer — Keys and Values — so the
per-layer footprint is multiplied by 2 (AAP §0.5.2; gemma ``init_cache``)."""


# -----------------------------------------------------------------------------
# CPU and RAM sampling (psutil)
# -----------------------------------------------------------------------------
def cpu_pct() -> float:
  """Return system-wide CPU utilization as a percentage in ``[0.0, 100.0]``.

  Uses ``psutil.cpu_percent(interval=None)``, a **non-blocking** sample taken
  relative to the previous call. The instrumentation harness calls this once
  per transformer layer, so successive non-blocking samples track utilization
  across the forward pass without ever stalling the event loop.

  Note:
    The *first* call after process start returns ``0.0`` because there is no
    prior reference point; subsequent calls return meaningful deltas. This is
    expected and harmless for telemetry.

  Returns:
    CPU utilization percentage as a ``float``.
  """
  return float(psutil.cpu_percent(interval=None))


def memory_used_gb() -> float:
  """Return used system memory in gigabytes (decimal, base-10 ``1e9``).

  Reads ``psutil.virtual_memory().used`` (bytes) and divides by ``1e9`` to
  report **decimal** gigabytes, matching the units the frontend Memory
  Breakdown panel renders against the 48 GB Apple-Silicon unified-memory pool.

  Returns:
    Used memory in GB as a ``float``.
  """
  return psutil.virtual_memory().used / 1e9


# -----------------------------------------------------------------------------
# GPU sampling (macOS powermetrics; 200 ms timeout; 0.0 fallback)
# -----------------------------------------------------------------------------
# Command taking exactly one GPU-power sample. ``-n 1`` requests a single
# sample and ``-i 200`` sets a 200 ms sample interval (paired with the 200 ms
# subprocess timeout in :data:`POWERMETRICS_TIMEOUT_S`).
_POWERMETRICS_CMD: tuple[str, ...] = (
    "powermetrics",
    "--samplers",
    "gpu_power",
    "-n",
    "1",
    "-i",
    "200",
)

# Tolerant matcher for the "GPU [HW] active residency: NN.NN%" line emitted by
# powermetrics. The optional "HW" token and flexible whitespace absorb minor
# formatting differences across macOS releases.
_GPU_RESIDENCY_RE = re.compile(
    r"GPU(?:\s+HW)?\s+active\s+residency:\s*([0-9.]+)%",
    re.IGNORECASE,
)


def gpu_pct() -> float:
  """Return Metal GPU active-residency percentage, or ``0.0`` on any failure.

  Behavior:
    * If the ``powermetrics`` binary is not on ``PATH`` (e.g. a non-macOS
      host) it returns :data:`GPU_FALLBACK` (``0.0``) immediately, without
      spawning a subprocess.
    * Otherwise it runs ``powermetrics --samplers gpu_power -n 1 -i 200`` with
      a :data:`POWERMETRICS_TIMEOUT_S` (200 ms) timeout, parses "GPU active
      residency" from stdout, and returns it as a ``float``.
    * On *any* failure — timeout, non-zero exit, missing binary, unparseable
      output, or missing root privileges — it returns :data:`GPU_FALLBACK`.

  ``powermetrics`` requires root on macOS, so when the backend runs without
  ``sudo`` this function will (correctly) return ``0.0``; that is the
  documented resilience default (AAP §0.7.1), surfaced by the frontend GPU
  waveform as a flat line. The expected no-root / non-macOS case is logged at
  ``DEBUG`` to stay quiet, and this function **never raises** — it always
  returns a ``float`` so the SSE stream is never interrupted.

  Returns:
    GPU active-residency percentage as a ``float``, or ``0.0`` on any failure.
  """
  if shutil.which("powermetrics") is None:
    logger.debug(
        "powermetrics not found on PATH; returning GPU fallback %.1f",
        GPU_FALLBACK,
    )
    return GPU_FALLBACK

  try:
    completed = subprocess.run(
        list(_POWERMETRICS_CMD),
        capture_output=True,
        text=True,
        timeout=POWERMETRICS_TIMEOUT_S,
        check=False,
    )
    match = _GPU_RESIDENCY_RE.search(completed.stdout or "")
    if match is not None:
      return float(match.group(1))
    logger.debug(
        "powermetrics produced no GPU residency match; using fallback %.1f",
        GPU_FALLBACK,
    )
    return GPU_FALLBACK
  # ANY failure collapses to the documented 0.0 fallback. ``Exception`` is
  # listed explicitly so the function can NEVER raise out into the SSE stream;
  # powermetrics needing root is the common case, so this is DEBUG, not ERROR.
  except (  # pylint: disable=broad-exception-caught
      subprocess.TimeoutExpired,
      subprocess.SubprocessError,
      FileNotFoundError,
      ValueError,
      Exception,
  ) as exc:
    logger.debug(
        "powermetrics GPU sampling failed (%s); using fallback %.1f",
        exc,
        GPU_FALLBACK,
    )
    return GPU_FALLBACK


# -----------------------------------------------------------------------------
# Memory math derived from the loaded configuration
# -----------------------------------------------------------------------------
def kv_cache_gb(
    seq_len: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
) -> float:
  """Estimate the key/value attention-cache footprint in gigabytes.

  Implements, exactly::

      kv_cache_gb = (
          seq_len * num_kv_heads * head_dim * num_layers
          * 2               # one Key tensor + one Value tensor per layer
          * BYTES_PER_BF16  # bf16 == 2 bytes / element
          / 1e9             # bytes -> decimal GB
      )

  The cache scales with **``num_kv_heads``**, not ``num_heads`` — Gemma 3 4B
  uses grouped-query attention with ``num_kv_heads == 4`` while
  ``num_heads == 8`` (AAP §0.7.2). Sourcing the head count from the loaded
  configuration mirrors ``gemma/gm/nn/_config.py:init_cache``, which builds a
  per-layer cache keyed ``layer_{i}`` from ``num_kv_heads`` / ``head_dim`` at
  ``dtype=jnp.bfloat16``; using ``num_heads`` would overestimate by 2x.

  Args:
    seq_len: Sequence length (number of cached token positions).
    num_kv_heads: Grouped-query KV head count from ``config.num_kv_heads``
      (``4`` for Gemma 3 4B).
    head_dim: Per-head dimension from ``config.head_dim`` (``256`` for
      Gemma 3 4B).
    num_layers: Transformer layer count from ``config.num_layers`` (``34`` for
      Gemma 3 4B).

  Returns:
    Estimated KV-cache size in GB as a ``float``.
  """
  return (
      seq_len
      * num_kv_heads
      * head_dim
      * num_layers
      * _KV_TENSORS_PER_LAYER
      * BYTES_PER_BF16
      / 1e9
  )


def kv_cache_gb_from_config(seq_len: int, config: Any) -> float:
  """Estimate KV-cache GB by reading the dimensions off a loaded ``config``.

  Convenience wrapper around :func:`kv_cache_gb` so callers can pass the loaded
  gemma ``TransformerConfig`` directly. The grouped-query head count, per-head
  dimension, and layer count are read from the configuration and are **never
  hardcoded**, so the estimate automatically tracks whichever Gemma variant is
  loaded.

  Args:
    seq_len: Sequence length (number of cached token positions).
    config: The loaded model configuration object exposing ``num_kv_heads``,
      ``head_dim`` and ``num_layers`` (e.g. gemma's ``TransformerConfig``).

  Returns:
    Estimated KV-cache size in GB as a ``float``.
  """
  return kv_cache_gb(
      seq_len,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      num_layers=config.num_layers,
  )


def activation_gb(
    *,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    bytes_per_elem: int = BYTES_PER_BF16,
) -> float:
  """Estimate the current layer's activation (hidden-state) footprint in GB.

  A single-layer hidden-state estimate computed directly from the layer's
  intermediate tensor shape (AAP §0.5.2)::

      activation_gb = batch_size * seq_len * embed_dim * bytes_per_elem / 1e9

  This deliberately models the dominant ``(batch, seq, embed)`` hidden-state
  tensor; it is not an exact accounting of every transient buffer. Callers pass
  the actual dimensions of the layer output (e.g. derived from ``x.shape``) so
  the value tracks the real tensor being processed.

  Args:
    batch_size: Batch dimension of the hidden-state tensor.
    seq_len: Sequence-length dimension of the hidden-state tensor.
    embed_dim: Embedding/model dimension of the hidden-state tensor.
    bytes_per_elem: Bytes per element; defaults to :data:`BYTES_PER_BF16`
      (bf16, 2 bytes).

  Returns:
    Estimated activation size in GB as a ``float``.
  """
  return batch_size * seq_len * embed_dim * bytes_per_elem / 1e9


# -----------------------------------------------------------------------------
# Convenience sampler assembling a LayerEvent-shaped dict
# -----------------------------------------------------------------------------
def _hidden_state_dims(
    x_shape: tuple[int, ...],
    config: Any,
    seq_len: int,
) -> tuple[int, int, int]:
  """Resolve ``(batch_size, seq_len, embed_dim)`` from a layer's tensor shape.

  The instrumented transformer hidden state is normally rank-3
  ``(batch, seq, embed)`` but may be flattened to rank-2 ``(seq, embed)``.
  Any other rank falls back to ``batch_size=1`` with the caller-supplied
  ``seq_len`` and the configuration's ``embed_dim`` so the activation estimate
  stays well-defined.

  Args:
    x_shape: Shape tuple of the current layer output.
    config: Loaded configuration exposing ``embed_dim`` (used as a fallback).
    seq_len: Caller-supplied sequence length used when ``x_shape`` is neither
      rank-2 nor rank-3.

  Returns:
    A ``(batch_size, seq_len, embed_dim)`` tuple of ints.
  """
  if len(x_shape) == 3:
    batch_size, shape_seq_len, embed_dim = x_shape
    return int(batch_size), int(shape_seq_len), int(embed_dim)
  if len(x_shape) == 2:
    shape_seq_len, embed_dim = x_shape
    return 1, int(shape_seq_len), int(embed_dim)
  return 1, int(seq_len), int(config.embed_dim)


def sample_layer_metrics(
    *,
    layer_index: int,
    seq_len: int,
    config: Any,
    x_shape: tuple[int, ...],
    elapsed_ms: float,
) -> dict[str, Any]:
  """Assemble one ``LayerEvent``-shaped metrics dict for a completed layer.

  Samples CPU, RAM and GPU utilization and computes the KV-cache and activation
  memory estimates, returning a plain ``dict`` whose keys match
  ``app.schemas.LayerEvent`` **exactly**: ``layer``, ``gpu_pct``, ``cpu_pct``,
  ``memory_used_gb``, ``kv_cache_gb``, ``activation_gb`` and ``elapsed_ms``.
  Keeping assembly here keeps ``app.instrumentation`` / ``app.sse`` thin; those
  callers can validate the result via ``schemas.LayerEvent(**event)`` (this
  module intentionally does not import the schema, to stay dependency-light).

  Args:
    layer_index: Zero-based index of the transformer layer that just finished.
    seq_len: Sequence length used for the KV-cache estimate.
    config: Loaded model configuration (read for ``num_kv_heads``,
      ``head_dim``, ``num_layers`` and ``embed_dim``).
    x_shape: Shape of the current layer output, used for the activation
      estimate.
    elapsed_ms: Wall-clock milliseconds elapsed for this layer's computation,
      measured immediately after the per-layer synchronization barrier.

  Returns:
    A ``dict`` with exactly the seven ``LayerEvent`` keys.
  """
  batch_size, shape_seq_len, embed_dim = _hidden_state_dims(
      x_shape, config, seq_len
  )
  return {
      "layer": int(layer_index),
      "gpu_pct": gpu_pct(),
      "cpu_pct": cpu_pct(),
      "memory_used_gb": memory_used_gb(),
      "kv_cache_gb": kv_cache_gb_from_config(seq_len, config),
      "activation_gb": activation_gb(
          batch_size=batch_size,
          seq_len=shape_seq_len,
          embed_dim=embed_dim,
      ),
      "elapsed_ms": float(elapsed_ms),
  }


# Explicit public API. ``sample_layer_metrics`` is the primary entry point for
# ``app.instrumentation`` / ``app.sse``; the individual samplers and the memory
# constants/helpers are exported for direct use and unit testing.
__all__ = [
    "POWERMETRICS_TIMEOUT_S",
    "GPU_FALLBACK",
    "MODEL_WEIGHTS_GB",
    "BYTES_PER_BF16",
    "cpu_pct",
    "memory_used_gb",
    "gpu_pct",
    "kv_cache_gb",
    "kv_cache_gb_from_config",
    "activation_gb",
    "sample_layer_metrics",
]
