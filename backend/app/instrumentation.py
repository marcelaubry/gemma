"""Eager, per-transformer-layer instrumentation harness for Gemma 3 4B.

This module is the core of the Gemma Compute Monitor backend. It drives a
Gemma 3 4B forward pass over a prompt **one transformer layer at a time**,
synchronizing each layer's device computation before sampling host-side
telemetry, so that exactly **one telemetry event is emitted per transformer
layer**.

Instrument, never reimplement (AAP §0.7.1)
------------------------------------------
The harness **composes the unmodified ``gemma`` library**: it reuses the loaded
model's own ``Block`` submodules, embedder, and final norm together with the
restored checkpoint weights, and only *orchestrates* the per-layer loop
externally. The attention math, RoPE, normalization, and KV logic are entirely
gemma's (``Block`` / ``Attention`` / ``RMSNorm``); none of it is rewritten here.

Why an eager harness is required
--------------------------------
gemma's ``Transformer.__call__`` is JIT-compiled as a single XLA program
(``nn.jit`` plus a flatten/unflatten batch-dim wrapper). Host Python cannot run
*between* layers inside that one compiled program, so per-layer telemetry
cannot be sampled from within it. Instead, this module **mirrors**
``Transformer._apply_attention`` *outside* the jit boundary: it binds the model
to its parameters with ``model.bind({'params': params})`` and calls each
``Block`` eagerly, forcing the layer's computation to finish with
``jax.block_until_ready(x)`` before sampling metrics and yielding an event.

Synchronization primitive (AAP §0.7.2)
--------------------------------------
The barrier is ``jax.block_until_ready(x)``, which waits until an array's device
computation has actually completed — the correct primitive for forcing
per-layer compute to finish before host metrics are read. JAX's ordered
side-effect synchronization API (which flushes debug-print-style side effects
rather than device computation) is deliberately **not** used here, because it
would not guarantee a layer's math is finished.

Dynamic layer count (AAP §0.7.2)
--------------------------------
The number of events equals ``config.num_layers``, read dynamically from the
loaded configuration (== 34 for Gemma 3 4B). The count is **never** hardcoded;
it is also never confused with the smaller Gemma 2B / Gemma 3 270M variants,
whose layer counts differ.

Execution model
---------------
``run_layers`` is intentionally a **synchronous generator**: ``app.sse`` runs it
in a worker thread (via an executor) and bridges its items into the async SSE
stream, so the event loop is never blocked by the blocking JAX calls. The
harness relies entirely on the ambient JAX backend configured by the
environment (the Metal backend on the Apple-Silicon target) and sets no
device-vendor flags.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from app import metrics

logger = logging.getLogger(__name__)


def _tokenize(model: Any, prompt: str) -> jnp.ndarray:
  """Encode ``prompt`` into a ``[1, L]`` int32 token array (gemma tokenizer).

  The tokenizer is resolved automatically from the model's own metadata
  (``model.INFO.tokenizer_version``) — there is no hardcoded tokenizer path. For
  Gemma 3 4B this resolves to the Gemma 3 SentencePiece tokenizer. The BOS token
  is prepended (``add_bos=True``) to match how gemma itself encodes a prompt
  before a forward pass.

  Args:
    model: The (unbound) ``gm.nn.Gemma3_4B`` module instance. Only its
      ``INFO.tokenizer_version`` class attribute is read here.
    prompt: The user prompt to encode. Must be a non-empty string.

  Returns:
    A JAX ``int32`` array of shape ``[1, L]`` (batch dimension of 1, ``L`` token
    ids), ready to feed into ``_encode_and_get_inputs``.

  Raises:
    ValueError: If ``prompt`` is not a non-empty string.
  """
  if not isinstance(prompt, str) or not prompt:
    raise ValueError("prompt must be a non-empty string.")

  # Lazy import keeps `from gemma import gm` (and the heavy JAX/Flax stack it
  # pulls in) out of module-import time, so importing this module for tooling or
  # inspection never forces a full gemma initialization. The gemma library is
  # composed as-is; the tokenizer is resolved, never reimplemented.
  from gemma import gm  # pylint: disable=import-outside-toplevel

  tokenizer = gm.text.Tokenizer.from_version(model.INFO.tokenizer_version)
  token_ids = tokenizer.encode(prompt, add_bos=True)

  # Add the batch dimension (batch size == 1) and pin the dtype to int32 so the
  # array satisfies gemma's typechecked `Int['B L']` token contract.
  tokens = jnp.asarray([token_ids], dtype=jnp.int32)
  logger.debug(
      "Tokenized prompt into %d token(s) (shape=%s).",
      len(token_ids),
      tuple(int(d) for d in tokens.shape),
  )
  return tokens


def run_layers(
    model: Any,
    params: Any,
    config: Any,
    prompt: str,
) -> Iterator[dict]:
  """Eagerly run the Gemma 3 4B forward pass, yielding one event per layer.

  Mirrors gemma's ``Transformer._apply_attention`` loop *outside* the ``nn.jit``
  boundary so host telemetry can be sampled between layers. The model is bound
  to its parameters for eager execution, the prompt is encoded with gemma's own
  tokenizer and ``_encode_and_get_inputs`` (positions and the attention mask are
  computed by gemma, never reimplemented here), and then each ``Block`` is
  invoked one at a time. After every layer the output is synchronized with
  ``jax.block_until_ready`` before metrics are sampled, so no event is ever
  emitted before its layer's device computation has completed (no speculative or
  pre-emitted events — AAP §0.1.1).

  Exactly ``config.num_layers`` events are produced (== 34 for Gemma 3 4B), one
  per loop iteration; the count is derived dynamically from the loaded
  configuration and is never hardcoded.

  This is a **synchronous generator**. ``app.sse`` drives it from a worker
  thread and bridges each yielded dict into the async SSE stream, so the
  blocking JAX work never stalls the event loop.

  Args:
    model: The loaded (unbound) ``gm.nn.Gemma3_4B`` module instance, as returned
      by ``app.model_loader.get_model_and_params``.
    params: The restored checkpoint parameter tree. Bound as
      ``{'params': params}`` for eager submodule execution.
    config: The loaded model configuration (gemma ``TransformerConfig``)
      exposing ``num_layers``, ``num_kv_heads``, ``head_dim`` and ``embed_dim``.
    prompt: The user prompt to analyze.

  Yields:
    One ``dict`` per transformer layer with exactly the seven
    ``app.schemas.LayerEvent`` keys: ``layer``, ``gpu_pct``, ``cpu_pct``,
    ``memory_used_gb``, ``kv_cache_gb``, ``activation_gb`` and ``elapsed_ms``.
    ``layer`` is the zero-based layer index (frontend labels ``L1…Ln``).
  """
  logger.info(
      "Starting eager per-layer instrumentation over %d layer(s).",
      config.num_layers,
  )

  # Bind the model to its parameters so its submodules (embedder, blocks,
  # final_norm) and the private encode helper run EAGERLY, outside the single
  # compiled `Transformer.__call__` XLA program. This is the idiomatic Flax
  # pattern for invoking submodules step by step; the params tree shape
  # {'params': ...} matches gemma's own `model.apply({'params': params}, ...)`.
  bound = model.bind({"params": params})

  # Encode the prompt to a [1, L] token array, then reuse gemma's own
  # `_encode_and_get_inputs` to obtain the embeddings, positions and attention
  # mask. We deliberately do NOT recompute positions/mask ourselves.
  tokens = _tokenize(model, prompt)
  # pylint: disable=protected-access
  # `_encode_and_get_inputs` is gemma-internal, but it is exactly the encoding
  # step the harness must reuse to stay faithful to the real forward pass
  # (instrument, never reimplement). Invoking it through the bound module is the
  # documented approach; for text-only inputs there are no images, so it simply
  # encodes the tokens and derives positions/mask.
  inputs = bound._encode_and_get_inputs(tokens=tokens)
  # pylint: enable=protected-access

  x = inputs.embeddings
  seq_len = int(x.shape[1])
  logger.debug(
      "Prompt encoded: seq_len=%d, embedding shape=%s.",
      seq_len,
      tuple(int(d) for d in x.shape),
  )

  # Mirror `Transformer._apply_attention`: run each Block eagerly, synchronize,
  # sample metrics, and emit one event. The loop bound is the DYNAMIC layer
  # count from the loaded configuration — never a literal constant.
  for i in range(config.num_layers):
    t0 = time.perf_counter()

    # `Block.__call__(x, segment_pos, cache, attn_mask) -> (cache, outputs)`. We
    # pass cache=None for a single full-sequence forward pass over the prompt;
    # the returned per-layer cache is deliberately discarded (bound to ``_``)
    # because the event's KV-cache figure is an analytical estimate
    # (`metrics.kv_cache_gb_from_config`), not the materialized cache.
    _, x = bound.blocks[i](
        x,
        inputs.positions,
        None,
        inputs.attention_mask,
    )

    # Barrier: force THIS layer's device computation to finish before sampling
    # host metrics, so the telemetry reflects work that has actually completed.
    # `block_until_ready` returns the same array once its computation is done.
    x = jax.block_until_ready(x)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Assemble the event via the shared sampler so the dict keys match
    # `schemas.LayerEvent` exactly. `x_shape` drives the activation estimate and
    # the loaded `config` drives the grouped-query-attention KV-cache estimate.
    event = metrics.sample_layer_metrics(
        layer_index=i,
        seq_len=seq_len,
        config=config,
        x_shape=tuple(int(d) for d in x.shape),
        elapsed_ms=elapsed_ms,
    )
    logger.debug("Layer %d completed in %.3f ms.", i, elapsed_ms)
    yield event

  # Apply the final norm to keep the eager harness faithful to gemma's real
  # forward pass (which ends `_apply_attention` with `x = self.final_norm(x)`).
  # No event is emitted for this step; per-token generation is handled
  # separately by `app.sse` via the gemma sampler, so it is not duplicated here.
  x = bound.final_norm(x)
  jax.block_until_ready(x)
  logger.info(
      "Eager per-layer instrumentation complete: %d event(s) emitted.",
      config.num_layers,
  )


# Explicit public API. `run_layers` is the single entry point consumed by
# `app.sse`; `_tokenize` is a private helper and is intentionally not exported.
__all__ = [
    "run_layers",
]
