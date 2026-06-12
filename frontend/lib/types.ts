/**
 * Gemma Compute Monitor — TypeScript mirrors of the four IMMUTABLE backend
 * wire contracts.
 *
 * These interfaces are a byte-for-byte mirror of the Pydantic models in
 * `backend/app/schemas.py` (the single source of truth). Field names, types,
 * and order are IMMUTABLE per AAP §0.1.2 / §0.7.1 — do NOT add, rename,
 * remove, or reorder fields, and do NOT add JSON-key aliases.
 *
 * The contract shapes originate from the Gemma 3 4B model configuration in the
 * unmodified gemma library (gemma/gm/nn/_gemma.py, gemma/gm/nn/_config.py):
 * Gemma 3 4B has 34 transformer layers (_NUM_LAYERS_GEMMA3_4B = 34), and the
 * KV-cache estimate uses grouped-query attention (num_kv_heads = 4,
 * head_dim = 256). The backend emits one LayerEvent per transformer layer.
 */

/**
 * SSE layer event — emitted once per transformer layer after that layer's GPU
 * work is synchronized.
 *
 * `layer` is the 0-BASED transformer layer index (the backend emits `layer = i`
 * for i in range(config.num_layers), i.e. 0..33 for Gemma 3 4B). The UI must
 * display it as `L{layer + 1}` (L1…L34).
 */
export interface LayerEvent {
  layer: number;
  gpu_pct: number;
  cpu_pct: number;
  memory_used_gb: number;
  kv_cache_gb: number;
  activation_gb: number;
  elapsed_ms: number;
}

/**
 * SSE final event — emitted exactly once after all layer events, carrying the
 * per-token decode latencies (milliseconds). `done` is the literal `true`.
 */
export interface FinalEvent {
  per_token_ms: number[];
  done: true;
}

/** POST /analyze request body. */
export interface AnalyzeRequest {
  prompt: string;
}

/**
 * GET /health response. `model` is the literal "gemma-3-4b" on the wire, kept
 * as `string` here so the UI does not over-narrow.
 */
export interface HealthResponse {
  status: "ready" | "loading";
  model: string;
}

/**
 * Discriminated union of every telemetry payload that can arrive on the
 * /analyze SSE stream.
 */
export type TelemetryEvent = LayerEvent | FinalEvent;

/**
 * Streaming lifecycle status surfaced by the useEventSource hook.
 */
export type StreamStatus = "idle" | "streaming" | "done" | "error";

/**
 * Type guard: classify a telemetry payload by CONTENT (robust to whether the
 * backend set an optional `event: layer` / `event: final` SSE name). A payload
 * is the FinalEvent if `done === true` or it carries the `per_token_ms` array.
 */
export function isFinalEvent(event: TelemetryEvent): event is FinalEvent {
  return (
    (event as FinalEvent).done === true || "per_token_ms" in event
  );
}

/**
 * Type guard: a payload is a LayerEvent when it is not the final event.
 */
export function isLayerEvent(event: TelemetryEvent): event is LayerEvent {
  return !isFinalEvent(event);
}
