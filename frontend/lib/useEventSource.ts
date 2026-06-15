/**
 * Gemma Compute Monitor — React hook that streams /analyze telemetry.
 *
 * TRANSPORT DECISION (binding): the native browser `EventSource` API is
 * GET-only and cannot send a POST body, but the IMMUTABLE contract is
 * `POST /analyze { prompt }`. The contract wins, so this hook consumes the SSE
 * stream with the native Fetch streaming API (`fetch` POST +
 * `response.body.getReader()` + `TextDecoder`). No third-party SSE library is
 * used (explicit AAP constraint). The hook is still named `useEventSource` per
 * the required module path.
 *
 * Events are classified by PAYLOAD CONTENT (robust to whether the backend set
 * an optional `event: layer` / `event: final` SSE name): a payload with
 * `done === true` or a `per_token_ms` array is the FinalEvent; anything else is
 * a LayerEvent. See `isFinalEvent` in ./types.
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getApiBaseUrl, MODEL_WARMING_MESSAGE } from "./api";
import {
  isFinalEvent,
  type FinalEvent,
  type LayerEvent,
  type StreamStatus,
  type TelemetryEvent,
} from "./types";

/** Message surfaced when a second analysis is rejected by the single-flight guard (HTTP 429). */
export const BUSY_MESSAGE = "An analysis is already in progress. Please wait for it to finish.";

/** Delay (ms) before surfacing the "model warming up" notice if no event has streamed yet. */
const WARMING_DELAY_MS = 5000;

/** Observable state + controls returned by useEventSource. */
export interface UseEventSourceResult {
  /** Lifecycle status of the current stream. */
  status: StreamStatus;
  /** All LayerEvents received so far, in arrival order. */
  layers: LayerEvent[];
  /** The most recently received LayerEvent, or null. */
  latestLayer: LayerEvent | null;
  /** The FinalEvent (with per_token_ms) once the stream completes, else null. */
  finalEvent: FinalEvent | null;
  /** Human-readable error/notice message (offline, 503 warming, 429 busy, parse/network), else null. */
  error: string | null;
  /** True while waiting >5s for the first streamed event (first-run warm-up). */
  warming: boolean;
  /** Begin a new analysis for `prompt` (aborts any in-flight stream first). */
  start: (prompt: string) => void;
  /** Abort the in-flight stream (client disconnect the backend honors). */
  stop: () => void;
  /** Reset all state back to idle (also stops any in-flight stream). */
  reset: () => void;
}

// ---------------------------------------------------------------------------
// Pure SSE-parsing helpers (exported for unit testing; no React, no DOM state).
// ---------------------------------------------------------------------------

/**
 * Split an accumulated decoded-text buffer into complete SSE frames.
 * Frames are separated by a blank line ("\n\n"). The trailing partial frame
 * (everything after the last separator) is returned as `rest` to be carried
 * into the next read. Callers should normalize CRLF to LF before calling.
 */
export function splitSseFrames(buffer: string): { frames: string[]; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  return { frames: parts, rest };
}

/**
 * Extract the JSON `data` payload from a single SSE frame. Concatenates all
 * `data:` lines (SSE joins multiple data lines with "\n"); ignores comments
 * (lines starting with ":") and other fields (`event:`, `id:`, `retry:`).
 * Returns the joined data string, or null if the frame has no data lines.
 */
export function parseSseData(frame: string): string | null {
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.length === 0 || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("data:")) {
      // Strip "data:" and at most one leading space, per the SSE spec.
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    // event:, id:, retry: lines are intentionally ignored — we classify by payload.
  }
  if (dataLines.length === 0) {
    return null;
  }
  return dataLines.join("\n");
}

/**
 * Parse one SSE frame into a TelemetryEvent, or null if the frame carries no
 * valid JSON data payload.
 */
export function parseTelemetryFrame(frame: string): TelemetryEvent | null {
  const data = parseSseData(frame);
  if (data === null) {
    return null;
  }
  try {
    return JSON.parse(data) as TelemetryEvent;
  } catch {
    return null;
  }
}

/**
 * Required numeric fields of the immutable LayerEvent contract (mirrors
 * backend/app/schemas.py). The backend always emits all seven, but a
 * malformed/empty frame (e.g. `{}`) must never be appended as a "layer" with
 * `undefined` fields — that would corrupt the bar chart, waveform, and memory
 * panels downstream.
 */
const LAYER_NUMERIC_FIELDS: readonly (keyof LayerEvent)[] = [
  "layer",
  "gpu_pct",
  "cpu_pct",
  "memory_used_gb",
  "kv_cache_gb",
  "activation_gb",
  "elapsed_ms",
];

/**
 * Runtime guard: a parsed payload is a well-formed LayerEvent only when every
 * required field is a FINITE number (rejects missing/undefined, null, NaN,
 * ±Infinity, and non-numeric types). handleFrame uses this to reject malformed
 * non-final frames before they ever enter panel state.
 */
export function isValidLayerEvent(payload: unknown): payload is LayerEvent {
  if (typeof payload !== "object" || payload === null) {
    return false;
  }
  const record = payload as Record<string, unknown>;
  return LAYER_NUMERIC_FIELDS.every(
    (field) =>
      typeof record[field] === "number" && Number.isFinite(record[field]),
  );
}

// ---------------------------------------------------------------------------
// The hook.
// ---------------------------------------------------------------------------

export function useEventSource(): UseEventSourceResult {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [layers, setLayers] = useState<LayerEvent[]>([]);
  const [latestLayer, setLatestLayer] = useState<LayerEvent | null>(null);
  const [finalEvent, setFinalEvent] = useState<FinalEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warming, setWarming] = useState<boolean>(false);

  // Mutable refs that must survive re-renders and be readable from async code.
  const controllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef<boolean>(true);
  const warmingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const userStoppedRef = useRef<boolean>(false);

  const clearWarmingTimer = useCallback(() => {
    if (warmingTimerRef.current !== null) {
      clearTimeout(warmingTimerRef.current);
      warmingTimerRef.current = null;
    }
  }, []);

  const abortInFlight = useCallback(() => {
    if (controllerRef.current !== null) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    userStoppedRef.current = true;
    clearWarmingTimer();
    abortInFlight();
    if (mountedRef.current) {
      setWarming(false);
      // Only downgrade to idle if we were actively streaming (not done/error).
      setStatus((prev) => (prev === "streaming" ? "idle" : prev));
    }
  }, [abortInFlight, clearWarmingTimer]);

  const reset = useCallback(() => {
    userStoppedRef.current = true;
    clearWarmingTimer();
    abortInFlight();
    if (mountedRef.current) {
      setStatus("idle");
      setLayers([]);
      setLatestLayer(null);
      setFinalEvent(null);
      setError(null);
      setWarming(false);
    }
  }, [abortInFlight, clearWarmingTimer]);

  const start = useCallback(
    (prompt: string) => {
      // Abort any previous run and reset observable state for a fresh stream.
      abortInFlight();
      clearWarmingTimer();
      userStoppedRef.current = false;

      const base = getApiBaseUrl();
      if (!base) {
        setStatus("error");
        setError("NEXT_PUBLIC_API_URL is not configured.");
        return;
      }

      const controller = new AbortController();
      controllerRef.current = controller;

      setStatus("streaming");
      setLayers([]);
      setLatestLayer(null);
      setFinalEvent(null);
      setError(null);
      setWarming(false);

      // Surface the "warming up" notice if no event has streamed within 5s.
      warmingTimerRef.current = setTimeout(() => {
        if (mountedRef.current) {
          setWarming(true);
        }
      }, WARMING_DELAY_MS);

      const run = async () => {
        try {
          const res = await fetch(`${base}/analyze`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
            },
            body: JSON.stringify({ prompt }),
            signal: controller.signal,
          });

          if (res.status === 503) {
            clearWarmingTimer();
            if (mountedRef.current) {
              setStatus("error");
              setError(MODEL_WARMING_MESSAGE);
              setWarming(false);
            }
            return;
          }
          if (res.status === 429) {
            clearWarmingTimer();
            if (mountedRef.current) {
              setStatus("error");
              setError(BUSY_MESSAGE);
              setWarming(false);
            }
            return;
          }
          if (!res.ok || res.body === null) {
            clearWarmingTimer();
            if (mountedRef.current) {
              setStatus("error");
              setError(`Request failed with status ${res.status}`);
              setWarming(false);
            }
            return;
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          const handleFrame = (frame: string): boolean => {
            const payload = parseTelemetryFrame(frame);
            if (payload === null) {
              return false;
            }
            if (isFinalEvent(payload)) {
              // First real event clears the warm-up notice.
              clearWarmingTimer();
              if (mountedRef.current) {
                setWarming(false);
                setFinalEvent(payload);
                setStatus("done");
              }
              return true; // signal completion
            }
            // Defensive contract validation (F5): a non-final payload is
            // accepted as a LayerEvent only when every required numeric field
            // is present and finite. Malformed frames (e.g. `{}`) are ignored
            // so they never corrupt panel state with undefined fields.
            if (!isValidLayerEvent(payload)) {
              return false;
            }
            // A valid layer event is also the first real event when warming.
            clearWarmingTimer();
            if (mountedRef.current) {
              setWarming(false);
              setLayers((prev) => [...prev, payload]);
              setLatestLayer(payload);
            }
            return false;
          };

          // Read loop — never blocks the main thread (awaits microtasks).
          for (;;) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
            const { frames, rest } = splitSseFrames(buffer);
            buffer = rest;
            let finished = false;
            for (const frame of frames) {
              if (handleFrame(frame)) {
                finished = true;
                break;
              }
            }
            if (finished) {
              await reader.cancel();
              controllerRef.current = null;
              return;
            }
          }

          // Flush any trailing buffered frame after the stream closes.
          buffer += decoder.decode();
          buffer = buffer.replace(/\r\n?/g, "\n");
          for (const frame of buffer.split("\n\n")) {
            if (handleFrame(frame)) {
              break;
            }
          }

          controllerRef.current = null;
          // If the stream ended without a final event and no error, mark done.
          if (mountedRef.current) {
            setStatus((prev) => (prev === "streaming" ? "done" : prev));
          }
        } catch (err) {
          clearWarmingTimer();
          controllerRef.current = null;
          // A user-initiated abort is not an error.
          if (userStoppedRef.current || (err instanceof DOMException && err.name === "AbortError")) {
            return;
          }
          if (mountedRef.current) {
            setStatus("error");
            setError(err instanceof Error ? err.message : "Streaming failed");
            setWarming(false);
          }
        }
      };

      void run();
    },
    [abortInFlight, clearWarmingTimer],
  );

  // Abort any in-flight stream on unmount (StrictMode-safe: abort is idempotent).
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearWarmingTimer();
      abortInFlight();
    };
  }, [abortInFlight, clearWarmingTimer]);

  return {
    status,
    layers,
    latestLayer,
    finalEvent,
    error,
    warming,
    start,
    stop,
    reset,
  };
}

export default useEventSource;
