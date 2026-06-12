/**
 * Gemma Compute Monitor — backend API helpers: base-URL resolution, /health
 * polling, and the two verbatim resilience messages surfaced by the UI.
 *
 * Pure TypeScript (no UI, no third-party libraries). Uses only the native
 * Fetch API + AbortController (DOM types). The backend base URL comes from the
 * build-time-inlined `NEXT_PUBLIC_API_URL` (the ngrok HTTPS URL, or
 * http://localhost:8000 for local dev).
 */

import type { HealthResponse } from "./types";

/**
 * Verbatim resilience message shown when /health times out or errors (the
 * backend is unreachable). Reproduced EXACTLY from AAP §0.5.3 — the dash is an
 * em-dash (U+2014). Do not alter the wording or punctuation.
 */
export const BACKEND_OFFLINE_MESSAGE =
  "Backend offline — start the local server and update the ngrok URL in Railway";

/**
 * Verbatim resilience message shown when /analyze does not begin streaming
 * within 5 seconds. Reproduced EXACTLY from the CP-PATH5 contract (AAP §0.5.3):
 * the dash between 20 and 40 is an en-dash (U+2013) and there is intentionally
 * NO trailing period. Do not alter the wording or punctuation.
 */
export const MODEL_WARMING_MESSAGE =
  "Model warming up, this may take 20–40 seconds on first run";

/** Default timeout (ms) for the /health probe. */
export const HEALTH_TIMEOUT_MS = 3000;

/** Delay (ms) used when "no streaming yet" should surface MODEL_WARMING_MESSAGE. */
export const ANALYZE_WARMING_DELAY_MS = 5000;

/**
 * Resolve the backend base URL from `NEXT_PUBLIC_API_URL`, stripping any
 * trailing slash so callers can safely join `${base}/health` / `${base}/analyze`.
 * Returns "" when unset (callers should treat that as "backend not configured").
 */
export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL ?? "";
  return raw.replace(/\/+$/, "");
}

/** Build a full backend URL for a given path (path should start with "/"). */
export function apiUrl(path: string): string {
  const base = getApiBaseUrl();
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

/**
 * GET {base}/health with an AbortController timeout. Returns the parsed
 * HealthResponse on success, or `null` on timeout / network error / non-OK
 * response. A `null` result is the UI's cue to render BACKEND_OFFLINE_MESSAGE.
 */
export async function checkHealth(
  timeoutMs: number = HEALTH_TIMEOUT_MS,
): Promise<HealthResponse | null> {
  const base = getApiBaseUrl();
  if (!base) {
    return null;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!res.ok) {
      return null;
    }
    const data = (await res.json()) as HealthResponse;
    return data;
  } catch {
    // AbortError (timeout) or network failure → backend offline.
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Convenience: true when the backend reports a fully-loaded model. */
export async function isBackendReady(timeoutMs?: number): Promise<boolean> {
  const health = await checkHealth(timeoutMs);
  return health?.status === "ready";
}

/**
 * Resolve after `ms` milliseconds OR immediately when `signal` aborts —
 * whichever comes first. The timer and the abort listener are always torn down
 * so no handles leak. Used by pollHealthUntilReady so an aborted poll stops
 * waiting promptly instead of blocking for the full inter-attempt interval.
 */
function abortAwareDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    // `finish` is a hoisted declaration so it can be passed to setTimeout
    // before its textual position; it always clears the timer and detaches the
    // abort listener, then resolves exactly once (whichever path fires first).
    const timer = setTimeout(finish, ms);
    function finish(): void {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    }
    signal?.addEventListener("abort", finish, { once: true });
  });
}

/**
 * Poll /health until the model reports "ready" (or the attempt budget is
 * exhausted). Resolves with the final HealthResponse, or `null` if the backend
 * stayed unreachable / never became ready within the budget. Useful for the
 * Dashboard to know when to enable the analyze action.
 *
 * Short-circuits to `null` when NEXT_PUBLIC_API_URL is unset — checkHealth()
 * would return null on every attempt, so looping and sleeping would be pure
 * waste. The inter-attempt wait is abort-aware: if `options.signal` aborts
 * mid-wait the poll stops promptly instead of blocking for the full interval.
 */
export async function pollHealthUntilReady(options?: {
  intervalMs?: number;
  maxAttempts?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}): Promise<HealthResponse | null> {
  const intervalMs = options?.intervalMs ?? 2000;
  const maxAttempts = options?.maxAttempts ?? 30;
  const timeoutMs = options?.timeoutMs ?? HEALTH_TIMEOUT_MS;
  const signal = options?.signal;

  // Backend not configured: every probe would fail — do not retry or sleep.
  if (!getApiBaseUrl()) {
    return null;
  }

  let last: HealthResponse | null = null;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (signal?.aborted) {
      return last;
    }
    last = await checkHealth(timeoutMs);
    if (last?.status === "ready") {
      return last;
    }
    // Abort-aware wait: resolve promptly if the caller aborts mid-interval
    // instead of blocking for the full `intervalMs`.
    await abortAwareDelay(intervalMs, signal);
  }
  return last;
}
