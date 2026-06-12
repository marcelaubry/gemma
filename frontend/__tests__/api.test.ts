/**
 * Tests for frontend/lib/api.ts — backend base-URL resolution, the two verbatim
 * resilience messages, and the pollHealthUntilReady defensive behavior.
 *
 * The resilience-message tests assert the EXACT message literals with explicit
 * Unicode code points (an INDEPENDENT oracle), rather than importing the
 * constant and comparing it against itself. This is the gap called out in the
 * review: the Dashboard/useEventSource suites use `MODEL_WARMING_MESSAGE` as
 * their own oracle, so a drifted constant stays green there — these assertions
 * catch any byte-exact CP-PATH5 regression (en-dash U+2013, NO trailing
 * period) on their own.
 */

import {
  BACKEND_OFFLINE_MESSAGE,
  MODEL_WARMING_MESSAGE,
  getApiBaseUrl,
  apiUrl,
  pollHealthUntilReady,
} from "@/lib/api";

const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  if (originalApiUrl === undefined) {
    delete process.env.NEXT_PUBLIC_API_URL;
  } else {
    process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
  }
  jest.clearAllMocks();
});

describe("resilience message constants (CP-PATH5 byte-exact)", () => {
  it("MODEL_WARMING_MESSAGE is the exact literal: en-dash U+2013 and NO trailing period", () => {
    // Independent oracle: the expected text is spelled out here with an
    // explicit \u2013 en-dash and no final period — NOT read from the constant.
    expect(MODEL_WARMING_MESSAGE).toBe(
      "Model warming up, this may take 20\u201340 seconds on first run",
    );
    expect(MODEL_WARMING_MESSAGE).toContain("\u2013"); // en-dash present
    expect(MODEL_WARMING_MESSAGE).not.toContain("\u2014"); // not an em-dash
    expect(MODEL_WARMING_MESSAGE).not.toContain("-"); // not an ASCII hyphen
    expect(MODEL_WARMING_MESSAGE.endsWith(".")).toBe(false); // no trailing period
    expect(MODEL_WARMING_MESSAGE.endsWith("run")).toBe(true);
  });

  it("BACKEND_OFFLINE_MESSAGE is the exact literal with an em-dash U+2014", () => {
    expect(BACKEND_OFFLINE_MESSAGE).toBe(
      "Backend offline \u2014 start the local server and update the ngrok URL in Railway",
    );
    expect(BACKEND_OFFLINE_MESSAGE).toContain("\u2014"); // em-dash present
    expect(BACKEND_OFFLINE_MESSAGE).not.toContain("\u2013"); // not an en-dash
  });
});

describe("getApiBaseUrl / apiUrl", () => {
  it("returns '' when NEXT_PUBLIC_API_URL is unset", () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    expect(getApiBaseUrl()).toBe("");
  });

  it("strips trailing slashes and joins paths cleanly", () => {
    process.env.NEXT_PUBLIC_API_URL = "https://x.ngrok.app///";
    expect(getApiBaseUrl()).toBe("https://x.ngrok.app");
    expect(apiUrl("/health")).toBe("https://x.ngrok.app/health");
    expect(apiUrl("analyze")).toBe("https://x.ngrok.app/analyze");
  });
});

describe("pollHealthUntilReady (F6 defensive behavior)", () => {
  it("short-circuits to null without fetching or sleeping when the base URL is unset", async () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    const fetchSpy = jest.fn();
    global.fetch = fetchSpy as unknown as typeof fetch;

    // A huge interval + attempt budget: if the helper looped/slept instead of
    // short-circuiting, this await would hang until the Jest timeout (failure).
    const result = await pollHealthUntilReady({
      intervalMs: 1_000_000,
      maxAttempts: 50,
    });

    expect(result).toBeNull();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("stops promptly when the signal aborts mid-wait (abort-aware delay)", async () => {
    process.env.NEXT_PUBLIC_API_URL = "https://test-tunnel.ngrok.app";
    // Always "loading" so the poll never resolves on its own and enters the wait.
    global.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ status: "loading", model: "gemma-3-4b" }),
    })) as unknown as typeof fetch;

    const controller = new AbortController();
    const settled = jest.fn();
    const promise = pollHealthUntilReady({
      intervalMs: 60_000, // long enough that only an abort can end the wait
      maxAttempts: 10,
      timeoutMs: 1000,
      signal: controller.signal,
    }).then((r) => {
      settled(r);
      return r;
    });

    // Flush microtasks: the first checkHealth() resolves and we park in the wait.
    await new Promise((r) => setTimeout(r, 0));
    expect(settled).not.toHaveBeenCalled();
    expect(global.fetch).toHaveBeenCalledTimes(1);

    // Abort mid-wait → the abort-aware delay resolves at once and the loop
    // returns the last health WITHOUT blocking for the full 60s interval.
    controller.abort();
    const result = await promise;

    expect(settled).toHaveBeenCalled();
    expect(result).toEqual({ status: "loading", model: "gemma-3-4b" });
    expect(global.fetch).toHaveBeenCalledTimes(1); // no further attempts after abort
  });

  it("returns the ready health once the backend reports ready", async () => {
    process.env.NEXT_PUBLIC_API_URL = "https://test-tunnel.ngrok.app";
    global.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ status: "ready", model: "gemma-3-4b" }),
    })) as unknown as typeof fetch;

    const result = await pollHealthUntilReady({ intervalMs: 10, maxAttempts: 3 });
    expect(result).toEqual({ status: "ready", model: "gemma-3-4b" });
  });
});
