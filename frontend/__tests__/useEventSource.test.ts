/**
 * SSE / contract-handling tests for the Gemma Compute Monitor frontend
 * (AAP §0.5.3). Covers the pure SSE-parsing helpers, telemetry classification,
 * and the useEventSource hook's Fetch-streaming behavior + error paths.
 *
 * The hook uses Fetch-streaming POST (native EventSource is GET-only and cannot
 * carry the immutable POST { prompt } body), so we mock global.fetch with a
 * body whose getReader() yields Uint8Array chunks of `data: {json}\n\n` frames.
 */

import { TextEncoder, TextDecoder } from "util";

// jest-environment-jsdom does not provide TextEncoder/TextDecoder, and
// jest.setup.js intentionally leaves streaming polyfills to the specific test.
// The hook constructs `new TextDecoder()` at runtime, so install the globals.
if (typeof (globalThis as unknown as { TextEncoder?: unknown }).TextEncoder === "undefined") {
  (globalThis as unknown as { TextEncoder: typeof TextEncoder }).TextEncoder = TextEncoder;
}
if (typeof (globalThis as unknown as { TextDecoder?: unknown }).TextDecoder === "undefined") {
  (globalThis as unknown as { TextDecoder: unknown }).TextDecoder = TextDecoder;
}

import { renderHook, act, waitFor } from "@testing-library/react";
import useEventSource, {
  splitSseFrames,
  parseSseData,
  parseTelemetryFrame,
  isValidLayerEvent,
  BUSY_MESSAGE,
} from "@/lib/useEventSource";
import { MODEL_WARMING_MESSAGE } from "@/lib/api";
import {
  isFinalEvent,
  isLayerEvent,
  type LayerEvent,
  type TelemetryEvent,
} from "@/lib/types";

const API = "https://test-tunnel.ngrok.app";
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
const originalFetch = global.fetch;

function makeLayer(layer: number, gpu_pct = 40): LayerEvent {
  return {
    layer,
    gpu_pct,
    cpu_pct: 10,
    memory_used_gb: 12,
    kv_cache_gb: 1,
    activation_gb: 0.5,
    elapsed_ms: 8,
  };
}

function frame(ev: unknown, sep = "\n\n"): string {
  return `data: ${JSON.stringify(ev)}${sep}`;
}

interface MockReader {
  read: jest.Mock;
  cancel: jest.Mock;
}

function mockFetchStream(
  frames: string[],
  opts: { chunkSize?: number } = {},
): MockReader {
  const enc = new TextEncoder();
  const bytes = enc.encode(frames.join(""));
  const chunkSize = opts.chunkSize ?? (bytes.length || 1);
  const chunks: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    chunks.push(bytes.slice(i, i + chunkSize));
  }
  let idx = 0;
  const reader: MockReader = {
    read: jest.fn(async () => {
      if (idx < chunks.length) {
        return { done: false, value: chunks[idx++] };
      }
      return { done: true, value: undefined };
    }),
    cancel: jest.fn(async () => undefined),
  };
  global.fetch = jest.fn(async () => ({
    ok: true,
    status: 200,
    body: { getReader: () => reader },
  })) as unknown as typeof fetch;
  return reader;
}

function mockFetchStatus(status: number): void {
  global.fetch = jest.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
  })) as unknown as typeof fetch;
}

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = API;
});

afterEach(() => {
  global.fetch = originalFetch;
  if (originalApiUrl === undefined) {
    delete process.env.NEXT_PUBLIC_API_URL;
  } else {
    process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
  }
  jest.clearAllMocks();
});

describe("pure SSE helpers", () => {
  it("splitSseFrames returns complete frames and carries the trailing partial as rest", () => {
    const { frames, rest } = splitSseFrames("data:{a}\n\ndata:{b}\n\npart");
    expect(frames).toEqual(["data:{a}", "data:{b}"]);
    expect(rest).toBe("part");
  });

  it("parseSseData strips data:, joins multiline data, and ignores comments/other fields", () => {
    expect(parseSseData('event: layer\ndata: {"layer":0}')).toBe('{"layer":0}');
    expect(parseSseData(": comment-only")).toBeNull();
    expect(parseSseData("data:nospace")).toBe("nospace");
    expect(parseSseData("data: a\ndata: b")).toBe("a\nb");
  });

  it("parseTelemetryFrame returns null on invalid JSON (never throws) and parses valid payloads", () => {
    expect(parseTelemetryFrame("data: not-json")).toBeNull();
    expect(parseTelemetryFrame(frame(makeLayer(0), ""))).toMatchObject({ layer: 0 });
    expect(
      parseTelemetryFrame('event: final\ndata: {"per_token_ms":[1,2],"done":true}'),
    ).toMatchObject({ done: true });
  });

  it("exposes the verbatim BUSY_MESSAGE", () => {
    expect(BUSY_MESSAGE).toBe(
      "An analysis is already in progress. Please wait for it to finish.",
    );
  });
});

describe("telemetry type guards", () => {
  it("classifies final vs layer events by payload content", () => {
    expect(isFinalEvent({ per_token_ms: [1], done: true })).toBe(true);
    expect(isFinalEvent({ per_token_ms: [] } as unknown as TelemetryEvent)).toBe(true);
    expect(isFinalEvent(makeLayer(3))).toBe(false);
    expect(isLayerEvent(makeLayer(3))).toBe(true);
  });
});

describe("isValidLayerEvent (F5 contract validation)", () => {
  it("accepts a fully-formed LayerEvent (all seven numeric fields finite)", () => {
    expect(isValidLayerEvent(makeLayer(0))).toBe(true);
    expect(isValidLayerEvent(makeLayer(33, 100))).toBe(true);
  });

  it("rejects non-objects and null", () => {
    expect(isValidLayerEvent(null)).toBe(false);
    expect(isValidLayerEvent(undefined)).toBe(false);
    expect(isValidLayerEvent(42)).toBe(false);
    expect(isValidLayerEvent("layer")).toBe(false);
  });

  it("rejects an empty object and objects missing any required field", () => {
    expect(isValidLayerEvent({})).toBe(false);
    const missingOne: Partial<LayerEvent> = { ...makeLayer(1) };
    delete missingOne.elapsed_ms;
    expect(isValidLayerEvent(missingOne)).toBe(false);
  });

  it("rejects non-finite or non-numeric field values", () => {
    expect(isValidLayerEvent({ ...makeLayer(2), gpu_pct: NaN })).toBe(false);
    expect(isValidLayerEvent({ ...makeLayer(2), cpu_pct: Infinity })).toBe(false);
    expect(isValidLayerEvent({ ...makeLayer(2), layer: "0" })).toBe(false);
    expect(isValidLayerEvent({ ...makeLayer(2), memory_used_gb: null })).toBe(false);
  });
});

describe("useEventSource streaming", () => {
  it("parses exactly 34 layer events then the final event (never 18) and cancels the reader", async () => {
    const layerFrames = Array.from({ length: 34 }, (_, i) =>
      frame(makeLayer(i, (i % 10) * 10)),
    );
    const finalFrame = frame({ per_token_ms: [12.5, 8.25, 30], done: true });
    const reader = mockFetchStream([...layerFrames, finalFrame]);

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hello");
    });

    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.layers).toHaveLength(34);
    expect(result.current.layers).not.toHaveLength(18);
    expect(result.current.latestLayer?.layer).toBe(33);
    expect(result.current.finalEvent?.per_token_ms).toEqual([12.5, 8.25, 30]);
    expect(reader.cancel).toHaveBeenCalledTimes(1);
  });

  it("reassembles frames split across tiny chunks (buffer carry-over)", async () => {
    const layerFrames = Array.from({ length: 34 }, (_, i) => frame(makeLayer(i)));
    const finalFrame = frame({ per_token_ms: [1, 2], done: true });
    mockFetchStream([...layerFrames, finalFrame], { chunkSize: 7 });

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.layers).toHaveLength(34);
    expect(result.current.finalEvent?.done).toBe(true);
  });

  it("normalizes CRLF line endings", async () => {
    const frames = [
      frame(makeLayer(0), "\r\n\r\n"),
      frame(makeLayer(1), "\r\n\r\n"),
      frame({ per_token_ms: [5], done: true }, "\r\n\r\n"),
    ];
    mockFetchStream(frames);

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.layers).toHaveLength(2);
  });

  it("ignores malformed non-final frames and never appends them as layers (F5)", async () => {
    const frames = [
      frame(makeLayer(0, 10)), // valid
      frame({}), // malformed: empty object
      frame({ layer: 1 }), // malformed: missing fields
      frame({ ...makeLayer(2, 30), gpu_pct: "high" }), // malformed: wrong type
      frame(makeLayer(3, 40)), // valid
      frame({ per_token_ms: [5, 6], done: true }),
    ];
    mockFetchStream(frames);

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("done"));
    // Only the two well-formed layer events are kept; the three malformed
    // frames are silently dropped (never appended with undefined fields).
    expect(result.current.layers).toHaveLength(2);
    expect(result.current.layers.map((l) => l.layer)).toEqual([0, 3]);
    expect(result.current.latestLayer?.layer).toBe(3);
    expect(
      result.current.layers.every(
        (l) => typeof l.gpu_pct === "number" && Number.isFinite(l.gpu_pct),
      ),
    ).toBe(true);
    expect(result.current.finalEvent?.done).toBe(true);
  });

  it("surfaces MODEL_WARMING_MESSAGE on HTTP 503", async () => {
    mockFetchStatus(503);
    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe(MODEL_WARMING_MESSAGE);
  });

  it("surfaces BUSY_MESSAGE on HTTP 429", async () => {
    mockFetchStatus(429);
    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe(BUSY_MESSAGE);
  });

  it("reports a status-coded error for other non-OK responses", async () => {
    mockFetchStatus(500);
    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("Request failed with status 500");
  });

  it("errors when NEXT_PUBLIC_API_URL is not configured (and does not fetch)", () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    const fetchSpy = jest.fn();
    global.fetch = fetchSpy as unknown as typeof fetch;

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("NEXT_PUBLIC_API_URL is not configured.");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("surfaces the warming flag after 5s with no streamed event", () => {
    jest.useFakeTimers();
    global.fetch = jest.fn(() => new Promise(() => {})) as unknown as typeof fetch;

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });
    expect(result.current.status).toBe("streaming");
    expect(result.current.warming).toBe(false);

    act(() => {
      jest.advanceTimersByTime(5000);
    });
    expect(result.current.warming).toBe(true);

    act(() => {
      result.current.stop();
    });
    jest.useRealTimers();
  });

  it("stop() aborts an in-flight stream and returns to idle", () => {
    global.fetch = jest.fn(() => new Promise(() => {})) as unknown as typeof fetch;

    const { result } = renderHook(() => useEventSource());
    act(() => {
      result.current.start("hi");
    });
    expect(result.current.status).toBe("streaming");

    act(() => {
      result.current.stop();
    });
    expect(result.current.status).toBe("idle");
  });
});
