/**
 * Panel-animation tests for the Gemma Compute Monitor frontend (AAP §0.5.3).
 *
 * One describe block per telemetry panel. The bar count is asserted to be the
 * TRUE Gemma 3 4B layer count of 34 (derived dynamically from telemetry),
 * never 18 (which belongs to Gemma 2B / Gemma 3 270M) — AAP §0.7.2.
 */

import { render, screen, act } from "@testing-library/react";
import type { LayerEvent } from "@/lib/types";
import LayerActivityBarChart from "@/components/LayerActivityBarChart";
import MemoryBreakdown from "@/components/MemoryBreakdown";
import GpuWaveform from "@/components/GpuWaveform";
import PerTokenCost from "@/components/PerTokenCost";

// Defensive rAF polyfill (some jsdom builds omit it); the GpuWaveform "draws"
// test installs its own spies over these.
if (typeof window.requestAnimationFrame === "undefined") {
  window.requestAnimationFrame = ((cb: FrameRequestCallback) =>
    setTimeout(() => cb(Date.now()), 0) as unknown as number) as typeof window.requestAnimationFrame;
  window.cancelAnimationFrame = ((id: number) =>
    clearTimeout(id as unknown as ReturnType<typeof setTimeout>)) as typeof window.cancelAnimationFrame;
}

// Gemma 3 4B has 34 transformer layers (_NUM_LAYERS_GEMMA3_4B = 34 in
// gemma/gm/nn/_gemma.py). NEVER 18 (that is Gemma 2B / Gemma 3 270M).
const GEMMA3_4B_LAYERS = 34;

function makeLayer(
  layer: number,
  gpu_pct = 40,
  extra: Partial<LayerEvent> = {},
): LayerEvent {
  return {
    layer,
    gpu_pct,
    cpu_pct: 10,
    memory_used_gb: 12,
    kv_cache_gb: 1,
    activation_gb: 0.5,
    elapsed_ms: 8,
    ...extra,
  };
}

function layerBars(): HTMLElement[] {
  return screen.getAllByTestId(/^layer-bar-\d+$/);
}

describe("LayerActivityBarChart", () => {
  it("renders exactly 34 bars for Gemma 3 4B (dynamic count, never 18)", () => {
    const layers = Array.from({ length: GEMMA3_4B_LAYERS }, (_, i) =>
      makeLayer(i, 50),
    );
    render(
      <LayerActivityBarChart
        layers={layers}
        latestLayer={layers[GEMMA3_4B_LAYERS - 1]}
      />,
    );

    const bars = layerBars();
    expect(bars).toHaveLength(34);
    expect(bars).not.toHaveLength(18);
    expect(screen.getByTestId("active-layer-counter")).toHaveTextContent(
      "Layer L34 / 34",
    );
    expect(screen.getByText("L1")).toBeInTheDocument();
    expect(screen.getByText("L34")).toBeInTheDocument();
  });

  it("derives the bar count dynamically from telemetry", () => {
    const layers = Array.from({ length: 5 }, (_, i) => makeLayer(i, 30));
    render(<LayerActivityBarChart layers={layers} latestLayer={layers[4]} />);

    expect(layerBars()).toHaveLength(5);
    expect(screen.getByTestId("active-layer-counter")).toHaveTextContent(
      "Layer L5 / 5",
    );
  });

  it("highlights the active bar at full opacity and dims prior bars to 0.4", () => {
    const layers = Array.from({ length: GEMMA3_4B_LAYERS }, (_, i) =>
      makeLayer(i, 60),
    );
    render(
      <LayerActivityBarChart layers={layers} latestLayer={makeLayer(33, 60)} />,
    );

    expect(Number(screen.getByTestId("layer-bar-33").style.opacity)).toBeCloseTo(1);
    expect(Number(screen.getByTestId("layer-bar-0").style.opacity)).toBeCloseTo(0.4);
  });

  it("shows '0 / count' when there is no latest layer (still dynamic 34 bars)", () => {
    const layers = Array.from({ length: GEMMA3_4B_LAYERS }, (_, i) => makeLayer(i));
    render(<LayerActivityBarChart layers={layers} latestLayer={null} />);

    expect(layerBars()).toHaveLength(34);
    expect(screen.getByTestId("active-layer-counter")).toHaveTextContent("0 / 34");
  });
});

describe("MemoryBreakdown", () => {
  it("renders the four memory segments and an accessible summary", () => {
    render(<MemoryBreakdown latestLayer={null} />);

    expect(screen.getByTestId("memory-segment-modelweights")).toBeInTheDocument();
    expect(screen.getByTestId("memory-segment-kvcache")).toBeInTheDocument();
    expect(screen.getByTestId("memory-segment-activations")).toBeInTheDocument();
    expect(screen.getByTestId("memory-segment-osother")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Unified memory usage/ }),
    ).toBeInTheDocument();
  });

  it("zeroes the dynamic segments when there is no telemetry", () => {
    render(<MemoryBreakdown latestLayer={null} />);

    expect(parseFloat(screen.getByTestId("memory-segment-kvcache").style.width)).toBeCloseTo(0);
    expect(parseFloat(screen.getByTestId("memory-segment-activations").style.width)).toBeCloseTo(0);
  });

  it("sizes KV-cache and activation segments against the 48 GB pool", () => {
    render(
      <MemoryBreakdown
        latestLayer={makeLayer(10, 50, { kv_cache_gb: 24, activation_gb: 12 })}
      />,
    );

    expect(parseFloat(screen.getByTestId("memory-segment-kvcache").style.width)).toBeCloseTo(50);
    expect(parseFloat(screen.getByTestId("memory-segment-activations").style.width)).toBeCloseTo(25);
  });
});

describe("GpuWaveform", () => {
  it("mounts and renders the canvas without throwing (jsdom + jest-canvas-mock)", () => {
    render(<GpuWaveform layers={[makeLayer(0, 50)]} />);

    expect(screen.getByTestId("gpu-waveform")).toBeInTheDocument();
    expect(screen.getByTestId("gpu-waveform-canvas")).toBeInTheDocument();
  });

  it("schedules animation frames and draws without throwing", () => {
    const callbacks: FrameRequestCallback[] = [];
    const rafSpy = jest
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((cb: FrameRequestCallback) => {
        callbacks.push(cb);
        return callbacks.length;
      });
    const cancelSpy = jest
      .spyOn(window, "cancelAnimationFrame")
      .mockImplementation(() => undefined);

    const { rerender, unmount } = render(
      <GpuWaveform layers={[makeLayer(0, 50)]} />,
    );

    expect(rafSpy).toHaveBeenCalled();
    // Run one captured frame to exercise the draw path against the mocked canvas.
    expect(() =>
      act(() => {
        callbacks[0]?.(performance.now());
      }),
    ).not.toThrow();
    expect(() =>
      rerender(<GpuWaveform layers={[makeLayer(0, 50), makeLayer(1, 70)]} />),
    ).not.toThrow();

    unmount();
    expect(cancelSpy).toHaveBeenCalled();

    rafSpy.mockRestore();
    cancelSpy.mockRestore();
  });
});

describe("PerTokenCost", () => {
  it("renders nothing before the final event (perTokenMs = null)", () => {
    const { container } = render(<PerTokenCost perTokenMs={null} />);

    expect(screen.queryByTestId("per-token-cost")).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for an empty token array", () => {
    render(<PerTokenCost perTokenMs={[]} />);

    expect(screen.queryByTestId("per-token-cost")).not.toBeInTheDocument();
  });

  it("renders one chip per token with brightness proportional to per_token_ms / max", () => {
    render(<PerTokenCost perTokenMs={[10, 20, 40]} />);

    const chips = screen.getAllByTestId(/^token-chip-\d+$/);
    expect(chips).toHaveLength(3);
    expect(screen.getByText("3 tokens")).toBeInTheDocument();
    expect(Number(screen.getByTestId("token-chip-0").style.opacity)).toBeCloseTo(0.4375);
    expect(Number(screen.getByTestId("token-chip-1").style.opacity)).toBeCloseTo(0.625);
    expect(Number(screen.getByTestId("token-chip-2").style.opacity)).toBeCloseTo(1);
  });

  it("uses base brightness (0.25) for a zero-cost token and full for the max", () => {
    render(<PerTokenCost perTokenMs={[0, 40]} />);

    expect(Number(screen.getByTestId("token-chip-0").style.opacity)).toBeCloseTo(0.25);
    expect(Number(screen.getByTestId("token-chip-1").style.opacity)).toBeCloseTo(1);
  });
});
