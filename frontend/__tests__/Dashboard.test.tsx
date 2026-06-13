/**
 * Dashboard orchestration + resilience-surfacing tests (AAP §0.5.3).
 *
 * The useEventSource hook is mocked so each test controls the streaming state;
 * @/lib/api is partially mocked (only checkHealth) so the REAL byte-exact
 * BACKEND_OFFLINE_MESSAGE (em-dash U+2014) and MODEL_WARMING_MESSAGE
 * (en-dash U+2013) constants are exercised verbatim. The four panels render for
 * real (canvas via jest-canvas-mock); no live backend is required.
 */

jest.mock("@/lib/useEventSource", () => ({ __esModule: true, default: jest.fn() }));
jest.mock("@/lib/api", () => {
  const actual = jest.requireActual("@/lib/api");
  return { __esModule: true, ...actual, checkHealth: jest.fn() };
});

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Dashboard from "@/components/Dashboard";
import useEventSource, { type UseEventSourceResult } from "@/lib/useEventSource";
import {
  checkHealth,
  BACKEND_OFFLINE_MESSAGE,
  MODEL_WARMING_MESSAGE,
} from "@/lib/api";
import type { LayerEvent } from "@/lib/types";

const mockUseEventSource = jest.mocked(useEventSource);
const mockCheckHealth = jest.mocked(checkHealth);

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

function hookState(
  overrides: Partial<UseEventSourceResult> = {},
): UseEventSourceResult {
  return {
    status: "idle",
    layers: [],
    latestLayer: null,
    finalEvent: null,
    error: null,
    warming: false,
    start: jest.fn(),
    stop: jest.fn(),
    reset: jest.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  mockUseEventSource.mockReturnValue(hookState());
  mockCheckHealth.mockResolvedValue({ status: "ready", model: "gemma-3-4b" });
});

afterEach(() => {
  jest.clearAllMocks();
});

describe("Dashboard", () => {
  it("renders the shell, health indicator, and the three streaming panels", async () => {
    render(<Dashboard />);

    expect(screen.getByTestId("dashboard")).toBeInTheDocument();
    expect(screen.getByTestId("layer-bar-chart")).toBeInTheDocument();
    expect(screen.getByTestId("memory-breakdown")).toBeInTheDocument();
    expect(screen.getByTestId("gpu-waveform")).toBeInTheDocument();
    expect(screen.queryByTestId("per-token-cost")).not.toBeInTheDocument();

    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent("ready"),
    );
  });

  it("surfaces the verbatim BACKEND_OFFLINE_MESSAGE when /health is unreachable", async () => {
    mockCheckHealth.mockResolvedValue(null);
    render(<Dashboard />);

    const banner = await screen.findByText(BACKEND_OFFLINE_MESSAGE);
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent("\u2014"); // em-dash preserved
  });

  it("surfaces the verbatim MODEL_WARMING_MESSAGE when the hook reports warming", async () => {
    mockUseEventSource.mockReturnValue(hookState({ warming: true }));
    render(<Dashboard />);

    const banner = await screen.findByText(MODEL_WARMING_MESSAGE);
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent("\u2013"); // en-dash preserved
  });

  it("surfaces hook error messages", async () => {
    mockUseEventSource.mockReturnValue(
      hookState({ status: "error", error: "Request failed with status 500" }),
    );
    render(<Dashboard />);

    expect(
      await screen.findByText("Request failed with status 500"),
    ).toBeInTheDocument();
  });

  it("calls start() with the trimmed prompt on submit", async () => {
    const start = jest.fn();
    mockUseEventSource.mockReturnValue(hookState({ start }));
    const user = userEvent.setup();
    render(<Dashboard />);

    await user.type(screen.getByLabelText("Prompt"), "  explain attention  ");
    await user.click(screen.getByRole("button", { name: "Analyze" }));

    expect(start).toHaveBeenCalledWith("explain attention");
  });

  it("disables the analyze button and shows 'Analyzing…' while streaming", async () => {
    mockUseEventSource.mockReturnValue(hookState({ status: "streaming" }));
    render(<Dashboard />);

    const button = screen.getByRole("button", { name: "Analyzing…" });
    expect(button).toBeDisabled();

    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent("ready"),
    );
  });

  it("renders the per-token cost panel only after the final event", async () => {
    mockUseEventSource.mockReturnValue(
      hookState({ finalEvent: { per_token_ms: [10, 20, 40], done: true } }),
    );
    render(<Dashboard />);

    expect(screen.getByTestId("per-token-cost")).toBeInTheDocument();
    expect(screen.getAllByTestId(/^token-chip-\d+$/)).toHaveLength(3);

    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent("ready"),
    );
  });

  it("distributes layer telemetry to the bar chart (34 bars, never 18)", async () => {
    const layers = Array.from({ length: 34 }, (_, i) => makeLayer(i, 50));
    mockUseEventSource.mockReturnValue(
      hookState({ layers, latestLayer: layers[33] }),
    );
    render(<Dashboard />);

    const bars = screen.getAllByTestId(/^layer-bar-\d+$/);
    expect(bars).toHaveLength(34);
    expect(bars).not.toHaveLength(18);
    expect(screen.getByTestId("active-layer-counter")).toHaveTextContent(
      "Layer L34 / 34",
    );

    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent("ready"),
    );
  });

  it("styles the prompt input and Analyze button via CSS classes (F3: keyboard-focus + interactive states)", async () => {
    render(<Dashboard />);

    // Controls carry the globals.css classes that define :focus-visible,
    // :hover, :active, and :disabled states — replacing the removed inline
    // `outline: none`, which left no visible keyboard-focus indicator.
    const input = screen.getByLabelText("Prompt");
    expect(input).toHaveClass("gcm-dash__input");
    expect(screen.getByRole("button", { name: "Analyze" })).toHaveClass(
      "gcm-dash__button",
    );
    // The input no longer suppresses the focus outline via an inline style.
    expect(input.getAttribute("style") ?? "").not.toContain("outline");

    await waitFor(() =>
      expect(screen.getByTestId("health-status")).toHaveTextContent("ready"),
    );
  });

  it("renders online resilience/error banners with role='alert' (F4: assertive live region)", async () => {
    // While the backend is reachable (checkHealth → ready), the model-warming
    // and stream-error banners both surface as assertive live regions.
    mockCheckHealth.mockResolvedValue({ status: "ready", model: "gemma-3-4b" });
    mockUseEventSource.mockReturnValue(
      hookState({
        status: "error",
        error: "Request failed with status 500",
        warming: true,
      }),
    );
    render(<Dashboard />);

    expect(await screen.findByText(MODEL_WARMING_MESSAGE)).toHaveAttribute(
      "role",
      "alert",
    );
    expect(screen.getByText("Request failed with status 500")).toHaveAttribute(
      "role",
      "alert",
    );
    // The offline banner is NOT shown while the backend is reachable.
    expect(screen.queryByText(BACKEND_OFFLINE_MESSAGE)).not.toBeInTheDocument();
    // None of the resilience banners use the weaker role="status".
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows ONLY the offline banner when offline, suppressing stale warming/error banners (QA Issue 8)", async () => {
    // Even when the hook still carries a stale model-warming flag and a prior
    // stream error, an offline backend (checkHealth → null) must surface a
    // single, unambiguous offline alert and suppress the contradictory banners
    // so the operator never sees stale, conflicting status at once.
    mockCheckHealth.mockResolvedValue(null);
    mockUseEventSource.mockReturnValue(
      hookState({
        status: "error",
        error: "Request failed with status 500",
        warming: true,
      }),
    );
    render(<Dashboard />);

    const offline = await screen.findByText(BACKEND_OFFLINE_MESSAGE);
    expect(offline).toHaveAttribute("role", "alert");
    // The stale warming and error banners are suppressed while offline.
    await waitFor(() => {
      expect(
        screen.queryByText(MODEL_WARMING_MESSAGE),
      ).not.toBeInTheDocument();
    });
    expect(
      screen.queryByText("Request failed with status 500"),
    ).not.toBeInTheDocument();
    // Exactly one alert region is present, and none uses the weaker
    // role="status".
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
