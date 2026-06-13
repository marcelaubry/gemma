"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";

import {
  BACKEND_OFFLINE_MESSAGE,
  MODEL_WARMING_MESSAGE,
  checkHealth,
} from "@/lib/api";
import useEventSource from "@/lib/useEventSource";
import LayerActivityBarChart from "@/components/LayerActivityBarChart";
import MemoryBreakdown from "@/components/MemoryBreakdown";
import GpuWaveform from "@/components/GpuWaveform";
import PerTokenCost from "@/components/PerTokenCost";

const PALETTE = {
  background: "#0f1117",
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
  secondary: "#8b5cf6",
  healthy: "#10b981",
  counter: "#f59e0b",
} as const;

const HEALTH_POLL_MS = 5000;

type HealthState = "unknown" | "ready" | "loading" | "offline";

export default function Dashboard() {
  const { status, layers, latestLayer, finalEvent, error, warming, start } =
    useEventSource();
  const [prompt, setPrompt] = useState("");
  const [health, setHealth] = useState<HealthState>("unknown");
  const activeRef = useRef(true);

  // Poll /health on mount and on an interval; null => backend offline.
  useEffect(() => {
    activeRef.current = true;
    const poll = async () => {
      const result = await checkHealth();
      if (!activeRef.current) {
        return;
      }
      setHealth(result === null ? "offline" : result.status);
    };
    void poll();
    const id = setInterval(() => void poll(), HEALTH_POLL_MS);
    return () => {
      activeRef.current = false;
      clearInterval(id);
    };
  }, []);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = prompt.trim();
    if (trimmed.length > 0 && status !== "streaming") {
      start(trimmed);
    }
  };

  // Resilience/error banners. Each banner carries a SEPARATE text color and
  // border color. The border may use the saturated red #ef4444 (a non-text UI
  // boundary — 4.24:1 on the #1e2130 panel, comfortably above the WCAG 1.4.11
  // 3:1 threshold for UI components), while the banner TEXT uses the lighter
  // #f87171 (5.77:1 on #1e2130) so the copy clears WCAG 2.x AA 4.5:1 for normal
  // text. Both reds are implementation-chosen (NOT part of the immutable AAP
  // palette in §0.1.2/§0.7.1), so tuning them for contrast is spec-compliant.
  const banners: {
    key: string;
    text: string;
    textColor: string;
    borderColor: string;
  }[] = [];
  if (health === "offline") {
    // Offline is the highest-priority status: when the backend is unreachable,
    // show ONLY the offline banner and suppress any stale model-warming or
    // stream-error banner left over from a prior analyze attempt, so the
    // operator sees a single, unambiguous status rather than contradictory
    // alerts (QA FINAL-ACCEPTANCE Issue 8).
    banners.push({
      key: "offline",
      text: BACKEND_OFFLINE_MESSAGE,
      textColor: "#f87171",
      borderColor: "#ef4444",
    });
  } else {
    if (warming) {
      // Amber (#f59e0b) is 7.43:1 on the panel, so text and border share it.
      banners.push({
        key: "warming",
        text: MODEL_WARMING_MESSAGE,
        textColor: PALETTE.counter,
        borderColor: PALETTE.counter,
      });
    }
    if (error !== null) {
      banners.push({
        key: "error",
        text: error,
        textColor: "#f87171",
        borderColor: "#ef4444",
      });
    }
  }

  return (
    <main
      data-testid="dashboard"
      style={{
        minHeight: "100vh",
        background: PALETTE.background,
        color: "#e5e7eb",
        fontFamily: "system-ui, sans-serif",
        padding: 24,
        boxSizing: "border-box",
      }}
    >
      <header style={{ marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>
          Gemma Compute Monitor
        </h1>
        {/* The descriptive lead-in is intentionally muted (opacity 0.7), but the
           opacity is scoped to that text ONLY — applying it to the whole <p>
           previously dimmed the palette-colored health-status word too, dropping
           the healthy green (#10b981) to an effective ~#108761 ≈ 4.18:1 (below
           WCAG AA 4.5:1). Keeping the status span at full opacity renders the
           immutable palette colors at full strength: healthy #10b981 ≈ 7.44:1
           and the offline counter #f59e0b ≈ 7.43:1, both comfortably ≥ AA. */}
        <p style={{ margin: "4px 0 0", fontSize: 13 }}>
          <span style={{ opacity: 0.7 }}>
            Per-layer telemetry for Gemma 3 4B ·{" "}
          </span>
          <span
            data-testid="health-status"
            style={{ color: health === "ready" ? PALETTE.healthy : PALETTE.counter }}
          >
            {health}
          </span>
        </p>
      </header>

      {/*
        Form controls are styled via CSS classes in app/globals.css (not inline
        style objects) so the input and button expose real :focus-visible,
        :hover, :active, and :disabled states for keyboard users. The button's
        disabled visual (while streaming or with an empty prompt) is handled by
        `.gcm-dash__button:disabled`.
      */}
      <form onSubmit={onSubmit} className="gcm-dash__form">
        <input
          id="prompt"
          name="prompt"
          aria-label="Prompt"
          placeholder="Enter a prompt to analyze…"
          value={prompt}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setPrompt(e.target.value)}
          className="gcm-dash__input"
        />
        <button
          type="submit"
          disabled={status === "streaming" || prompt.trim().length === 0}
          className="gcm-dash__button"
        >
          {status === "streaming" ? "Analyzing…" : "Analyze"}
        </button>
      </form>

      {/*
        Resilience/error banners (backend-offline, model-warming, stream-error)
        are urgent operator notices, so they use role="alert" (an assertive live
        region) per the checkpoint accessibility requirement.
      */}
      {banners.map((b) => (
        <div
          key={b.key}
          role="alert"
          style={{
            background: PALETTE.panel,
            border: `1px solid ${b.borderColor}`,
            color: b.textColor,
            borderRadius: 8,
            padding: "10px 12px",
            marginBottom: 12,
            fontSize: 13,
          }}
        >
          {b.text}
        </div>
      ))}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 16,
        }}
      >
        <LayerActivityBarChart layers={layers} latestLayer={latestLayer} />
        <MemoryBreakdown latestLayer={latestLayer} />
        <GpuWaveform layers={layers} />
      </div>

      <div style={{ marginTop: 16 }}>
        <PerTokenCost perTokenMs={finalEvent ? finalEvent.per_token_ms : null} />
      </div>
    </main>
  );
}
