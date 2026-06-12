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

  const banners: { key: string; text: string; color: string }[] = [];
  if (health === "offline") {
    banners.push({ key: "offline", text: BACKEND_OFFLINE_MESSAGE, color: "#ef4444" });
  }
  if (warming) {
    banners.push({ key: "warming", text: MODEL_WARMING_MESSAGE, color: PALETTE.counter });
  }
  if (error !== null) {
    banners.push({ key: "error", text: error, color: "#ef4444" });
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
        <p style={{ margin: "4px 0 0", fontSize: 13, opacity: 0.7 }}>
          Per-layer telemetry for Gemma 3 4B ·{" "}
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
            border: `1px solid ${b.color}`,
            color: b.color,
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
