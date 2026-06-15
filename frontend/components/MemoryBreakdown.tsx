"use client";

import type { LayerEvent } from "@/lib/types";

const PALETTE = {
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
  secondary: "#8b5cf6",
  healthy: "#10b981",
} as const;

/** Total unified-memory pool on the target M-series machine (AAP: 48 GB). */
const POOL_GB = 48;
/** Static model-weights footprint for Gemma 3 4B bf16 (AAP: ~8.5 GB). */
const MODEL_WEIGHTS_GB = 8.5;
/** Static OS/other reservation (display constant). */
const OS_OTHER_GB = 3;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export interface MemoryBreakdownProps {
  /** Latest layer event supplying kv_cache_gb / activation_gb, or null. */
  latestLayer: LayerEvent | null;
}

export default function MemoryBreakdown({ latestLayer }: MemoryBreakdownProps) {
  const kvCacheGb = latestLayer ? Math.max(latestLayer.kv_cache_gb, 0) : 0;
  const activationGb = latestLayer ? Math.max(latestLayer.activation_gb, 0) : 0;

  const segments = [
    { label: "Model Weights", gb: MODEL_WEIGHTS_GB, color: PALETTE.healthy },
    { label: "KV Cache", gb: kvCacheGb, color: PALETTE.primary },
    { label: "Activations", gb: activationGb, color: PALETTE.secondary },
    { label: "OS / Other", gb: OS_OTHER_GB, color: PALETTE.border },
  ];

  const usedGb = segments.reduce((sum, s) => sum + s.gb, 0);

  return (
    <section
      data-testid="memory-breakdown"
      style={{
        background: PALETTE.panel,
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 12,
        padding: 16,
        color: "#e5e7eb",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 14 }}>Unified Memory</h2>
        <span style={{ color: PALETTE.healthy, fontSize: 13, fontWeight: 600 }}>
          {usedGb.toFixed(1)} / {POOL_GB} GB
        </span>
      </header>

      <div
        role="img"
        aria-label={`Unified memory usage ${usedGb.toFixed(1)} of ${POOL_GB} GB`}
        style={{
          display: "flex",
          width: "100%",
          height: 28,
          borderRadius: 6,
          overflow: "hidden",
          background: "#0f1117",
          border: `1px solid ${PALETTE.border}`,
        }}
      >
        {segments.map((s) => (
          <div
            key={s.label}
            data-testid={`memory-segment-${s.label.replace(/[^a-z]/gi, "").toLowerCase()}`}
            title={`${s.label}: ${s.gb.toFixed(2)} GB`}
            style={{
              width: `${clamp((s.gb / POOL_GB) * 100, 0, 100)}%`,
              background: s.color,
              transition: "width 150ms ease-out",
            }}
          />
        ))}
      </div>

      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: "12px 0 0",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 6,
          fontSize: 12,
        }}
      >
        {segments.map((s) => (
          <li key={s.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              aria-hidden
              style={{ width: 10, height: 10, borderRadius: 2, background: s.color, display: "inline-block" }}
            />
            <span style={{ opacity: 0.85 }}>{s.label}</span>
            <span style={{ marginLeft: "auto", opacity: 0.7 }}>{s.gb.toFixed(2)} GB</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
