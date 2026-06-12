"use client";

const PALETTE = {
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
} as const;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export interface PerTokenCostProps {
  /** finalEvent.per_token_ms once the stream completes; null/empty hides the panel. */
  perTokenMs: number[] | null;
}

export default function PerTokenCost({ perTokenMs }: PerTokenCostProps) {
  // Rendered ONLY after the { done: true } final event has arrived.
  if (perTokenMs === null || perTokenMs.length === 0) {
    return null;
  }

  const max = perTokenMs.reduce((m, v) => Math.max(m, v), 0);

  return (
    <section
      data-testid="per-token-cost"
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
        <h2 style={{ margin: 0, fontSize: 14 }}>Per-Token Compute Cost</h2>
        <span style={{ fontSize: 13, opacity: 0.7 }}>{perTokenMs.length} tokens</span>
      </header>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {perTokenMs.map((ms, i) => {
          const ratio = max > 0 ? clamp(ms / max, 0, 1) : 0;
          // Brightness scales linearly with per_token_ms relative to the max.
          const opacity = 0.25 + 0.75 * ratio;
          return (
            <span
              key={i}
              data-testid={`token-chip-${i}`}
              title={`Token ${i + 1}: ${ms.toFixed(1)} ms`}
              style={{
                width: 14,
                height: 14,
                borderRadius: 3,
                background: PALETTE.primary,
                opacity,
              }}
            />
          );
        })}
      </div>
    </section>
  );
}
