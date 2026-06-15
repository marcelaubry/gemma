"use client";

import { useMemo } from "react";
import type { LayerEvent } from "@/lib/types";

const PALETTE = {
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
  counter: "#f59e0b",
} as const;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export interface LayerActivityBarChartProps {
  /** All layer events received so far, in arrival order. */
  layers: LayerEvent[];
  /** The most recently completed layer (highlighted), or null. */
  latestLayer: LayerEvent | null;
}

export default function LayerActivityBarChart({
  layers,
  latestLayer,
}: LayerActivityBarChartProps) {
  // Derive the bar count DYNAMICALLY from telemetry. For Gemma 3 4B this
  // reaches 34 as layers stream in. NEVER hardcode 18 (Gemma 2B / 3 270M).
  const { count, bars } = useMemo(() => {
    if (layers.length === 0) {
      return { count: 0, bars: [] as (LayerEvent | null)[] };
    }
    const maxIndex = layers.reduce((m, l) => Math.max(m, l.layer), -1);
    const total = maxIndex + 1;
    const byIndex = new Map<number, LayerEvent>();
    for (const l of layers) {
      byIndex.set(l.layer, l);
    }
    const arr: (LayerEvent | null)[] = Array.from(
      { length: total },
      (_, i) => byIndex.get(i) ?? null,
    );
    return { count: total, bars: arr };
  }, [layers]);

  const activeIndex = latestLayer ? latestLayer.layer : -1;

  return (
    <section
      data-testid="layer-bar-chart"
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
        <h2 style={{ margin: 0, fontSize: 14 }}>Layer Activity</h2>
        <span data-testid="active-layer-counter" style={{ color: PALETTE.counter, fontSize: 13, fontWeight: 600 }}>
          {activeIndex >= 0 ? `Layer L${activeIndex + 1} / ${count}` : `0 / ${count}`}
        </span>
      </header>

      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 2,
          height: 160,
          width: "100%",
        }}
      >
        {bars.map((ev, i) => {
          const pct = ev ? clamp(ev.gpu_pct, 0, 100) : 0;
          const hasData = ev !== null;
          const isActive = i === activeIndex;
          // Active (just-completed) bar = full primary; prior completed = 40%.
          const opacity = hasData ? (isActive ? 1 : 0.4) : 0.12;
          return (
            <div
              key={i}
              data-testid={`layer-bar-${i}`}
              title={`L${i + 1}: ${pct.toFixed(1)}% GPU`}
              style={{
                flex: "1 1 0",
                minWidth: 2,
                height: `${hasData ? Math.max(pct, 2) : 100}%`,
                background: hasData ? PALETTE.primary : PALETTE.border,
                opacity,
                borderRadius: "2px 2px 0 0",
                transition: "height 120ms ease-out, opacity 120ms ease-out",
              }}
            />
          );
        })}
      </div>

      {/*
        L1…Ln endpoint axis labels. opacity 0.7 (not 0.5) makes the effective
        label color (#e5e7eb blended over the #1e2130 panel) ≈ #a9acb3 = 7.02:1,
        clearing WCAG 2.x AA 4.5:1 for normal text (the prior 0.5 was 4.29:1).
        fontSize is also raised 8 → 10 for legibility of these small endpoint
        labels. The bar opacities (active 1 / prior 0.4) are independent.
      */}
      <div
        style={{
          display: "flex",
          gap: 2,
          marginTop: 4,
          fontSize: 10,
          opacity: 0.7,
          justifyContent: "space-between",
        }}
      >
        <span>L1</span>
        {count > 1 ? <span>{`L${count}`}</span> : null}
      </div>
    </section>
  );
}
