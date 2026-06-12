"use client";

import { useEffect, useRef } from "react";
import type { LayerEvent } from "@/lib/types";

const PALETTE = {
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
} as const;

/** Sliding window for the scrolling waveform (AAP: 10-second window). */
const WINDOW_MS = 10_000;
const FALLBACK_W = 600;
const FALLBACK_H = 160;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

interface Sample {
  t: number;
  v: number;
}

export interface GpuWaveformProps {
  /** All layer events so far; each contributes one gpu_pct sample. */
  layers: LayerEvent[];
}

export default function GpuWaveform({ layers }: GpuWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const samplesRef = useRef<Sample[]>([]);
  const lastCountRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  // Append a timestamped gpu_pct sample for every newly-arrived layer event.
  useEffect(() => {
    const now =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    // A new analysis run resets `layers` back to [] (see Dashboard /
    // useEventSource.start). Detect that the array shrank and reset the
    // waveform's per-run state so (a) stale samples from the previous run are
    // discarded immediately and (b) the append cursor restarts from 0.
    // Without this, a second run's growth never exceeds the prior run's stale
    // count and no new samples are appended — the waveform would freeze on the
    // first run's data.
    if (layers.length < lastCountRef.current) {
      lastCountRef.current = 0;
      samplesRef.current = [];
    }
    for (let i = lastCountRef.current; i < layers.length; i++) {
      samplesRef.current.push({ t: now, v: clamp(layers[i].gpu_pct, 0, 100) });
    }
    lastCountRef.current = layers.length;
  }, [layers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    // Guard the 2D context for jsdom/test safety (jsdom has no real canvas).
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) {
      return;
    }

    const resize = () => {
      const cssW = canvas.clientWidth || FALLBACK_W;
      const cssH = canvas.clientHeight || FALLBACK_H;
      const dpr =
        typeof window !== "undefined" && window.devicePixelRatio
          ? window.devicePixelRatio
          : 1;
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => resize());
      resizeObserver.observe(canvas);
    }

    const draw = () => {
      const now =
        typeof performance !== "undefined" ? performance.now() : Date.now();
      const cutoff = now - WINDOW_MS;
      // Prune samples outside the 10s window.
      const pruned = samplesRef.current.filter((s) => s.t >= cutoff);
      samplesRef.current = pruned;

      const w = canvas.clientWidth || FALLBACK_W;
      const h = canvas.clientHeight || FALLBACK_H;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = PALETTE.panel;
      ctx.fillRect(0, 0, w, h);

      // Horizontal gridlines at 25/50/75%.
      ctx.strokeStyle = PALETTE.border;
      ctx.lineWidth = 1;
      for (const frac of [0.25, 0.5, 0.75]) {
        const y = h - frac * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      if (pruned.length > 0) {
        const xFor = (t: number) => ((t - cutoff) / WINDOW_MS) * w;
        const yFor = (v: number) => h - (clamp(v, 0, 100) / 100) * h;

        // Filled area under the curve.
        ctx.beginPath();
        ctx.moveTo(xFor(pruned[0].t), h);
        for (const s of pruned) {
          ctx.lineTo(xFor(s.t), yFor(s.v));
        }
        ctx.lineTo(xFor(pruned[pruned.length - 1].t), h);
        ctx.closePath();
        ctx.fillStyle = "rgba(99, 102, 241, 0.18)";
        ctx.fill();

        // Stroke line.
        ctx.beginPath();
        pruned.forEach((s, i) => {
          const x = xFor(s.t);
          const y = yFor(s.v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = PALETTE.primary;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, []);

  return (
    <section
      data-testid="gpu-waveform"
      style={{
        background: PALETTE.panel,
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 12,
        padding: 16,
        color: "#e5e7eb",
      }}
    >
      <header style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 14 }}>GPU Utilization</h2>
      </header>
      <canvas
        ref={canvasRef}
        data-testid="gpu-waveform-canvas"
        style={{ width: "100%", height: 160, display: "block", borderRadius: 6 }}
      />
    </section>
  );
}
