# Gemma Compute Monitor — Frontend

> The Next.js 16 (App Router) dashboard for the **Gemma Compute Monitor**. It
> connects to a locally running FastAPI backend (exposed over an **ngrok** HTTPS
> tunnel), streams per-transformer-layer telemetry for a **Gemma 3 4B** model,
> and visualizes it across **four animated panels plus a per-token compute-cost
> strip** — all behind a single shared-password gate, in a dark-only theme.
> It is deployed to a live, clickable **Railway** URL.

This is the **frontend deep-dive**. For the system-wide master document — full
end-to-end setup, the backend, the per-session ngrok workflow, and Railway
deployment — see [`../COMPUTE_MONITOR.md`](../COMPUTE_MONITOR.md). For the
backend deep-dive, see [`../backend/README.md`](../backend/README.md). This
README is self-sufficient for running and deploying the frontend; it defers the
full system narrative to `../COMPUTE_MONITOR.md` rather than duplicating it.

The frontend is a **pure client** of the backend HTTP/SSE API — it never imports
backend code. The dashboard is built with **native browser APIs only**: the bar
chart and gauges are CSS/SVG, and the GPU waveform is an HTML `<canvas>` driven
by `requestAnimationFrame`. There are **no third-party charting libraries**.

## Contents

- [Prerequisites](#prerequisites)
- [Local development](#local-development)
- [Environment variables](#environment-variables)
- [The four panels + per-token cost](#the-four-panels--per-token-cost)
- [Dark palette](#dark-palette)
- [API contracts consumed](#api-contracts-consumed)
- [Resilience messaging](#resilience-messaging)
- [Railway deployment](#railway-deployment)
- [Testing](#testing)
- [Project structure](#project-structure)

## Prerequisites

- **Node.js 20 LTS (>= 20.9.0)** and npm. Next.js 16 requires Node 20+.
- A running backend reachable over **HTTPS**. Locally, expose the FastAPI
  server (port 8000) with `ngrok http 8000` (see the per-session workflow
  below). For pure local development you may instead point at
  `http://localhost:8000`.

## Local development

From the `frontend/` directory:

1. Install dependencies:
   ```bash
   npm install
   ```
2. Create your local environment file and set the two required variables:
   ```bash
   cp .env.example .env.local
   # then edit .env.local:
   #   NEXT_PUBLIC_API_URL=<your ngrok HTTPS URL, or http://localhost:8000>
   #   NEXT_PUBLIC_PASSWORD=<the shared gate password>
   ```
3. Start the dev server. Next.js 16 uses **Turbopack by default** — no
   `--turbopack` flag is needed:
   ```bash
   npm run dev
   ```
   Open <http://localhost:3000>.
4. Enter the password (`NEXT_PUBLIC_PASSWORD`) at the gate; the dashboard then
   renders.

> **Note:** `NEXT_PUBLIC_*` values are inlined into the client bundle at
> **build time**. If you change `NEXT_PUBLIC_API_URL` or `NEXT_PUBLIC_PASSWORD`,
> restart `npm run dev` locally (or **redeploy** on Railway) for the new value
> to take effect.

## Environment variables

Both are also documented in [`.env.example`](./.env.example).

| Variable | Required | Purpose |
|----------|----------|---------|
| `NEXT_PUBLIC_API_URL` | Yes | Base URL of the backend (the **ngrok HTTPS URL**, no trailing slash). Used for `GET {NEXT_PUBLIC_API_URL}/health` and `POST {NEXT_PUBLIC_API_URL}/analyze`. |
| `NEXT_PUBLIC_PASSWORD` | Yes | The shared gate password, compared against a value persisted in the browser's `localStorage`. |

> **`NEXT_PUBLIC_*` are build-time-inlined and browser-visible — not secret.**
> They are baked into the client bundle when `next build` runs, so they are
> readable in the browser, and changing them on Railway requires a **redeploy**
> to take effect. The password gate is a lightweight shared-access gate, **not**
> real authentication (multi-user sessions and backend auth are out of scope).

## The four panels + per-token cost

The dashboard renders four real-time panels driven by the SSE telemetry stream,
plus a final per-token compute-cost strip:

1. **Layer Activity Bar Chart** — one bar per transformer layer. The layer count
   is derived **dynamically from the telemetry stream** (Gemma 3 4B has **34**
   transformer layers), never a hardcoded value. Bar height is proportional to
   `gpu_pct`; the just-completed bar is highlighted in the primary accent
   `#6366f1`, previously completed bars dim to 40% opacity, labels read
   `L1…Ln`, and the active layer counter uses `#f59e0b`. Rendered with CSS/SVG.
2. **Unified Memory Breakdown** — a stacked bar of Model Weights (~8.5 GB,
   static), KV Cache (`kv_cache_gb`, updates per event), Activations
   (`activation_gb`, updates per event), and OS/Other (static), shown against
   the **48 GB** unified-memory pool using the healthy accent `#10b981`.
   Rendered with CSS/SVG.
3. **GPU Utilization Waveform** — a scrolling HTML `<canvas>` plotting `gpu_pct`
   over a 10-second sliding window, updated on each event via
   `requestAnimationFrame` so the main thread is never blocked.
4. **Per-Token Compute Cost** — rendered only after the final `{ done: true }`
   event arrives. One chip per generated token, chip brightness scaling linearly
   with `per_token_ms` relative to the maximum.

## Dark palette

The dashboard is **dark-only** — there is no light mode. The palette is fixed
and defined as CSS custom properties in [`app/globals.css`](./app/globals.css):

| Token | Hex | Role |
|-------|-----|------|
| Background | `#0f1117` | Page background |
| Panels | `#1e2130` | Panel / card surfaces |
| Borders | `#2d3348` | Borders |
| Primary accent | `#6366f1` | Primary accent (active layer, buttons) |
| Secondary accent | `#8b5cf6` | Secondary accent |
| Healthy / active | `#10b981` | Memory / healthy indicators |
| Layer counter | `#f59e0b` | Active layer counter |

## API contracts consumed

The frontend consumes four **immutable** wire contracts, mirrored byte-for-byte
in TypeScript in `lib/types.ts`:

**`GET /health`** response:

```json
{ "status": "ready" | "loading", "model": "gemma-3-4b" }
```

**`POST /analyze`** request body:

```json
{ "prompt": "string" }
```

The `/analyze` response is a Server-Sent Events stream of one **layer event**
per transformer layer, terminated by a final **per-token event**:

```json
{ "layer": 0, "gpu_pct": 0.0, "cpu_pct": 0.0, "memory_used_gb": 0.0,
  "kv_cache_gb": 0.0, "activation_gb": 0.0, "elapsed_ms": 0.0 }
```

```json
{ "per_token_ms": [0.0], "done": true }
```

`layer` is the **0-based** transformer layer index; the UI displays it as
`L{layer + 1}` (e.g. `L1…L34` for Gemma 3 4B).

> **Transport note (important):** the native browser `EventSource` API is
> **GET-only** and cannot send a POST body, so the dashboard consumes the SSE
> stream using the **Fetch streaming API** (`fetch` POST + a `ReadableStream`
> reader) rather than `EventSource`, with no third-party SSE library. This logic
> lives in `lib/useEventSource.ts`.

## Resilience messaging

The UI surfaces two operator-facing messages verbatim:

- On a `/health` timeout or error:

  > Backend offline — start the local server and update the ngrok URL in Railway

- If `/analyze` does not begin streaming within 5 seconds:

  > Model warming up, this may take 20–40 seconds on first run.

## Railway deployment

**A live, clickable Railway URL is the mandatory deliverable.** Deployment is
configured by [`railway.json`](./railway.json):

- **Builder:** `NIXPACKS`
- **Build command:** `npm install && npm run build`
- **Start command:** `npm run start` (Next.js binds to Railway's injected
  `PORT`; no port is hardcoded)
- **Restart policy:** `ON_FAILURE`, up to 10 retries, single replica

Steps:

1. Create a Railway service from this repository with the **root directory set
   to `frontend/`**.
2. Set the service environment variables `NEXT_PUBLIC_API_URL` (your current
   ngrok HTTPS URL) and `NEXT_PUBLIC_PASSWORD` **before** building — they are
   inlined into the client bundle at build time.
3. Deploy. Railway provisions a live `*.up.railway.app` URL.

### Per-session ngrok workflow

The backend runs locally and is reached through an ngrok tunnel whose HTTPS URL
**changes each session**. At the start of **every** backend session:

1. Run `ngrok http 8000` and copy the HTTPS URL.
2. Update `NEXT_PUBLIC_API_URL` in the Railway environment variables.
3. Trigger a Railway redeploy.

Step 3 is **required** because `NEXT_PUBLIC_*` values are build-time-inlined — a
new ngrok URL does not reach the live frontend until a redeploy rebuilds the
bundle.

### CORS

The backend enforces a CORS allow-list. The deployed Railway origin must be
present in the backend's `CORS_ALLOW_ORIGINS` (see
[`../backend/README.md`](../backend/README.md)), otherwise the browser blocks
the cross-origin `/health` and `/analyze` requests.

## Testing

From the `frontend/` directory:

```bash
npm test
```

Tests use **Jest + React Testing Library** (wired through `next/jest`) and live
in `frontend/__tests__/`. They cover the **auth gate** (correct / incorrect
password behavior) and **panel animation** (panels render and update from mock
telemetry events). The HTML `<canvas>` is mocked via `jest-canvas-mock`.

## Project structure

```
frontend/
├── app/
│   ├── layout.tsx        # root layout (imports globals.css)
│   ├── page.tsx          # composes <PasswordGate><Dashboard/></PasswordGate>
│   └── globals.css       # dark-only palette tokens + shared component styles
├── components/
│   ├── PasswordGate.tsx        # single-shared-password access gate
│   ├── Dashboard.tsx           # orchestrates the SSE stream + panels
│   ├── LayerActivityBarChart.tsx
│   ├── MemoryBreakdown.tsx
│   ├── GpuWaveform.tsx
│   └── PerTokenCost.tsx
├── lib/
│   ├── types.ts          # TypeScript mirrors of the four immutable contracts
│   ├── useEventSource.ts # Fetch-streaming SSE client hook
│   └── api.ts            # /health polling + resilience messaging
├── package.json
├── next.config.js
├── tsconfig.json
├── railway.json
└── .env.example
```

Modules beyond `PasswordGate.tsx`, `app/globals.css`, `lib/types.ts`, and the
configuration files are delivered in later checkpoints; the structure above is
the complete intended frontend.
