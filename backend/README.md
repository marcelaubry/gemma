# Gemma Compute Monitor — Backend

The **backend** of the Gemma Compute Monitor is a local **FastAPI** application
that loads **Gemma 3 4B** once at startup, **instruments** its forward pass to
emit one telemetry event after each transformer layer, and streams those events
to the browser over **Server-Sent Events (SSE)**. It runs **locally on a MacBook
Pro Apple Silicon (M-series, 48 GB unified memory recommended, macOS 14+) using
the JAX Metal backend** — there is **no CUDA** anywhere in the dependency tree.

**Core principle — instrument, never reimplement.** This backend composes the
**unmodified** sibling [`gemma`](../gemma) library purely through the canonical
idiom `from gemma import gm` (the same idiom shown in the repository-root
[`README.md`](../README.md)). Model loading, tokenization, and the transformer
forward pass are used **as-is** and only *instrumented* — none of them is
rewritten here. The `gemma/**` source tree is **read/compose-only** and is never
modified by this product.

> **This README is the backend deep-dive.** For the system-wide master document
> — the full end-to-end setup, the per-session **ngrok** workflow, **Railway**
> deployment, the color palette, and the frontend overview — see
> [**`../COMPUTE_MONITOR.md`**](../COMPUTE_MONITOR.md). For the frontend
> deep-dive, see [**`../frontend/README.md`**](../frontend/README.md). This
> document is self-sufficient for **running the backend locally**; it
> deliberately defers the full deployment/ngrok narrative to
> `../COMPUTE_MONITOR.md` rather than duplicating it.

---

## Contents

- [Prerequisites](#prerequisites)
- [Installation (order is significant)](#installation-order-is-significant)
- [Configuration (environment variables)](#configuration-environment-variables)
- [Running the server](#running-the-server)
- [API reference](#api-reference)
- [How instrumentation works](#how-instrumentation-works-instrument-not-reimplement)
- [Module layout](#module-layout)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Hardware** — MacBook Pro with **Apple Silicon** (M-series). **48 GB unified
  memory is recommended**: the Gemma 3 4B weights occupy roughly **8.5 GB** and
  share the single unified-memory pool with the OS, the KV cache, and per-layer
  activations.
- **macOS 14+**.
- **Python >= 3.12** — required and governed by gemma's
  `requires-python = ">=3.12"` (declared in the repository-root
  [`pyproject.toml`](../pyproject.toml)). The backend virtual environment **must**
  target Python 3.12 or newer.
- **Gemma 3 4B Orbax checkpoint** — downloaded locally. Its directory path is
  supplied to the backend through the `GEMMA_WEIGHTS_PATH` environment variable
  (see [Configuration](#configuration-environment-variables)).
- **`powermetrics`** — a **built-in macOS utility** used to read Metal GPU
  utilization. It **requires root**; without elevated privileges the GPU metric
  (`gpu_pct`) falls back to `0.0`. This fallback is **expected** and is the
  practical default unless the backend runs elevated (see
  [Troubleshooting](#troubleshooting)).
- **`ngrok` CLI** *(for the public deployment only)* — exposes the local backend
  on port **8000** to the public Railway frontend through a per-session HTTPS
  tunnel. The full per-session workflow lives in
  [`../COMPUTE_MONITOR.md`](../COMPUTE_MONITOR.md); it is not required to run the
  backend purely locally.

---

## Installation (order is significant)

> **The dependency install ORDER matters.** The Metal JAX backend is
> **version-lock sensitive**, so the steps below must be performed in **exactly**
> the order shown. **`jax-metal` must never be upgraded beyond `0.1.1` without
> explicit user approval.** Every pin is also captured in
> [`backend/requirements.txt`](./requirements.txt).

Run the following **from inside this `backend/` directory**, in this exact order:

```sh
# 1. Create and activate a Python 3.12 virtual environment.
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install the Metal backend FIRST (order matters — version-lock sensitive).
pip install jax-metal==0.1.1

# 3. Install the matching jaxlib AND jax (released in lockstep — keep equal).
#    0.5.0 is the highest `jax-metal==0.1.1`-compatible version (0.4.25 was too
#    old: the Metal PJRT plugin needs jaxlib>=0.4.34); 0.5.1+ break jax-metal.
pip install jaxlib==0.5.0 jax==0.5.0

# 4. Install the web stack / SSE / system-metrics dependencies.
#    (In zsh, quote the bracketed extra: 'uvicorn[standard]==0.49.0'.)
#    fastapi 0.136.3 pulls a FIXED Starlette (>=0.46.0); the old 0.111.0 forced
#    the vulnerable starlette 0.37.2.
pip install fastapi==0.136.3 uvicorn[standard]==0.49.0 sse-starlette==3.4.4 psutil==7.2.2

# 5. Install gemma LAST — used AS-IS (version 4.0.1). The portable PyPI pin is
#    the default and resolves from any directory:
pip install gemma==4.0.1
#    For local development against the IN-REPO gemma source, install it editable
#    from the repository root INSTEAD (its pyproject.toml lives there):
#        pip install -e .          # from the repository root
#        # equivalently, from inside backend/:  pip install -e ..

# 6. Create your local .env and point GEMMA_WEIGHTS_PATH at the checkpoint.
cp .env.example .env
# then edit .env to set GEMMA_WEIGHTS_PATH (and CORS_ALLOW_ORIGINS).
```

All pins are also captured in [`backend/requirements.txt`](./requirements.txt).
Because the manifest now uses the portable `gemma==4.0.1` pin (not a
CWD-relative editable path), it resolves identically from **either** the
repository root **or** this `backend/` directory:

```sh
# from the repository root (matches CI manifest-closure checks):
pip install -r backend/requirements.txt
# ...or from inside backend/:
pip install -r requirements.txt
```

> **⚠️ Compatibility callout (primary dependency risk — RESOLVED).**
> `jax-metal==0.1.1` is compatible only with **older** jax/jaxlib, and **gemma
> 4.0.1 depends on an *unpinned, modern* JAX** — the repository-root
> [`pyproject.toml`](../pyproject.toml) lists a **bare `jax`** with no upper
> bound. The original `jaxlib==0.4.25` pin was **too old** for `jax-metal==0.1.1`
> (the Metal PJRT plugin requires `jaxlib>=0.4.34`), which made the macOS-arm64
> install unsatisfiable. This is now resolved by pinning **`jax==0.5.0` /
> `jaxlib==0.5.0`** (step 3 above), per AAP §0.7.2:
>
> 1. `0.5.0` is the **highest `jax-metal==0.1.1`-compatible** version
>    (community-verified working combination; `0.5.1+` break `jax-metal 0.1.1`).
>    `jax` and `jaxlib` are released in lockstep — keep both at **exactly**
>    `0.5.0`.
> 2. `jax==0.5.0` satisfies gemma 4.0.1's bare `jax` dependency, so the whole
>    backend set resolves on macOS arm64 / Python 3.12.
> 3. If a future gemma release ever needs a JAX newer than `0.5.0`, optionally
>    export `ENABLE_PJRT_COMPATIBILITY=1` and **escalate to the user before
>    changing `jax-metal`** — **never** silently upgrade it.
>
> **No CUDA / `jax[cuda]` may ever appear in the dependency tree.** Only the
> Apple-Silicon Metal JAX backend is permitted.

---

## Configuration (environment variables)

The backend reads its configuration from process environment variables, which
you populate from a local `.env` (copied from
[`backend/.env.example`](./.env.example)). The settings module
[`app/config.py`](./app/config.py) reads these **exact** variable names.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMMA_WEIGHTS_PATH` | **Yes** | — | Absolute local filesystem path to the downloaded **Gemma 3 4B Orbax checkpoint** directory. Loaded once during startup via `gm.ckpts.load_params(GEMMA_WEIGHTS_PATH, text_only=True)`. |
| `CORS_ALLOW_ORIGINS` | No | local dev origins | **Comma-separated** allow-list of browser origins permitted to call the API — the **live Railway origin** plus your local dev origin (e.g. `https://<your-app>.up.railway.app,http://localhost:3000`). |
| `PORT` | No | `8000` | TCP port the ASGI server binds to. The ngrok tunnel targets this same port. |
| `ENABLE_PJRT_COMPATIBILITY` | No | *(commented out)* | **Leave unset by default.** Only relevant to the JAX/Metal install on the deployment Mac (set to `1` if a newer but still `jax-metal==0.1.1`-compatible jaxlib ≈ `0.5.0` is used). It has **no effect on application logic** and must **never** be used as a workaround for upgrading `jax-metal`. |

> **CORS.** The backend explicitly **allow-lists the Railway origin** supplied
> through `CORS_ALLOW_ORIGINS`, and additionally permits **permissive local
> development origins** (e.g. `http://localhost:3000`) so a developer can run the
> frontend locally without extra configuration. The single password gate is a
> lightweight **frontend** access control — it is **not** backend authentication.

---

## Running the server

From **inside this `backend/` directory** (with the virtual environment
activated and `GEMMA_WEIGHTS_PATH` set):

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### What happens at startup

When the server boots, a FastAPI **lifespan hook** performs the **one-time**
model load and then flips an internal readiness flag from `loading` → `ready`:

```python
from gemma import gm

model  = gm.nn.Gemma3_4B(text_only=True)
params = gm.ckpts.load_params(GEMMA_WEIGHTS_PATH, text_only=True)
```

- Passing `text_only=True` **strips the default SigLiP vision encoder**, since
  this is a text-only telemetry workload.
- The model loads **once** and is held in application state for the lifetime of
  the process — it is never reloaded per request.
- The **first** load can take **~20–40 seconds**.

While the model is still `loading`:

- `GET /health` returns `{ "status": "loading", "model": "gemma-3-4b" }`.
- `POST /analyze` returns **HTTP `503`** until the model is `ready`.

---

## API reference

> The four JSON contracts below are **immutable** — they must not drift between
> the backend and the frontend, and are reproduced **byte-for-byte**.

### `GET /health`

Reports readiness. Responds well within **500 ms** (it does no model work — it
just reads the readiness flag).

**Response:**

```
{ status: "ready" | "loading", model: "gemma-3-4b" }
```

### `POST /analyze`

Streams per-layer telemetry over **Server-Sent Events**. The stream consists of
**one layer event per transformer layer** — **34 for Gemma 3 4B**, derived
**dynamically** from the loaded model configuration (`config.num_layers`) and
**never hardcoded** — terminated by a single final **per-token event**.

**Request body:**

```
{ prompt: string }
```

**SSE layer event** (emitted once after each transformer layer):

```
{ layer: int, gpu_pct: float, cpu_pct: float, memory_used_gb: float, kv_cache_gb: float, activation_gb: float, elapsed_ms: float }
```

**SSE final event** (terminates the stream):

```
{ per_token_ms: float[], done: true }
```

**Behavior:**

- Returns **`503`** while the model is still loading.
- A **second concurrent `POST /analyze` is rejected with `HTTP 429`** — a
  **single-flight** guard ensures **exactly one analysis runs at a time**.
- The stream **stops cleanly on client disconnect** (closing the browser's
  `EventSource` halts the generator).

---

## How instrumentation works (instrument, not reimplement)

gemma's `Transformer.__call__` is **JIT-compiled as a single XLA program** (the
`nn.jit` decorator in [`gemma/gm/nn/_transformer.py`](../gemma/gm/nn/_transformer.py)).
Inside that compiled region, host-side telemetry **cannot** run between layers —
the whole forward pass is one fused device computation.

To obtain per-layer telemetry without rewriting any model code, the backend
instead **reuses the loaded `Block` submodules and weights** and drives the
per-layer loop **eagerly** on the host, mirroring the loop in
`Transformer._apply_attention`. After each layer it calls
**`jax.block_until_ready(x)`** on that layer's output — the **Metal-compatible
barrier** that flushes the layer's device compute before host metrics are
sampled:

```python
import jax

# Eager, host-driven loop over the SAME Block submodules gemma already built.
for i in range(config.num_layers):          # 34 for Gemma 3 4B — never hardcoded
    cache_i, x = model.blocks[i].apply(block_params, x, positions, cache_i, mask)
    jax.block_until_ready(x)                 # Metal-compatible barrier — flush layer
    # ...sample host metrics, then emit ONE layer event...
```

Key points:

- The **attention math is never rewritten** — only the *orchestration* (the loop
  that walks layer by layer) is externalized. The `Block` modules and their
  loaded weights are gemma's own.
- The barrier is **`jax.block_until_ready(x)`**, **not** `jax.effects_barrier()`:
  `effects_barrier()` waits for *ordered side-effects* (e.g. `jax.debug.print`),
  which is the wrong primitive for forcing per-layer compute to finish.
- The layer count comes from **`config.num_layers`** (which is **34** for
  Gemma 3 4B), read from the loaded configuration — **never** a hardcoded
  constant such as 18 (18 belongs to Gemma 2B / Gemma 3 270M, **not** Gemma 3 4B).
- Per-token timing for the final event is produced by driving gemma's own
  sampler (`gm.text.Sampler` / `gm.text.ChatSampler`); blocking JAX work runs off
  the event loop so the SSE generator never stalls the server.

### Metrics

| Field | Source |
| --- | --- |
| `cpu_pct` | `psutil` process/system CPU percentage. |
| `memory_used_gb` | `psutil` resident memory, in GB. |
| `gpu_pct` | Parsed GPU active residency from `powermetrics --samplers gpu_power` (200 ms timeout). **Falls back to `0.0`** when `powermetrics` is unavailable or lacks root. |
| `kv_cache_gb` | Computed from the loaded config using **grouped-query attention** head counts: `num_kv_heads = 4`, `head_dim = 256`, `num_layers = 34`, in **bf16** (2 bytes), across keys and values. |
| `activation_gb` | Estimated from the current layer's intermediate tensor shapes. |
| *(model weights)* | Treated as **static (~8.5 GB)** — they do not change across layers, so they are not re-measured per event. |

> The KV-cache estimate uses **`num_kv_heads = 4`** (grouped-query attention),
> **not** `num_heads = 8`. Using the query-head count would overestimate the
> cache; Gemma 3 4B shares 4 KV heads across its 8 query heads.

---

## Module layout

```
backend/
├── app/
│   ├── __init__.py          # package marker
│   ├── config.py            # settings: GEMMA_WEIGHTS_PATH, CORS_ALLOW_ORIGINS, PORT, MODEL_ID
│   ├── model_loader.py      # builds Gemma3_4B(text_only=True), loads params, holds readiness flag + config
│   ├── instrumentation.py   # eager per-layer harness (reuses gemma Block modules + block_until_ready)
│   ├── metrics.py           # psutil CPU/RAM + powermetrics GPU + KV-cache / activation memory math
│   ├── schemas.py           # Pydantic models for the four immutable contracts
│   ├── sse.py               # async generator → EventSourceResponse
│   └── main.py              # FastAPI app: lifespan model load, CORS, routes, single-flight 429 guard
├── tests/
│   ├── __init__.py
│   ├── test_analyze.py      # SSE layer-event count/schema, 429 concurrency, powermetrics 0.0 fallback
│   └── test_health.py       # /health readiness within 500 ms
├── requirements.txt         # exact pins (install order matters) + editable gemma
└── .env.example             # environment-variable template
```

- **`config.py`** — settings (environment-variable hub).
- **`model_loader.py`** — performs the one-time load and exposes the loaded
  `config` so the layer count and KV-cache dimensions are read **dynamically**.
- **`instrumentation.py`** — the eager per-layer harness (see
  [How instrumentation works](#how-instrumentation-works-instrument-not-reimplement)).
- **`metrics.py`** — `psutil` sampling, the `powermetrics` subprocess, and the
  memory math.
- **`schemas.py`** — Pydantic models reproducing the four immutable contracts.
- **`sse.py`** — the async generator wired to `sse-starlette`'s
  `EventSourceResponse`, running blocking JAX work off the event loop.
- **`main.py`** — the FastAPI app: lifespan model load, the CORS allow-list, the
  `GET /health` and `POST /analyze` routes, and the single-flight `429` guard.

---

## Testing

Run the test suite **from inside `backend/`** (with the virtual environment
activated):

```sh
python -m pytest        # or simply: pytest
```

The tests cover the mandated checks:

- **Layer-event count** — parameterized to **`config.num_layers` (34)**, never a
  hardcoded `18`.
- **Layer-event schema validity** — every layer event matches the immutable
  layer-event contract.
- **`/health` readiness within 500 ms.**
- **The `429` concurrency rejection** — a second concurrent `/analyze` while one
  is in flight is rejected (single-flight guard).
- **The `powermetrics`-unavailable `0.0` fallback** — `gpu_pct` degrades to
  `0.0` when `powermetrics` cannot be invoked.

---

## Troubleshooting

The frontend surfaces two **resilience messages** (reproduced here **verbatim**
so backend operators recognize them):

- On a `/health` timeout or error:

  > Backend offline — start the local server and update the ngrok URL in Railway

- If `/analyze` does not begin streaming within 5 seconds:

  > Model warming up, this may take 20–40 seconds on first run

### Common issues

- **GPU shows 0%.** `powermetrics` requires **root**; without `sudo` the
  `gpu_pct` metric falls back to `0.0` (the subprocess uses a **200 ms
  timeout**). This is **expected** unless the backend runs elevated.
- **`jax-metal` / `jaxlib` import or load failure.** This is the version-lock
  issue. **Do NOT upgrade `jax-metal`.** Pin `jax`/`jaxlib` to the **highest
  `jax-metal==0.1.1`-compatible** version (approximately `0.5.0`, kept equal to
  each other), optionally export `ENABLE_PJRT_COMPATIBILITY=1`, and **escalate to
  the user** before changing `jax-metal`. Never add CUDA / `jax[cuda]` to the
  dependency tree.
- **First request is slow / returns `503`.** The model loads at startup
  (**~20–40 s on first run**). `GET /health` returns `loading` and
  `POST /analyze` returns `503` until the model is `ready`.
- **`429` on `/analyze`.** Another analysis is already in flight — this is the
  **single-flight** guard working as designed (exactly one analysis at a time).
  Retry once the current stream completes.
- **`GEMMA_WEIGHTS_PATH` unset or invalid.** The lifespan model load fails. Set a
  valid absolute checkpoint path in your `.env` (see
  [Configuration](#configuration-environment-variables)).
- **Layer-count expectation (34, not 18).** Gemma 3 4B has **34** transformer
  layers, so the stream emits **34** layer events and the dashboard renders 34
  bars. The value `18` belongs to Gemma 2B / Gemma 3 270M, **not** Gemma 3 4B.
  The count is derived dynamically from `config.num_layers`, never hardcoded.

---

*The [`gemma/**`](../gemma) library in this repository is **read/compose-only**
and is **never modified** by the Gemma Compute Monitor. This backend
**instruments** gemma; it does **not** reimplement model loading, tokenization,
or the forward pass. For the system-wide overview, ngrok workflow, and Railway
deployment, see [`../COMPUTE_MONITOR.md`](../COMPUTE_MONITOR.md); for the
frontend, see [`../frontend/README.md`](../frontend/README.md).*
