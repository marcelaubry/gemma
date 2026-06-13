"""FastAPI application entrypoint for the Gemma Compute Monitor backend.

This module wires the whole backend together and exposes its **public HTTP
surface**. It is the object uvicorn serves: the ASGI application is named
``app`` so the canonical launch command resolves ``app.main:app``::

    # from inside backend/
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Responsibilities (AAP §0.5.2 / agent prompt Phases 2-6)
-------------------------------------------------------
1. **Lifespan model load.** A FastAPI lifespan hook triggers the *one-time*
   Gemma 3 4B load (``app.model_loader.load_model``) at startup, flipping the
   readiness flag ``loading -> ready``. The load runs as a **background task**
   (off the event loop, on a worker thread) so the server starts accepting
   connections immediately and ``GET /health`` stays responsive (well under
   500 ms) during the ~20-40 s first load. A load failure is logged but never
   crashes the server: ``/health`` keeps reporting ``"loading"`` so the
   frontend can show its "model warming up" / "backend offline" messaging.
2. **CORS allow-list.** ``CORSMiddleware`` is configured from
   ``app.config.settings.cors_allow_origins`` — the live Railway origin
   (supplied via the ``CORS_ALLOW_ORIGINS`` env var) plus permissive
   local-development origins. Credentials are allowed, so a bare ``"*"`` is
   intentionally *never* used.
3. ``GET /health`` ->
   ``{ "status": "ready" | "loading", "model": "gemma-3-4b" }``.
4. ``POST /analyze`` -> ``503`` while the model is still loading; otherwise a
   **single-flight** guard returns ``429`` if an analysis is already running;
   otherwise a Server-Sent-Events stream
   (``sse_starlette.sse.EventSourceResponse``) built by ``app.sse``.

Single-flight guard (agent prompt Phase 4)
------------------------------------------
Exactly one ``/analyze`` may run at a time. A module-level
:data:`analyze_lock` (an :class:`asyncio.Lock`) enforces this: a second
concurrent request returns ``429`` *immediately* (it never waits/queues). The
lock is acquired only on the uncontended fast path — we check
``analyze_lock.locked()`` and raise ``429`` instead of ever awaiting a held
lock — which keeps the lock loop-agnostic and therefore safe to reuse across
the independent event loops created by successive test clients.

Because the response is a *streaming* ``EventSourceResponse``, the lock cannot
be released when the request handler returns (the stream is produced
afterwards). Instead the lock release is tied to the **stream's lifecycle**:
the response's ``body_iterator`` is wrapped in an async generator whose
``finally`` releases the lock. That ``finally`` runs on normal completion, on
an error, and on ``aclose()`` (which ``EventSourceResponse`` invokes when the
client disconnects) — so the lock can never leak and strand the endpoint in a
permanent ``429``.

Design constraints
------------------
* **No heavy import at module load.** ``app.sse`` (transitively) imports
  ``jax`` at module scope, so it is imported **lazily inside** the ``/analyze``
  handler — only after the readiness and single-flight guards pass. Importing
  this module (``from app.main import app``) therefore never triggers a
  JAX/Metal initialization and never loads the model; the load happens only in
  the lifespan hook at server startup. The cheap siblings
  (``config``, ``model_loader``, ``schemas``) are imported normally.
* **No non-Metal GPU backend** is referenced anywhere (the Apple
  Silicon Metal JAX backend is the only one targeted, per the AAP).
* **No persistent state.** Telemetry is ephemeral; nothing is written to disk
  or a database. The only process state is the in-memory loaded model held by
  ``app.model_loader`` and the transient single-flight lock here.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Sibling imports. Only the CHEAP modules are imported at module scope:
# config/model_loader/schemas pull in no JAX/gemma. ``app.sse`` is imported
# lazily inside ``/analyze`` (see the handler) because it transitively imports
# ``jax`` and would otherwise make ``from app.main import app`` expensive.
from app import config, model_loader, schemas

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Single-flight concurrency guard (agent prompt Phase 4)
# -----------------------------------------------------------------------------
analyze_lock = asyncio.Lock()
"""Process-wide single-flight guard for ``POST /analyze``.

Exactly one analysis may run at a time. The endpoint checks
:meth:`asyncio.Lock.locked` and returns ``429`` for a second concurrent
request instead of ever awaiting the held lock, so the lock is only ever
acquired on its uncontended fast path. (On CPython that fast path does not bind
the lock to the running event loop, which keeps a single module-level lock
reusable across the separate event loops spun up by successive test clients.)
The lock is released when the SSE stream finishes — see
:func:`_release_analyze_lock` and the ``body_iterator`` wrapper in
:func:`analyze`."""


def _release_analyze_lock() -> None:
  """Release :data:`analyze_lock` if (and only if) it is currently held.

  Guarding on :meth:`asyncio.Lock.locked` makes the release **idempotent**:
  calling it when the lock is already free is a no-op rather than a
  ``RuntimeError`` ("Lock is not acquired."). ``asyncio.Lock`` does not track an
  owning task, so releasing from the streaming context (a different task than
  the request handler that acquired it) is valid.
  """
  if analyze_lock.locked():
    analyze_lock.release()


# -----------------------------------------------------------------------------
# Lifespan: one-time background model load (agent prompt Phase 2)
# -----------------------------------------------------------------------------
async def _load_model_in_background() -> None:
  """Run the blocking one-time model load off the event loop.

  ``model_loader.load_model`` is synchronous and blocking (the first load can
  take ~20-40 s and drives JAX), so it is dispatched to a worker thread via
  ``anyio.to_thread.run_sync`` to keep the event loop — and therefore
  ``GET /health`` — responsive throughout. Any failure (e.g. an unset
  ``GEMMA_WEIGHTS_PATH`` or a checkpoint error) is logged and swallowed here:
  the server must keep serving ``/health`` (which then reports ``"loading"``)
  rather than crash, so the frontend can surface its resilience messaging. The
  load is idempotent, so re-entry is harmless.
  """
  try:
    await anyio.to_thread.run_sync(model_loader.load_model)
  except Exception:  # pylint: disable=broad-exception-caught
    # Do NOT crash the server: /health must keep reporting "loading" so the
    # frontend shows "Model warming up" / "Backend offline" guidance and
    # /analyze keeps returning 503 (not 500).
    logger.exception(
        "Background model load failed; backend will keep reporting"
        " status='loading'. Fix GEMMA_WEIGHTS_PATH / the checkpoint and"
        " restart to retry."
    )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
  """FastAPI lifespan: kick off the one-time model load, then serve.

  On startup the model load is launched as a **non-blocking background task**
  (``asyncio.create_task``) so the application begins accepting requests
  immediately while the model warms up; ``GET /health`` reports ``"loading"``
  until the task flips the flag to ``"ready"``. The task handle is stashed on
  ``fastapi_app.state.model_load_task`` for observability. On shutdown a
  still-pending task is cancelled and awaited so no orphaned work outlives the
  server.

  Args:
    fastapi_app: The FastAPI application instance (provided by the framework).
      Named ``fastapi_app`` rather than ``app`` to avoid shadowing the
      module-level :data:`app` singleton.

  Yields:
    Control to the running server for the duration of its lifetime.
  """
  logger.info(
      "Starting Gemma Compute Monitor backend; launching one-time model load"
      " in the background (model id=%s).",
      config.MODEL_ID,
  )
  task = asyncio.create_task(
      _load_model_in_background(), name="gemma-model-load"
  )
  # Expose the load task for observability/diagnostics (no persistent state).
  fastapi_app.state.model_load_task = task
  try:
    yield
  finally:
    # Graceful shutdown: cancel the load task if it is still in flight so it is
    # never left dangling, then await it (suppressing the resulting cancellation
    # / any late error) for a clean teardown.
    if not task.done():
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        logger.debug("Model load task cancelled during shutdown.")
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug(
            "Model load task raised during shutdown cancellation.",
            exc_info=True,
        )


# -----------------------------------------------------------------------------
# Security/hardening response headers (QA FINAL-ACCEPTANCE Issue 7)
# -----------------------------------------------------------------------------
class SecurityHeadersMiddleware:
  """Pure-ASGI middleware adding conservative security headers to every response.

  Implemented as a **raw ASGI** middleware (deliberately NOT
  ``starlette.middleware.base.BaseHTTPMiddleware``) so it never buffers or
  otherwise interferes with the streaming ``EventSourceResponse`` served by
  ``POST /analyze``: it only rewrites the header list on the single
  ``http.response.start`` event and passes every other ASGI event — including
  every SSE ``http.response.body`` chunk — through untouched. The 429
  single-flight rejection and the client-disconnect cleanup are likewise
  unaffected, since neither the request stream nor the body stream is wrapped.

  Each header is applied with :meth:`MutableHeaders.setdefault`, so any value a
  handler already chose is preserved — most importantly the SSE response's own
  ``Cache-Control`` (``no-cache``) and ``Content-Type`` (``text/event-stream``)
  are left intact. ``Strict-Transport-Security`` is emitted **only** when the
  request arrived over HTTPS — directly (``scope['scheme'] == 'https'``) or via
  a TLS-terminating proxy such as ngrok/Railway (``X-Forwarded-Proto: https``) —
  so plain-HTTP local development never receives an HSTS policy.

  Headers set (defaults only):
    * ``X-Content-Type-Options: nosniff`` — disable MIME sniffing.
    * ``X-Frame-Options: DENY`` — disallow framing (clickjacking defense).
    * ``Referrer-Policy: no-referrer`` — never leak the URL as a referrer.
    * ``Cache-Control: no-store`` — this telemetry API is stateless and must
      not be cached (no-op for the SSE stream, which already sets no-cache).
    * ``Strict-Transport-Security`` — HTTPS transports only.
  """

  def __init__(self, app: ASGIApp) -> None:
    self.app = app

  async def __call__(
      self, scope: Scope, receive: Receive, send: Send
  ) -> None:
    # Only HTTP responses carry headers; pass websockets/lifespan straight
    # through so this middleware is a transparent no-op for them.
    if scope["type"] != "http":
      await self.app(scope, receive, send)
      return

    # Treat the request as secure if it arrived over TLS directly or was
    # forwarded as HTTPS by a TLS-terminating proxy (ngrok / Railway).
    is_https = scope.get("scheme") == "https"
    request_headers = dict(scope.get("headers") or [])
    forwarded_proto = request_headers.get(b"x-forwarded-proto")
    if forwarded_proto is not None:
      first_proto = forwarded_proto.split(b",", 1)[0].strip().lower()
      if first_proto == b"https":
        is_https = True

    async def send_with_security_headers(message: Message) -> None:
      if message["type"] == "http.response.start":
        headers = MutableHeaders(scope=message)
        headers.setdefault("x-content-type-options", "nosniff")
        headers.setdefault("x-frame-options", "DENY")
        headers.setdefault("referrer-policy", "no-referrer")
        # setdefault preserves the SSE response's own Cache-Control (no-cache);
        # plain JSON responses (e.g. /health) get this no-store default.
        headers.setdefault("cache-control", "no-store")
        if is_https:
          headers.setdefault(
              "strict-transport-security",
              "max-age=63072000; includeSubDomains",
          )
      await send(message)

    await self.app(scope, receive, send_with_security_headers)


# -----------------------------------------------------------------------------
# Application object (MUST be named ``app`` -> ``app.main:app``)
# -----------------------------------------------------------------------------
app = FastAPI(title="Gemma Compute Monitor", lifespan=lifespan)
"""The ASGI application served by uvicorn (``uvicorn app.main:app``)."""

# CORS allow-list (agent prompt Phase 3). ``cors_allow_origins`` is an explicit,
# de-duplicated list — the Railway origin (from the ``CORS_ALLOW_ORIGINS`` env
# var) plus the permissive local-development origins — never a bare ``"*"``,
# which would be incompatible with ``allow_credentials=True``.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security/hardening headers (QA FINAL-ACCEPTANCE Issue 7) on EVERY response —
# including the streaming SSE /analyze response and error responses. Added after
# CORSMiddleware; Starlette wraps the most-recently-added middleware OUTERMOST,
# so on the response path this runs after CORS has set its headers and uses
# setdefault, never clobbering the CORS or SSE-specific headers.
app.add_middleware(SecurityHeadersMiddleware)


# -----------------------------------------------------------------------------
# GET /health (agent prompt Phase 5)
# -----------------------------------------------------------------------------
@app.get("/health", response_model=schemas.HealthResponse)
async def health() -> schemas.HealthResponse:
  """Report model-load readiness for the frontend health poll.

  Returns the immutable health contract
  ``{ "status": "ready" | "loading", "model": "gemma-3-4b" }``. The handler does
  **no blocking work** — it only reads the in-memory readiness flag — so it
  responds well within the mandated 500 ms even while the model is still
  loading in the background.

  Returns:
    A :class:`app.schemas.HealthResponse` whose ``status`` mirrors
    :func:`app.model_loader.get_status` and whose ``model`` is the fixed
    :data:`app.config.MODEL_ID` (``"gemma-3-4b"``).
  """
  return schemas.HealthResponse(
      status=model_loader.get_status(),
      model=config.MODEL_ID,
  )


# -----------------------------------------------------------------------------
# POST /analyze (agent prompt Phase 6)
# -----------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(req: schemas.AnalyzeRequest, request: Request):
  """Stream per-layer telemetry for ``req.prompt`` over Server-Sent Events.

  Guard order (all three behaviours are part of the binding contract):

  1. **Readiness (``503``).** If the model is not loaded yet
     (:func:`app.model_loader.is_ready` is ``False``), reject with
     ``503 Service Unavailable`` — the frontend keeps polling ``/health`` until
     ``ready``.
  2. **Single-flight (``429``).** If an analysis is already running
     (:data:`analyze_lock` is held), reject the second concurrent request
     *immediately* with ``429 Too Many Requests`` (never queue/await).
  3. **Stream.** Otherwise acquire the lock and return the
     :class:`~sse_starlette.sse.EventSourceResponse` produced by
     :func:`app.sse.analyze_response` — one event per transformer layer
     followed by a final per-token event.

  The single-flight lock is released when the SSE stream ends. Because the
  stream is produced *after* this coroutine returns, the response's
  ``body_iterator`` is wrapped so its ``finally`` releases the lock on normal
  completion, on error, and on ``aclose()`` (client disconnect). ``app.sse`` is
  imported lazily here so that importing this module never pulls in JAX/gemma.

  Args:
    req: The validated request body (``{ "prompt": string }``, non-empty).
    request: The Starlette request, forwarded to the SSE layer for
      client-disconnect detection.

  Returns:
    An :class:`~sse_starlette.sse.EventSourceResponse` streaming the telemetry.

  Raises:
    HTTPException: ``503`` while the model is loading, or ``429`` if an analysis
      is already in progress.
  """
  # (1) Readiness guard -> 503 while the one-time model load is still pending.
  if not model_loader.is_ready():
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model is loading",
    )

  # (2) Single-flight guard -> 429 if another analysis is already streaming.
  # Checked BEFORE acquiring so a second concurrent caller is rejected
  # immediately instead of waiting for the lock.
  if analyze_lock.locked():
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="An analysis is already in progress",
    )

  # Acquire on the uncontended fast path (the lock is free here; there is no
  # await between the locked() check above and this acquire, so no other
  # coroutine can slip in within this single-threaded event loop).
  await analyze_lock.acquire()

  # (3) Build the SSE response. Lazy import keeps JAX/gemma out of module load.
  try:
    from app import sse  # pylint: disable=import-outside-toplevel

    response = sse.analyze_response(request, req.prompt)
  except Exception:
    # Never leak the lock if constructing the response fails.
    _release_analyze_lock()
    raise

  # Tie lock release to the STREAM lifecycle. ``EventSourceResponse`` streams by
  # iterating ``self.body_iterator`` and calls its ``aclose()`` on client
  # disconnect, so wrapping it guarantees release on completion, error, AND
  # disconnect — preventing a permanent 429.
  inner = getattr(response, "body_iterator", None)
  if inner is None:
    # Defensive: a non-streaming response has no work to span, so release now
    # rather than risk stranding the lock. (Real ``EventSourceResponse`` always
    # exposes ``body_iterator``.)
    _release_analyze_lock()
    return response

  async def _release_when_stream_ends() -> AsyncIterator:
    """Re-yield every SSE chunk, releasing the lock when the stream ends."""
    try:
      async for chunk in inner:
        yield chunk
    finally:
      # Release the single-flight lock first so the endpoint is immediately
      # available again, then proactively close the wrapped iterator so the
      # SSE generator's own cleanup (signalling its worker thread to stop) runs
      # promptly rather than waiting for garbage collection.
      _release_analyze_lock()
      aclose = getattr(inner, "aclose", None)
      if aclose is not None:
        try:
          await aclose()
        except Exception:  # pylint: disable=broad-exception-caught
          logger.debug(
              "Error while closing the SSE iterator during cleanup.",
              exc_info=True,
          )

  response.body_iterator = _release_when_stream_ends()
  return response


# Public API. ``app`` is the ASGI object served by uvicorn; ``lifespan`` and
# ``analyze_lock`` are exported for tests and observability.
__all__ = ["app", "lifespan", "analyze_lock"]
