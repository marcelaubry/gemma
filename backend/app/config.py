"""Backend settings for the Gemma Compute Monitor (env-var configuration hub).

This module is the **single, side-effect-free source of truth** for every piece
of runtime configuration the FastAPI telemetry backend needs. All values are
read once, at import time, from process environment variables — which the
operator populates from ``backend/.env`` (see ``backend/.env.example`` for the
canonical template documenting each variable). No ``.env`` file is parsed here:
``python-dotenv`` is intentionally **not** a dependency, so the operator or
process manager (uvicorn, the shell, Railway, etc.) is responsible for exporting
the variables before the app starts.

Consumers
---------
* ``app.model_loader`` reads :data:`GEMMA_WEIGHTS_PATH`
  (``settings.gemma_weights_path``) to restore the Orbax checkpoint via
  ``gm.ckpts.load_params(...)``.
* ``app.main`` reads :data:`CORS_ALLOW_ORIGINS` for the CORS allow-list,
  :data:`PORT` for the uvicorn bind port, and :data:`MODEL_ID` for the
  ``GET /health`` response.
* ``app.schemas`` / ``app.metrics`` reference :data:`MODEL_ID`; the
  ``HealthResponse.model`` default mirrors this exact literal.

Environment-variable contract
------------------------------
The names below are a hard, shared contract with ``backend/.env.example`` — do
NOT rename any of them.

``GEMMA_WEIGHTS_PATH``
    Absolute filesystem path to the downloaded Gemma 3 4B Orbax checkpoint
    directory. **Required at model-load time**, but intentionally *optional at
    import time*: it defaults to an empty string so tooling and tests can import
    this module (and the rest of the package) without a checkpoint present. The
    loud failure for a missing/empty path happens later, in ``app.model_loader``
    when ``gm.ckpts.load_params`` is actually called — never here.
``CORS_ALLOW_ORIGINS``
    Comma-separated allow-list of browser origins permitted to call the API
    (the live Railway frontend origin plus any extra origins). Parsed into a
    de-duplicated ``list[str]`` and merged with the permissive local-development
    origins in :data:`LOCAL_DEV_ORIGINS` so local work needs no configuration.
``PORT``
    Optional TCP port for the ASGI server. Defaults to ``8000``. Parsed to an
    ``int``; a missing, empty, or non-numeric value falls back to the default
    rather than crashing at import.
``ENABLE_PJRT_COMPATIBILITY``
    Intentionally **not** consumed here. It only influences the JAX/Metal
    install on the deployment Mac and has no effect on application logic.

Design notes
------------
* **Side-effect free**: importing this module only reads environment variables
  and builds small in-memory values. It imports nothing from the rest of the
  backend package and no heavy third-party libraries (no ``jax``/``gemma``), so
  it stays cheap to import from tests and tooling.
* **No new dependencies**: only the standard library (``os``, ``dataclasses``)
  is used. ``pydantic-settings`` / ``BaseSettings`` are deliberately avoided
  even though Pydantic ships transitively with FastAPI.
* **No secrets**: telemetry is ephemeral and nothing here is sensitive; the
  weights path is an ordinary local filesystem path, not a credential.

Public API (canonical vs. convenience)
--------------------------------------
The frozen :data:`settings` instance is the **canonical** accessor —
``from app.config import settings`` then ``settings.gemma_weights_path`` etc.
The module-level constants :data:`GEMMA_WEIGHTS_PATH`,
:data:`CORS_ALLOW_ORIGINS`, :data:`PORT`, and :data:`MODEL_ID` are convenience
aliases backed by the very same values, provided so either import style
resolves identically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_ID: str = "gemma-3-4b"
"""Fixed model identifier reported by ``GET /health`` and mirrored by
``app.schemas.HealthResponse.model``. This is a compile-time constant — **not**
an environment variable — and must remain exactly ``"gemma-3-4b"`` to satisfy
the immutable health-response contract (AAP §0.1.2)."""

LOCAL_DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
"""Permissive local-development browser origins for the Next.js dev server
(``next dev`` listens on port 3000). These are always merged into the CORS
allow-list so a developer can run the frontend locally without configuring
``CORS_ALLOW_ORIGINS``. The live Railway origin is supplied separately through
the environment variable and explicitly allow-listed (AAP §0.5.2)."""

_DEFAULT_PORT: int = 8000
"""Fallback ASGI bind port for an unset, empty, or non-numeric ``PORT``."""


# -----------------------------------------------------------------------------
# Parsing helpers (pure functions — deterministic, no direct environment access)
# -----------------------------------------------------------------------------
def _parse_origins(raw: str) -> list[str]:
  """Parse the ``CORS_ALLOW_ORIGINS`` value into a clean allow-list.

  The raw value is a comma-separated string of browser origins (for example
  ``"https://app.up.railway.app, http://localhost:3000"``). Each entry is
  whitespace-stripped and empty entries (from a leading/trailing/duplicate comma
  or an entirely empty value) are dropped. The permissive
  :data:`LOCAL_DEV_ORIGINS` are then appended so local development always works
  without configuration. Order is preserved — environment-supplied origins
  first, local defaults after — and duplicates are removed, keeping the first
  occurrence.

  Args:
    raw: The raw, comma-separated ``CORS_ALLOW_ORIGINS`` environment value. An
      empty string is valid and yields just the local-development defaults.

  Returns:
    A de-duplicated ``list[str]`` of origins with no empty entries, suitable for
    Starlette's ``CORSMiddleware(allow_origins=...)``.
  """
  # Environment origins first (preserving their given order), then the local
  # development defaults. Empties are filtered out during de-duplication below.
  candidates: list[str] = [entry.strip() for entry in raw.split(",")]
  candidates.extend(LOCAL_DEV_ORIGINS)

  seen: set[str] = set()
  origins: list[str] = []
  for origin in candidates:
    if origin and origin not in seen:
      seen.add(origin)
      origins.append(origin)
  return origins


def _parse_port(raw: str | None, default: int = _DEFAULT_PORT) -> int:
  """Parse the ``PORT`` environment value into an integer.

  Kept tolerant so importing this module never raises on a misconfigured or
  unset ``PORT``: a missing (``None``), empty, or non-numeric value falls back
  to ``default`` instead of propagating a ``ValueError`` at import time.

  Args:
    raw: The raw ``PORT`` environment value, or ``None`` when the variable is
      unset.
    default: The port to use when ``raw`` cannot be parsed as an integer.

  Returns:
    The resolved port as an ``int``.
  """
  if raw is None:
    return default
  try:
    return int(raw.strip())
  except ValueError:
    return default


# -----------------------------------------------------------------------------
# Settings object
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
  """Immutable snapshot of backend configuration resolved from the environment.

  Instances are frozen so configuration cannot be mutated after start-up. The
  module-level :data:`settings` singleton is the canonical accessor; construct
  additional instances directly only in tests.

  Attributes:
    gemma_weights_path: Absolute path to the Gemma 3 4B Orbax checkpoint
      directory (from ``GEMMA_WEIGHTS_PATH``). May be an empty string at import
      time; ``app.model_loader`` raises a clear error if it is still empty when
      the model is loaded.
    cors_allow_origins: De-duplicated browser-origin allow-list (parsed from
      ``CORS_ALLOW_ORIGINS`` and merged with :data:`LOCAL_DEV_ORIGINS`).
    port: TCP port the ASGI server binds to (from ``PORT``; defaults to
      ``8000``).
    model_id: Fixed model identifier; defaults to :data:`MODEL_ID` and should
      not be overridden.
  """

  gemma_weights_path: str
  cors_allow_origins: list[str]
  port: int
  model_id: str = MODEL_ID


# -----------------------------------------------------------------------------
# Module-level singleton, built once from the environment at import time.
# -----------------------------------------------------------------------------
settings: Settings = Settings(
    gemma_weights_path=os.environ.get("GEMMA_WEIGHTS_PATH", ""),
    cors_allow_origins=_parse_origins(os.environ.get("CORS_ALLOW_ORIGINS", "")),
    port=_parse_port(os.environ.get("PORT")),
)
"""Canonical, import-time settings singleton. Prefer
``from app.config import settings`` and access fields such as
``settings.gemma_weights_path``."""


# -----------------------------------------------------------------------------
# Convenience module-level aliases (backed by the singleton above).
# -----------------------------------------------------------------------------
# These let callers import individual values directly (e.g.
# ``from app.config import CORS_ALLOW_ORIGINS``) using the SAME names as the
# environment variables they come from. They are exact aliases of the
# corresponding ``settings`` fields, so both access styles always agree.
GEMMA_WEIGHTS_PATH: str = settings.gemma_weights_path
"""Convenience alias of ``settings.gemma_weights_path``."""

CORS_ALLOW_ORIGINS: list[str] = settings.cors_allow_origins
"""Convenience alias of ``settings.cors_allow_origins``."""

PORT: int = settings.port
"""Convenience alias of ``settings.port``."""


__all__ = [
    "Settings",
    "settings",
    "MODEL_ID",
    "LOCAL_DEV_ORIGINS",
    "GEMMA_WEIGHTS_PATH",
    "CORS_ALLOW_ORIGINS",
    "PORT",
]
