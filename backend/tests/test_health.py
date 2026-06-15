"""Tests for ``GET /health`` (mandated checks 4 and 5).

Covers contract shape, 500 ms readiness, ``loading -> ready`` reflection, and
the ``/analyze`` 503-while-loading gate for the Gemma Compute Monitor backend
(``app.main``). This module exercises **two** of the seven mandated checks from
the Agent Action Plan:

* **Check 4** -- ``GET /health`` responds **within 500 ms** and returns the
  EXACT immutable contract
  ``{ status: "ready" | "loading", model: "gemma-3-4b" }``.
* **Check 5** -- ``POST /analyze`` returns **HTTP 503** while the model is still
  loading (the readiness flag is ``"loading"`` / ``is_ready()`` is ``False``).
* Plus: the ``/health`` response **reflects the ``loading -> ready`` readiness
  flag**, proven by parametrizing the shape test over both states.

Why the ``503`` check lives here (with the health/readiness tests) rather than
in ``test_analyze.py``: it validates the very same ``loading -> ready``
readiness flag that ``/health`` reports, so the two checks share one cohesive
readiness story.

Isolation strategy
------------------
Every client is built through the shared :func:`make_client` factory fixture
from ``backend/tests/conftest.py``. That factory constructs a
``fastapi.testclient.TestClient`` for the real ``app.main:app`` **without**
running the FastAPI lifespan and with ``app.model_loader``'s
``load_model``/``get_status``/``is_ready`` monkeypatched, so the real ~8.5 GB
Metal-only Gemma 3 4B model is **never** loaded and readiness is forced
deterministically. Combined with the conftest top-level JAX/gemma import safety
net, these tests never touch JAX, Metal, a checkpoint, or root privileges and
run in milliseconds.

Only ``app.schemas`` is imported directly here: it is pure Pydantic (no
JAX/gemma), so importing it is safe and side-effect free. ``app.main`` is
**never** imported directly -- the client must come from :func:`make_client` so
the conftest safety net and readiness patches are guaranteed to apply.

Run from inside ``backend/``::

    python -m pytest tests/test_health.py -q

Constraints (AAP 0.7): target runtime Python >= 3.12; no CUDA / non-Metal GPU
backend; no root required; no ``gemma/**`` file is modified.
"""

from __future__ import annotations

import time

import pytest

# ``app.schemas`` is pure Pydantic (no jax/gemma), so a direct import is safe
# and lets each test validate the /health body against the REAL immutable
# contract. ``app.main`` is intentionally NOT imported here: the TestClient is
# obtained via the conftest ``make_client`` fixture so the heavy-dependency
# import safety net and the readiness monkeypatches are guaranteed to apply.
from app import schemas

# The immutable model identifier from the GET /health contract (AAP 0.1.2),
# reproduced byte-for-byte. It must equal ``app.config.MODEL_ID`` and the
# ``app.schemas.HealthResponse.model`` default; keeping the literal in one place
# documents the contract and keeps the assertions below self-explanatory.
EXPECTED_MODEL_ID = "gemma-3-4b"

# The two -- and only two -- keys the GET /health JSON body must contain.
EXPECTED_HEALTH_KEYS = {"status", "model"}


# -----------------------------------------------------------------------------
# Check 4 (shape half) + loading -> ready reflection
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["loading", "ready"])
def test_health_contract_shape(make_client, status):
  """``GET /health`` returns EXACTLY the immutable contract in both states.

  Parametrizing over both readiness states does double duty: it pins the
  response *shape* (Check 4) AND proves the body **reflects** the
  ``loading -> ready`` readiness flag, because ``body["status"]`` must equal the
  forced ``status`` for each parameter. Validating the body through the real
  :class:`app.schemas.HealthResponse` guarantees the test fails if anyone
  changes the contract (renames ``model``, drops a field, or allows a status
  other than ``"ready"`` / ``"loading"``).

  Args:
    make_client: Conftest factory building a ``TestClient`` whose readiness is
      forced to ``status`` without loading the real model.
    status: The readiness state under test (``"loading"`` or ``"ready"``).
  """
  client = make_client(status)
  resp = client.get("/health")
  assert resp.status_code == 200
  body = resp.json()
  # EXACTLY the two contract keys -- no more, no less.
  assert set(body.keys()) == EXPECTED_HEALTH_KEYS
  assert body["status"] == status  # reflects the loading -> ready flag
  assert body["model"] == EXPECTED_MODEL_ID  # immutable model id, byte-for-byte
  # Validate against the real Pydantic contract (status Literal["ready",
  # "loading"], model Literal["gemma-3-4b"]); raises ValidationError on drift.
  schemas.HealthResponse(**body)


# -----------------------------------------------------------------------------
# Check 4 (timing half): /health responds within 500 ms in BOTH states
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["loading", "ready"])
def test_health_responds_within_500ms(make_client, status):
  """``GET /health`` stays responsive (< 500 ms) even while the model loads.

  The handler does no blocking work -- it only reads the in-memory readiness
  flag -- so it must answer well within the mandated 500 ms budget (typically
  under 10 ms) in BOTH the ``"loading"`` and ``"ready"`` states. Asserting the
  ``"loading"`` case is the meaningful half: it proves ``/health`` is
  non-blocking and never waits on the (background) model load, which is exactly
  what lets the frontend poll readiness during the ~20-40 s first load.

  Timing is measured ONLY around the request, not around client construction:
  the deterministic conftest stub means no real model load ever competes with
  the response, so the elapsed time reflects the documented non-blocking
  contract rather than fixture setup cost.

  Args:
    make_client: Conftest factory building a ``TestClient`` with forced
      readiness (no real model load).
    status: The readiness state under test (``"loading"`` or ``"ready"``).
  """
  client = make_client(status)
  start = time.perf_counter()
  resp = client.get("/health")
  elapsed = time.perf_counter() - start
  assert resp.status_code == 200
  # Mandated: /health must respond well within 500 ms (typically < 10 ms).
  assert elapsed < 0.5, f"/health took {elapsed * 1000:.1f} ms (>= 500 ms)"


# -----------------------------------------------------------------------------
# Check 5: POST /analyze returns 503 while the model is loading
# -----------------------------------------------------------------------------
def test_analyze_returns_503_while_loading(make_client):
  """``POST /analyze`` is gated with ``503`` until the model is ready.

  With readiness forced to ``"loading"`` (so ``model_loader.is_ready()`` is
  ``False``), a ``POST /analyze`` carrying a VALID body must return ``503``.

  The ``{ "prompt": "hello" }`` body deliberately satisfies
  :class:`app.schemas.AnalyzeRequest` (a non-empty ``prompt`` string) so FastAPI
  request validation passes and the handler reaches the readiness gate
  (``if not model_loader.is_ready(): raise HTTPException(503)``). An empty or
  missing body would short-circuit at validation with ``422`` -- which is NOT
  what this check targets. The status must be ``503`` specifically: not ``422``
  (would mean validation fired first), not ``429`` (the single-flight guard is
  checked only AFTER the readiness gate), and not ``200`` (the stream must never
  start while loading).

  Args:
    make_client: Conftest factory; ``make_client("loading")`` forces
      ``is_ready()`` to ``False``.
  """
  client = make_client("loading")  # is_ready() -> False
  # Body must be valid so we hit the readiness gate, not a 422 validation error.
  resp = client.post("/analyze", json={"prompt": "hello"})
  assert resp.status_code == 503


# -----------------------------------------------------------------------------
# (Recommended) explicit loading -> ready reflection in a single test
# -----------------------------------------------------------------------------
def test_health_reflects_readiness_transition(make_client):
  """``/health`` reports each readiness state as it is set, in order.

  An explicit, self-documenting companion to the parametrized
  :func:`test_health_contract_shape`: it asserts ``"loading"`` BEFORE creating
  the ``"ready"`` client because ``make_client`` re-patches the shared
  ``model_loader.get_status`` stub on each call, so the ``"loading"`` assertion
  must be made while that stub is still in effect. This makes the
  ``loading -> ready`` transition the backend reports over its lifetime obvious
  at a glance.

  Args:
    make_client: Conftest factory building readiness-controlled clients.
  """
  loading_client = make_client("loading")
  assert loading_client.get("/health").json()["status"] == "loading"
  ready_client = make_client("ready")
  assert ready_client.get("/health").json()["status"] == "ready"
