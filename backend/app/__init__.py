"""Gemma Compute Monitor — FastAPI backend application package."""

# ``__version__`` is a static string literal only; it is intentionally not
# computed at runtime (e.g. from importlib.metadata) so that importing this
# package stays cheap and free of side effects.
#
# Keep this package marker minimal and side-effect-free: do NOT import
# submodules (``.main``, ``.model_loader``, ``.sse``) or heavy third-party
# dependencies (``jax``, ``gemma``) here. Eager imports would slow tooling,
# break cheap test collection, and could trigger JAX/model initialization
# prematurely. The one-time model load must happen only inside the FastAPI
# lifespan hook in ``app.main``.
__version__ = "0.1.0"
