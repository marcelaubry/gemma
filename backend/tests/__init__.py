"""Test package for the Gemma Compute Monitor FastAPI backend (pytest).

This module is a package marker only. Its sole purpose is to mark
``backend/tests`` as a Python package so that pytest's default ("prepend")
import mode adds ``backend`` -- the first ancestor directory without an
``__init__.py`` -- to ``sys.path``. With ``backend`` on ``sys.path`` the test
modules import the application under test via ``import app`` (for example,
``from app.main import app``) and reference one another as ``tests.test_*``.

It is intentionally free of imports, fixtures, and executable statements so
that collection stays cheap and never triggers a model load or a JAX/Metal
import. Shared fixtures belong in ``backend/tests/conftest.py``, not here.
"""
