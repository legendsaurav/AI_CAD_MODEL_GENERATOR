"""
tests/conftest.py
=================
Adds the repo root to sys.path so pytest can import all project modules
regardless of which directory pytest is launched from.
"""
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
