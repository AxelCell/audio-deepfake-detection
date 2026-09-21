"""Present so that pytest puts the repo root on sys.path.

Without this file, pytest's default import mode adds only tests/ to sys.path
and `from src.protocol import ...` fails with ModuleNotFoundError. An empty
conftest.py at the root is the conventional fix; there is no other reason for
this file to exist.
"""
