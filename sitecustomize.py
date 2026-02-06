"""
Test runner hardening.

Some developer machines have globally-installed pytest plugins that can break
collection under newer Python versions (notably: langsmith + pydantic v1 on 3.12).

Python imports `sitecustomize` automatically if it's importable on `sys.path`.
Because the repo root is on `sys.path` when running `python -m pytest` from the
project directory, this ensures the test suite is runnable without asking
contributors to tweak their global environment.
"""

import os

# Disable auto-loading *external* pytest plugins. Projects that need plugins
# should list them explicitly (and CI should install them).
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

