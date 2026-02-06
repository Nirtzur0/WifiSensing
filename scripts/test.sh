#!/usr/bin/env bash
set -euo pipefail

# Keep pytest deterministic across dev machines that may have globally-installed
# pytest plugins (some of which are incompatible with Python 3.12).
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

exec python3 -m pytest "$@"

