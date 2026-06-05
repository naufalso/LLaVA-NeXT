#!/bin/bash
# Setup script to install lmms-eval from the local git submodule.
#
# Usage:
#   bash scripts/setup_lmms_eval.sh
#
# This script initialises the lmms-eval submodule (if not already present)
# and installs it in editable (development) mode so that local changes to the
# submodule are immediately reflected at runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Initializing and updating lmms-eval submodule ..."
git -C "$REPO_ROOT" submodule update --init --recursive -- lmms-eval

echo "==> Installing lmms-eval from submodule in editable mode ..."
pip install -e "$REPO_ROOT/lmms-eval"

echo "==> lmms-eval installed successfully from submodule."
echo "    Location: $REPO_ROOT/lmms-eval"
python -c "import lmms_eval; print(f'    Version : {lmms_eval.__version__}')" 2>/dev/null || true
