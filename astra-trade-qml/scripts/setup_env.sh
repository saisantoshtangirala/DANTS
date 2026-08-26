#!/usr/bin/env bash
# Sets up a Python virtualenv for either the training or paper-trading environment.
# Usage: ./scripts/setup_env.sh [training|paper]
set -euo pipefail

MODE="${1:-paper}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$MODE" != "training" && "$MODE" != "paper" ]]; then
  echo "Usage: $0 [training|paper]" >&2
  exit 1
fi

python3 -m venv "$ROOT_DIR/.venv"
source "$ROOT_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements/requirements-$MODE.txt"

echo "Environment ready. Activate with: source $ROOT_DIR/.venv/bin/activate"
