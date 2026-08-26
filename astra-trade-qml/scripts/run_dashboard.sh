#!/usr/bin/env bash
# Launches the Streamlit dashboard.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python3 -m src.main --mode dashboard
