#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$BASE_DIR/service/.venv/bin/python"

cd "$BASE_DIR"
echo "[$(date '+%Y-%m-%dT%H:%M:%S')] Starting MGC fetch..."
exec "$PY" pipeline/fetch/fetch_mgc_yfinance.py
