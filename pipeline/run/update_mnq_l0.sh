#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/kemal/futures"
PY="/usr/bin/python3"

cd "$BASE_DIR"
mkdir -p _LOG

notify_failure() {
  local status=$?
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] MNQ L0 update failed status=$status"
  "$PY" -u pipeline/mnq_ml/notify_l0_failure.py --exit-code "$status" --log "$BASE_DIR/_LOG/mnq_l0_update.log" || true
  exit "$status"
}

trap notify_failure ERR

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] MNQ L0 update start"

echo "[mnq_l0] refresh recent 1m from yfinance"
"$PY" -u pipeline/mnq_ml/fetch_yfinance_1m.py --replace-yfinance

echo "[mnq_l0] rebuild derived 5m/15m"
"$PY" -u pipeline/mnq_ml/build_derived_timeframes.py --force

echo "[mnq_l0] audit 1m continuity/integrity"
"$PY" -u pipeline/mnq_ml/audit_l0_duckdb.py

echo "[mnq_l0] audit 5m/15m yfinance parity"
"$PY" -u pipeline/mnq_ml/audit_yfinance_timeframe_parity.py

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] MNQ L0 update done"
