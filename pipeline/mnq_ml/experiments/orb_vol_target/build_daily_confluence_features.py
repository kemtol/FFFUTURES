#!/usr/bin/env python3
"""Build daily no-lookahead confluence features for MNQ ORB.

For MNQ trade date D, every feature in this file is computed from external
daily yfinance data with date < D. This is regime/context, not intraday
confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_SOURCE_DB = "data/Level_0_Raw/yfinance_daily.duckdb"

DAILY_CONFLUENCE_FEATURES = [
    "dc_spy_return_1d",
    "dc_spy_return_5d",
    "dc_spy_return_20d",
    "dc_spy_dist_sma20",
    "dc_spy_dist_sma50",
    "dc_spy_realized_vol_20d",
    "dc_qqq_return_1d",
    "dc_qqq_return_5d",
    "dc_qqq_return_20d",
    "dc_qqq_dist_sma20",
    "dc_qqq_dist_sma50",
    "dc_qqq_realized_vol_20d",
    "dc_qqq_minus_spy_return_1d",
    "dc_qqq_minus_spy_return_5d",
    "dc_qqq_minus_spy_return_20d",
    "dc_qqq_spy_beta_60d",
    "dc_qqq_beta_residual_5d",
    "dc_qqq_beta_residual_20d",
    "dc_vix_prev_close",
    "dc_vix_change_1d",
    "dc_vix_change_5d",
    "dc_vix_percentile_20d",
    "dc_vix_percentile_60d",
    "dc_vix_dist_sma20",
    "dc_tnx_prev_close",
    "dc_tnx_change_1d",
    "dc_tnx_change_5d",
    "dc_dxy_return_1d",
    "dc_dxy_return_5d",
]

FEATURE_DESCRIPTIONS = {
    "dc_spy_return_1d": "SPY one-day return through the prior external daily close; broad-market short-term tone.",
    "dc_spy_return_5d": "SPY five-trading-day return through D-1; broad-market short trend.",
    "dc_spy_return_20d": "SPY twenty-trading-day return through D-1; monthly broad-market regime.",
    "dc_spy_dist_sma20": "SPY distance from its 20D moving average through D-1; broad-market trend stretch.",
    "dc_spy_dist_sma50": "SPY distance from its 50D moving average through D-1; broader trend regime.",
    "dc_spy_realized_vol_20d": "SPY 20D daily return volatility through D-1; broad-market realized-volatility background.",
    "dc_qqq_return_1d": "QQQ one-day return through D-1; Nasdaq/tech short-term tone.",
    "dc_qqq_return_5d": "QQQ five-trading-day return through D-1; short-term Nasdaq trend.",
    "dc_qqq_return_20d": "QQQ twenty-trading-day return through D-1; monthly Nasdaq regime.",
    "dc_qqq_dist_sma20": "QQQ distance from its 20D moving average through D-1; Nasdaq trend stretch.",
    "dc_qqq_dist_sma50": "QQQ distance from its 50D moving average through D-1; broader Nasdaq trend regime.",
    "dc_qqq_realized_vol_20d": "QQQ 20D daily return volatility through D-1; Nasdaq realized-volatility background.",
    "dc_qqq_minus_spy_return_1d": "QQQ 1D return minus SPY 1D return; Nasdaq one-day relative strength.",
    "dc_qqq_minus_spy_return_5d": "QQQ 5D return minus SPY 5D return; short-term Nasdaq leadership.",
    "dc_qqq_minus_spy_return_20d": "QQQ 20D return minus SPY 20D return; monthly Nasdaq leadership.",
    "dc_qqq_spy_beta_60d": "Rolling 60D beta of QQQ daily returns to SPY daily returns through D-1.",
    "dc_qqq_beta_residual_5d": "QQQ 5D return minus beta-adjusted SPY 5D return; beta-adjusted short-term leadership.",
    "dc_qqq_beta_residual_20d": "QQQ 20D return minus beta-adjusted SPY 20D return; beta-adjusted monthly leadership.",
    "dc_vix_prev_close": "VIX prior daily close; volatility/fear level before the trade day.",
    "dc_vix_change_1d": "VIX one-day point change through D-1; fresh volatility pressure.",
    "dc_vix_change_5d": "VIX five-day point change through D-1; short-term volatility pressure.",
    "dc_vix_percentile_20d": "VIX percentile versus its trailing 20D window through D-1; one-month vol regime.",
    "dc_vix_percentile_60d": "VIX percentile versus its trailing 60D window through D-1; three-month vol regime.",
    "dc_vix_dist_sma20": "VIX distance from its 20D moving average through D-1; volatility stretch.",
    "dc_tnx_prev_close": "10Y yield prior daily close; rates level before the trade day.",
    "dc_tnx_change_1d": "10Y yield one-day point change through D-1; fresh rates pressure.",
    "dc_tnx_change_5d": "10Y yield five-day point change through D-1; short-term rates trend.",
    "dc_dxy_return_1d": "DXY one-day return through D-1; fresh USD pressure.",
    "dc_dxy_return_5d": "DXY five-trading-day return through D-1; short-term USD trend.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def percentile_last(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float((values <= values[-1]).mean())


def load_daily_prices(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise SystemExit(f"Missing yfinance daily DB: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    raw = con.execute(
        """
        select date, symbol, adj_close, close
        from daily_ohlcv
        where symbol in ('SPY', 'QQQ', 'VIX', 'TNX', 'DXY')
        order by date, symbol
        """
    ).fetchdf()
    con.close()
    if raw.empty:
        raise SystemExit(f"No daily rows found in {db_path}")
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw["price"] = raw["adj_close"].fillna(raw["close"]).astype(float)
    wide = raw.pivot(index="date", columns="symbol", values="price").sort_index()
    required = ["SPY", "QQQ", "VIX", "TNX", "DXY"]
    missing = [col for col in required if col not in wide.columns]
    if missing:
        raise SystemExit(f"Missing required daily confluence symbols: {missing}")
    return wide[required].dropna()


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=prices.index)
    spy_ret = prices["SPY"].pct_change()
    qqq_ret = prices["QQQ"].pct_change()
    dxy_ret = prices["DXY"].pct_change()

    for symbol, ret in [("spy", spy_ret), ("qqq", qqq_ret)]:
        price = prices[symbol.upper()]
        features[f"dc_{symbol}_return_1d"] = price.pct_change(1)
        features[f"dc_{symbol}_return_5d"] = price.pct_change(5)
        features[f"dc_{symbol}_return_20d"] = price.pct_change(20)
        features[f"dc_{symbol}_dist_sma20"] = price / price.rolling(20, min_periods=10).mean() - 1.0
        features[f"dc_{symbol}_dist_sma50"] = price / price.rolling(50, min_periods=25).mean() - 1.0
        features[f"dc_{symbol}_realized_vol_20d"] = ret.rolling(20, min_periods=10).std(ddof=0)

    features["dc_qqq_minus_spy_return_1d"] = features["dc_qqq_return_1d"] - features["dc_spy_return_1d"]
    features["dc_qqq_minus_spy_return_5d"] = features["dc_qqq_return_5d"] - features["dc_spy_return_5d"]
    features["dc_qqq_minus_spy_return_20d"] = features["dc_qqq_return_20d"] - features["dc_spy_return_20d"]

    beta = qqq_ret.rolling(60, min_periods=30).cov(spy_ret) / spy_ret.rolling(60, min_periods=30).var()
    features["dc_qqq_spy_beta_60d"] = beta
    features["dc_qqq_beta_residual_5d"] = features["dc_qqq_return_5d"] - beta * features["dc_spy_return_5d"]
    features["dc_qqq_beta_residual_20d"] = features["dc_qqq_return_20d"] - beta * features["dc_spy_return_20d"]

    vix = prices["VIX"]
    features["dc_vix_prev_close"] = vix
    features["dc_vix_change_1d"] = vix.diff(1)
    features["dc_vix_change_5d"] = vix.diff(5)
    features["dc_vix_percentile_20d"] = vix.rolling(20, min_periods=10).apply(percentile_last, raw=True)
    features["dc_vix_percentile_60d"] = vix.rolling(60, min_periods=30).apply(percentile_last, raw=True)
    features["dc_vix_dist_sma20"] = vix / vix.rolling(20, min_periods=10).mean() - 1.0

    tnx = prices["TNX"]
    features["dc_tnx_prev_close"] = tnx
    features["dc_tnx_change_1d"] = tnx.diff(1)
    features["dc_tnx_change_5d"] = tnx.diff(5)

    dxy = prices["DXY"]
    features["dc_dxy_return_1d"] = dxy_ret
    features["dc_dxy_return_5d"] = dxy.pct_change(5)

    return features[DAILY_CONFLUENCE_FEATURES].reset_index(names="daily_confluence_feature_date")


def build_for_trade_dates(feature_frame: pd.DataFrame, trade_dates: pd.Series) -> pd.DataFrame:
    feature_frame = feature_frame.copy()
    feature_frame["daily_confluence_feature_date"] = pd.to_datetime(feature_frame["daily_confluence_feature_date"]).dt.date
    feature_frame = feature_frame.sort_values("daily_confluence_feature_date").reset_index(drop=True)
    clean = feature_frame.dropna(subset=DAILY_CONFLUENCE_FEATURES).copy()
    if clean.empty:
        raise SystemExit("Daily confluence feature frame has no complete rows.")

    rows: list[dict[str, Any]] = []
    dates = clean["daily_confluence_feature_date"].to_list()
    for ny_date in sorted(pd.to_datetime(trade_dates).dt.date.unique()):
        pos = np.searchsorted(dates, ny_date, side="left") - 1
        if pos < 0:
            raise SystemExit(f"No prior daily confluence row available for MNQ date {ny_date}")
        row = clean.iloc[int(pos)].to_dict()
        row["ny_date"] = ny_date
        rows.append(row)
    out = pd.DataFrame(rows)
    ordered = ["ny_date", "daily_confluence_feature_date"] + DAILY_CONFLUENCE_FEATURES
    return out[ordered].sort_values("ny_date").reset_index(drop=True)


def summarize(out: pd.DataFrame, source_db: Path) -> dict[str, Any]:
    return {
        "source_db": str(source_db),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "feature_count": int(len(DAILY_CONFLUENCE_FEATURES)),
        "min_ny_date": str(out["ny_date"].min()),
        "max_ny_date": str(out["ny_date"].max()),
        "min_feature_date": str(out["daily_confluence_feature_date"].min()),
        "max_feature_date": str(out["daily_confluence_feature_date"].max()),
        "feature_nulls": {str(k): int(v) for k, v in out[DAILY_CONFLUENCE_FEATURES].isna().sum().to_dict().items()},
        "lookahead_violations": int((out["daily_confluence_feature_date"] >= out["ny_date"]).sum()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    source_db = project_path(args.source_db)
    l1_context = project_path(cfg["outputs"]["l1_context"])
    output_path = project_path(cfg["outputs"]["daily_confluence"])
    manifest_path = project_path(cfg["outputs"]["daily_confluence_manifest"])

    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing feature file: {output_path}")
    if not l1_context.exists():
        raise SystemExit(f"Missing MNQ L1 context: {l1_context}")

    l1 = pd.read_parquet(l1_context, columns=["ny_date"])
    prices = load_daily_prices(source_db)
    feature_frame = build_feature_frame(prices)
    out = build_for_trade_dates(feature_frame, l1["ny_date"])
    bad_nulls = out[DAILY_CONFLUENCE_FEATURES].isna().sum()
    bad_nulls = bad_nulls[bad_nulls > 0]
    if not bad_nulls.empty:
        raise SystemExit(f"Unexpected daily confluence nulls: {bad_nulls.to_dict()}")
    violations = out[out["daily_confluence_feature_date"] >= out["ny_date"]]
    if len(violations):
        raise SystemExit(f"Daily confluence lookahead violations: {len(violations)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "output": str(output_path),
        "feature_columns": DAILY_CONFLUENCE_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "no_lookahead_contract": "For MNQ trade date D, features use the latest complete external daily row with date < D.",
        **summarize(out, source_db),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
