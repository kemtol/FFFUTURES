"""LGBM training pipeline for ORB breakout prediction."""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

FEATURE_COLS = [
    "orb_range", "orb_range_atr_ratio", "breakout_side", "breakout_strength",
    "breakout_candle_volume_ratio", "session", "orb_tf", "time_in_session",
    "day_of_week", "htf_trend_1h", "htf_trend_4h", "atr_14_1m",
    "price_vs_prev_close_pct",
]

TARGETS = ["y_60m", "y_120m", "y_240m"]

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def train(features: pd.DataFrame, test_months: int = 6) -> dict[str, lgb.Booster]:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    features = features.dropna(subset=FEATURE_COLS)
    features = features.sort_values("_breakout_ts")

    cutoff = features["_breakout_ts"].max() - pd.DateOffset(months=test_months)
    train_df = features[features["_breakout_ts"] < cutoff]
    test_df  = features[features["_breakout_ts"] >= cutoff]

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    models = {}
    for target in TARGETS:
        print(f"\n--- Training {target} ---")
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[target]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df[target]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

        params = {
            "objective":    "binary",
            "metric":       "auc",
            "learning_rate": 0.05,
            "num_leaves":   31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":  5,
            "verbose":      -1,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, preds)
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, (preds >= 0.5).astype(int)))

        # Save model
        model_path = MODELS_DIR / f"lgbm_{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save feature importance
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        imp.to_csv(REPORTS_DIR / f"feature_importance_{target}.csv", index=False)

        models[target] = model

    return models


if __name__ == "__main__":
    import sqlite3
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from ml.features.session import SESSION_BOUNDARIES
    from ml.features.orb import compute_orb, detect_breakouts
    from ml.labels.outcome import label_outcomes
    from ml.features.builder import build_features

    print("Loading 1m data...")
    conn = sqlite3.connect("data/MGC_1m.db")
    df_1m = pd.read_sql("SELECT * FROM investing_ohlcv_1m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms", conn)
    conn.close()

    conn5 = sqlite3.connect("data/MGC_5m.db")
    df_5m = pd.read_sql("SELECT * FROM investing_ohlcv_5m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms", conn5)
    conn5.close()

    conn15 = sqlite3.connect("data/MGC_15m.db")
    df_15m = pd.read_sql("SELECT * FROM investing_ohlcv_15m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms", conn15)
    conn15.close()

    all_breakouts = []
    for session in ["tokyo", "london", "us"]:
        for orb_tf, src_df in [("5m", df_5m), ("15m", df_15m), ("30m", df_15m)]:
            print(f"Computing ORB {session} {orb_tf}...")
            orb = compute_orb(src_df, session, orb_tf)
            bo  = detect_breakouts(src_df, orb)
            all_breakouts.append(bo)

    breakouts = pd.concat(all_breakouts, ignore_index=True)
    print(f"Total breakouts: {len(breakouts)}")

    print("Labeling outcomes...")
    labeled = label_outcomes(breakouts, df_1m)

    print("Building features...")
    features = build_features(labeled, df_1m)

    print("Training models...")
    train(features)
