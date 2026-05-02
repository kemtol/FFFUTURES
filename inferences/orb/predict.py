"""Inference: score a new ORB breakout event with trained LGBM models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.features.builder import FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"
TARGETS = ["y_60m", "y_120m", "y_240m"]


@dataclass
class BreakoutScore:
    breakout_ts: pd.Timestamp
    session: str
    orb_tf: str
    side: int
    p_60m: float
    p_120m: float
    p_240m: float

    def is_valid(self, threshold: float = 0.6) -> bool:
        return all(p >= threshold for p in [self.p_60m, self.p_120m, self.p_240m])


def load_models() -> dict:
    models = {}
    for target in TARGETS:
        path = MODELS_DIR / f"lgbm_{target}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train_lgbm.py first.")
        with open(path, "rb") as f:
            models[target] = pickle.load(f)
    return models


def score(features_row: pd.DataFrame, models: dict | None = None) -> BreakoutScore:
    if models is None:
        models = load_models()

    X = features_row[FEATURE_COLS]
    return BreakoutScore(
        breakout_ts=features_row["_breakout_ts"].iloc[0],
        session=features_row["_session"].iloc[0],
        orb_tf=features_row["_orb_tf"].iloc[0],
        side=int(features_row["breakout_side"].iloc[0]),
        p_60m=float(models["y_60m"].predict(X)[0]),
        p_120m=float(models["y_120m"].predict(X)[0]),
        p_240m=float(models["y_240m"].predict(X)[0]),
    )
