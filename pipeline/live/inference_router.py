"""
SMART_1 Inference Router — CONS ML (Meta-v7 Refined) + AGGR Mechanical (v1.12).

Encapsulates the decisions that turn raw events into trade orders:

- CONS path: load Meta-v7 Refined config; gate DEMA-cross signals through
  conservative_brain.predict with dynamic per-session_cluster thresholds.
- AGGR path: mechanical risk_pts <= cap filter on pullback events (no ML).
- Position queue: single position at a time; if CONS holds, AGGR is skipped.
- Daily kill switch: combined PnL <= -$700 over the CT trading day halts new
  entries (existing positions still exit normally).

This module does no I/O against the live broker. It only decides whether to
take a candidate signal. Caller wires entries/exits/state-persistence.

Mode tag on the active position drives exit logic (CONS = trend-flip / SL;
AGGR = v1.12 fixed SL+TP+timeout). Exit logic lives at the call site; the
router only exposes `current_position_mode()` so the caller can branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.live.pullback_detector import PullbackEvent


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONS_MODEL = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"
DEFAULT_CONS_CONFIG = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json"

CONS_FEATURES = (
    "entry_adx",
    "cci_abs",
    "st_gap_ratio",
    "efficiency_ratio",
    "volatility_zscore",
    "session_cluster",
)

# Defaults match the validated simulation
# (`SIM_CONS_ML_AGGR_MECH_MGC_*d.json`).
DEFAULT_RISK_CAP_PTS = 12.0
DEFAULT_DAILY_CAP_USD = -700.0


@dataclass
class RouteDecision:
    take: bool
    mode: str            # "CONS", "AGGR", or "" (when not taken)
    reason: str
    prob: float = float("nan")
    threshold: float = float("nan")
    risk_pts: float = float("nan")


def _ct_trading_day(ts_utc: pd.Timestamp) -> "object":
    """Topstep trading day: 5pm-3:10pm CT. Subtract 15h10m from CT time."""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_ct = ts_utc.tz_convert("America/Chicago")
    return (ts_ct - pd.Timedelta(hours=15, minutes=10)).date()


class InferenceRouter:
    """Stateful router. One instance per live process."""

    def __init__(
        self,
        cons_model_path: Path = DEFAULT_CONS_MODEL,
        cons_config_path: Path = DEFAULT_CONS_CONFIG,
        risk_cap_pts: float = DEFAULT_RISK_CAP_PTS,
        daily_cap_usd: float = DEFAULT_DAILY_CAP_USD,
    ) -> None:
        self.cons_brain = lgb.Booster(model_file=str(cons_model_path))
        cfg = json.loads(Path(cons_config_path).read_text())
        self._threshold_map: dict[int, float] = {
            int(k): float(v) for k, v in cfg["thresholds"].items()
        }
        self.risk_cap_pts = float(risk_cap_pts)
        self.daily_cap_usd = float(daily_cap_usd)

        # State
        self._position_mode: Optional[str] = None   # None | "CONS" | "AGGR"
        self._daily_pnl: dict = {}                  # trade_day -> pnl_usd

    # ── state helpers ────────────────────────────────────────────────────

    def current_position_mode(self) -> Optional[str]:
        return self._position_mode

    def on_entry(self, mode: str) -> None:
        assert mode in ("CONS", "AGGR")
        self._position_mode = mode

    def on_exit(self, ts_utc: pd.Timestamp, pnl_usd: float) -> None:
        day = _ct_trading_day(ts_utc)
        self._daily_pnl[day] = self._daily_pnl.get(day, 0.0) + float(pnl_usd)
        self._position_mode = None

    def daily_pnl(self, ts_utc: pd.Timestamp) -> float:
        return float(self._daily_pnl.get(_ct_trading_day(ts_utc), 0.0))

    def _daily_cap_blocks(self, ts_utc: pd.Timestamp) -> bool:
        return self.daily_pnl(ts_utc) <= self.daily_cap_usd

    def threshold_for(self, session_cluster: int) -> float:
        return self._threshold_map.get(int(session_cluster), 0.50)

    # ── routing ──────────────────────────────────────────────────────────

    def route_cons(
        self,
        *,
        ts_utc: pd.Timestamp,
        features: dict,
    ) -> RouteDecision:
        if self._position_mode is not None:
            return RouteDecision(False, "", f"queue_busy:{self._position_mode}")
        if self._daily_cap_blocks(ts_utc):
            return RouteDecision(False, "", "daily_cap")

        x = np.array([[float(features[f]) for f in CONS_FEATURES]], dtype=float)
        prob = float(self.cons_brain.predict(x)[0])
        thr = self.threshold_for(int(features["session_cluster"]))
        if prob < thr:
            return RouteDecision(False, "", "ml_reject", prob=prob, threshold=thr)
        return RouteDecision(True, "CONS", "ok", prob=prob, threshold=thr)

    def route_aggr(
        self,
        *,
        ts_utc: pd.Timestamp,
        event: PullbackEvent,
    ) -> RouteDecision:
        if self._position_mode is not None:
            return RouteDecision(False, "", f"queue_busy:{self._position_mode}",
                                 risk_pts=event.risk_pts)
        if self._daily_cap_blocks(ts_utc):
            return RouteDecision(False, "", "daily_cap", risk_pts=event.risk_pts)
        if event.risk_pts > self.risk_cap_pts:
            return RouteDecision(False, "", "risk_cap_exceeded",
                                 risk_pts=event.risk_pts)
        return RouteDecision(True, "AGGR", "ok", risk_pts=event.risk_pts)
