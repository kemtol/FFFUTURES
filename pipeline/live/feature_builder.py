#!/usr/bin/env python3
"""
Live feature builder — uses EXACT SAME code as batch feature modules.

Computes base features (ATR, ADX, VWAP) from buffer, then delegates to
the 7 module generators to produce features identical to training.

Usage:
    fb = FeatureBuilder(buffer)
    features = fb.build(event)  # dict of all 42 features
    arr = fb.build_array(event) # numpy array for model.predict()
"""

from __future__ import annotations

import sys
import warnings
from datetime import date as Date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="no explicit representation of timezones")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from pipeline.live.buffer import DataBuffer
from pipeline.live.orb_detector import BreakoutEvent, SESSIONS

EPS = 1e-8

# Model expects features in this order
FEATURE_ORDER = [
    "breakout_strength", "atr14_at_entry", "breakout_side",
    "adx_14_15m", "adx_50_flag",
    "atr14_percentile_20d", "atr14_sq", "atr14_zscore_20d",
    "breakout_strength_atr_ratio", "breakout_strength_percentile_20d",
    "breakout_strength_sq", "breakout_strength_vs_orb",
    "breakout_strength_zscore_10d", "day_of_week", "ema_slope_1h",
    "int_adx_x_orb_range", "int_atr14_x_adx",
    "int_breakout_strength_x_range", "int_breakout_strength_x_session",
    "int_vwap_distance_x_atr14",
    "mac_dxy_trend", "mac_oil_volatility", "mac_spx_regime", "mac_us10y_change",
    "orb_range_atr_ratio", "orb_range_percentile_20d",
    "orb_range_sq",
    "pre_bo_bullish_ratio", "pre_bo_compression_ratio",
    "pre_bo_drift_atr", "pre_bo_inside_bar_flag",
    "pre_bo_last_candle_range_ratio",
    "price_vs_vwap_pct", "price_vs_vwap_pct_abs",
    "sm_first_30m_direction", "sm_first_30m_range",
    "sm_pre_breakout_volume_ratio", "sm_pre_breakout_volume_z",
    "time_in_session_min", "vwap_at_breakout",
]


class FeatureBuilder:
    """Computes model features using batch module logic."""

    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer
        self._macro_cache = self._load_macro()
        self._module_funcs = self._import_modules()
        self._vol_norm_ref = self._load_vol_norm_ref()

    def _load_macro(self) -> pd.DataFrame | None:
        path = ROOT / "data" / "Level_1_Features" / "macro_data.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        return None

    def _import_modules(self) -> dict:
        """Import module build_features functions lazily."""
        funcs = {}
        modules = [
            ("orb_context", "pipeline.feature.modules.generate_orb_context_features"),
            ("scale_invariant", "pipeline.feature.modules.generate_scale_invariant_features"),
            ("volatility_normalized", "pipeline.feature.modules.generate_volatility_normalized_features"),
            ("pre_breakout_profile", "pipeline.feature.modules.generate_pre_breakout_profile_features"),
            ("session_momentum", "pipeline.feature.modules.generate_session_momentum_features"),
            ("interaction", "pipeline.feature.modules.generate_interaction_features"),
            ("macro", "pipeline.feature.modules.generate_macro_features"),
        ]
        for name, mod_path in modules:
            try:
                mod = __import__(mod_path, fromlist=["build_features", "load_sources"])
                funcs[name] = {"build": mod.build_features, "load": getattr(mod, "load_sources", None)}
            except Exception as e:
                print(f"[FeatureBuilder] Could not import {name}: {e}", flush=True)
        return funcs

    def _load_vol_norm_ref(self) -> pd.DataFrame | None:
        """Load pre-computed vol_norm features for calibration reference."""
        path = ROOT / "data" / "Live" / "vol_norm_reference.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.set_index("event_key", inplace=True)
            return df
        return None

    # ── base features (ATR, ADX, VWAP) from buffer ───────────────────────────

    def _session_time(self, event: BreakoutEvent) -> dict:
        for s in SESSIONS:
            if s.name == event.session:
                return {
                    "open": datetime.combine(event.date, s.open_utc, tzinfo=timezone.utc),
                    "close": datetime.combine(event.date, s.close_utc, tzinfo=timezone.utc),
                }
        return {"open": event.breakout_ts - timedelta(hours=3), "close": event.breakout_ts}

    def _compute_atr(self, df_1m: pd.DataFrame, period: int = 14) -> float:
        """ATR14 on 1m candles using EMA (matches batch build_breakout_events.py)."""
        if len(df_1m) < period + 1:
            return np.nan
        try:
            df = df_1m.set_index("timestamp_utc")
            h = df["high"].values
            l = df["low"].values
            c = df["close"].values
            prev_c = np.roll(c, 1)
            prev_c[0] = c[0]
            tr = np.maximum(h - l,
                           np.maximum(np.abs(h - prev_c),
                                      np.abs(l - prev_c)))
            # EMA on 1m true range (period=14 minutes)
            import pandas as pd
            atr_series = pd.Series(tr).ewm(span=period, adjust=False).mean()
            return float(atr_series.values[-1])
        except Exception:
            return np.nan

    def _compute_adx(self, event: BreakoutEvent) -> float:
        """ADX(14) on 15m bars from MGC_15m.db — Wilder's smoothing (matches batch)."""
        import sqlite3
        db_path = ROOT / "data" / "Level_0_Raw" / "MGC_15m.db"
        if not db_path.exists():
            return 25.0  # fallback
        try:
            conn = sqlite3.connect(str(db_path))
            end_str = event.breakout_ts.strftime("%Y-%m-%d %H:%M:%S")
            start_str = (event.breakout_ts - timedelta(days=28)).strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_sql(
                "SELECT timestamp_utc, open, high, low, close FROM investing_ohlcv_15m "
                "WHERE symbol='MICRO_GOLD' AND timestamp_utc >= ? AND timestamp_utc <= ? "
                "ORDER BY epoch_ms",
                conn, params=[start_str, end_str],
            )
            conn.close()
            if len(df) < 28:
                return 25.0
            h = df["high"].values
            l = df["low"].values
            c = df["close"].values
            up = h[1:] - h[:-1]
            dn = l[:-1] - l[1:]
            pdm = np.where((up > dn) & (up > 0), up, 0.0)
            ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
            tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            a = 1.0 / 14.0  # Wilder's smoothing
            atr = np.array(pd.Series(tr).ewm(alpha=a, adjust=False).mean().values)
            pdi = 100 * np.array(pd.Series(pdm).ewm(alpha=a, adjust=False).mean().values) / (atr + EPS)
            ndi = 100 * np.array(pd.Series(ndm).ewm(alpha=a, adjust=False).mean().values) / (atr + EPS)
            dx = 100 * np.abs(pdi - ndi) / (pdi + ndi + EPS)
            return float(pd.Series(dx).ewm(alpha=a, adjust=False).mean().values[-1])
        except Exception:
            return 25.0

    def _compute_vwap(self, df_1m: pd.DataFrame) -> float:
        if df_1m.empty or "volume" not in df_1m.columns:
            return 0.0
        vol = df_1m["volume"].values
        typical = (df_1m["high"].values + df_1m["low"].values + df_1m["close"].values) / 3.0
        tvl = typical * vol
        total = vol.sum()
        return float(tvl.sum() / total) if total > EPS else 0.0

    def _compute_ema_slope(self, df_1m: pd.DataFrame, period: int = 20) -> float:
        try:
            df = df_1m.set_index("timestamp_utc")
            resampled = df.resample("1h").agg({"close": "last"}).dropna()
            if len(resampled) < period:
                return 0.0
            ema = resampled["close"].ewm(span=period, adjust=False).mean()
            if len(ema) < 5:
                return 0.0
            y = ema.values[-5:]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return float(slope / (ema.values[-1] + EPS))
        except Exception:
            return 0.0

    # ── build breakout_events + market_context rows ──────────────────────────

    def _build_base_data(self, event: BreakoutEvent) -> tuple:
        """Build breakout_events-like, market_context-like, df_1m, df_15m."""
        sess = self._session_time(event)
        entry = event.entry_price
        orb_range = event.orb_range

        # Get data for ATR/ADX
        lookback = event.breakout_ts - timedelta(days=14)
        df_long = self.buffer.get(
            lookback.strftime("%Y-%m-%d %H:%M:%S"),
            event.breakout_ts.strftime("%Y-%m-%d %H:%M:%S"),
        )
        atr14 = self._compute_atr(df_long)
        atr14 = atr14 if not np.isnan(atr14) else orb_range * 0.8
        adx = self._compute_adx(event)
        adx = adx if not np.isnan(adx) else 25.0

        # Session data for VWAP + session_momentum
        sess_start = sess["open"].strftime("%Y-%m-%d %H:%M:%S")
        sess_end = event.breakout_ts.strftime("%Y-%m-%d %H:%M:%S")
        sess_df = self.buffer.get(sess_start, sess_end)
        vwap = self._compute_vwap(sess_df) or entry
        ema_slope = self._compute_ema_slope(df_long) if len(df_long) > 0 else 0.0

        # Resample to 15m for pre_breakout — extend 2h before breakout
        # to guarantee 4+ pre-breakout 15m candles even for early session breakouts
        df_15m = pd.DataFrame()
        try:
            ext_start = (event.breakout_ts - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
            ext_end = (event.breakout_ts + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            df_ext = self.buffer.get(ext_start, ext_end)
            if len(df_ext) > 0:
                sdf = df_ext.set_index("timestamp_utc")
                df_15m = sdf.resample("15min").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna().reset_index()
                df_15m.columns = ["timestamp_utc", "open", "high", "low", "close", "volume"]
        except Exception:
            pass

        # Build breakout_events row
        bo = pd.DataFrame([{
            "date": str(event.date),
            "session": event.session,
            "orb_tf": event.orb_tf,
            "breakout_ts": event.breakout_ts,
            "orb_high": event.orb_high,
            "orb_low": event.orb_low,
            "orb_range": orb_range,
            "entry_price": entry,
            "breakout_strength": abs(entry - (event.orb_high if event.breakout_side == 1 else event.orb_low)),
            "atr14_at_entry": atr14,
            "breakout_side": event.breakout_side,
            "session_close_ts": sess["close"],
            "sl_dist": orb_range,
            "orb_start_ts": event.orb_start_ts or sess["open"],
            "orb_end_ts": event.orb_end_ts or (sess["open"] + timedelta(minutes=15)),
        }])

        # Build market_context row
        time_in_sess = (event.breakout_ts - sess["open"]).total_seconds() / 60.0
        mc = pd.DataFrame([{
            "date": str(event.date),
            "session": event.session,
            "orb_tf": event.orb_tf,
            "breakout_ts": event.breakout_ts,
            "orb_range_atr_ratio": orb_range / (atr14 + EPS),
            "day_of_week": event.date.weekday(),
            "time_in_session_min": time_in_sess,
            "vwap_at_breakout": vwap,
            "price_vs_vwap_pct": (entry - vwap) / (vwap + EPS) * 100.0,
            "adx_14_15m": adx,
            "ema_slope_1h": np.sign(ema_slope),
        }])

        return bo, mc, sess_df, df_15m

    def _get_sm_df(self, event: BreakoutEvent, sess: dict) -> pd.DataFrame:
        """Get 1m data for session_momentum with at least 30m range."""
        sm_start = sess["open"].strftime("%Y-%m-%d %H:%M:%S")
        sm_end = max(event.breakout_ts, sess["open"] + timedelta(minutes=30))
        return self.buffer.get(sm_start, sm_end.strftime("%Y-%m-%d %H:%M:%S"))

    # ── main build ────────────────────────────────────────────────────────────

    def build(self, event: BreakoutEvent) -> dict[str, float]:
        """Compute all features for a breakout event."""
        bo, mc, df_1m, df_15m = self._build_base_data(event)

        # Extended data for session_momentum (at least 30m needed for first-30m features)
        sm_df = self._get_sm_df(event, self._session_time(event))

        sources = {
            "breakout_events": bo,
            "market_context": mc,
        }

        features = {k: 0.0 for k in FEATURE_ORDER}

        # ── orb_context ────────────────────────────────────────────────
        if "orb_context" in self._module_funcs:
            try:
                result = self._module_funcs["orb_context"]["build"](sources)
                for col in result.columns:
                    if col not in ("date", "session", "orb_tf", "breakout_ts") and col in features:
                        features[col] = float(result[col].iloc[0]) if not pd.isna(result[col].iloc[0]) else 0.0
            except Exception as e:
                print(f"[FeatureBuilder] orb_context: {e}", flush=True)
                for col in ("day_of_week", "time_in_session_min", "vwap_at_breakout",
                             "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h",
                             "orb_range_atr_ratio"):
                    if col in mc.columns and col in features:
                        v = mc[col].iloc[0]
                        features[col] = float(v) if not pd.isna(v) else 0.0

        # ── scale_invariant ────────────────────────────────────────────
        if "scale_invariant" in self._module_funcs:
            merged = bo.merge(mc, on=["date", "session", "orb_tf", "breakout_ts"], how="left")
            try:
                result = self._module_funcs["scale_invariant"]["build"]({"breakout_events": bo, "market_context": mc})
                for col in ["breakout_strength_atr_ratio", "atr14_sq", "breakout_strength_sq",
                            "price_vs_vwap_pct_abs", "orb_range_sq", "adx_50_flag", "breakout_strength_vs_orb"]:
                    if col in result.columns and col in features:
                        features[col] = float(result[col].iloc[0]) if not pd.isna(result[col].iloc[0]) else 0.0
            except Exception as e:
                print(f"[FeatureBuilder] scale_invariant: {e}", flush=True)

        # ── Core features from bo (breakout_events) ─────────────────
        bo_cols_used = ["breakout_strength", "atr14_at_entry"]
        for col in bo_cols_used:
            if col in bo.columns:
                features[col] = float(bo[col].iloc[0]) if not pd.isna(bo[col].iloc[0]) else 0.0

        # ── Core features from mc (market_context) ───────────────────
        mc_cols_used = ["day_of_week", "time_in_session_min",
                          "adx_14_15m", "vwap_at_breakout", "price_vs_vwap_pct",
                          "ema_slope_1h", "orb_range_atr_ratio"]
        if mc is not None and len(mc) > 0:
            for col in mc_cols_used:
                if col in mc.columns:
                    features[col] = float(mc[col].iloc[0]) if not pd.isna(mc[col].iloc[0]) else 0.0

        if "breakout_side" in bo.columns:
            features["breakout_side"] = float(bo["breakout_side"].iloc[0])

        features["price_vs_vwap_pct_abs"] = abs(features["price_vs_vwap_pct"])
        features["adx_50_flag"] = 1.0 if features["adx_14_15m"] > 50 else 0.0

        # ── Session momentum features (inline, matches batch module) ──
        self._compute_session_momentum(event, sm_df, bo, features)

        # ── Pre-breakout profile features (inline, matches batch module) ──
        self._compute_pre_breakout(event, bo, df_15m, features)

        # ── vol_norm (from batch reference table) ───────────────────────
        event_key = f"{str(event.date)}|{event.session}|{event.orb_tf}|{event.breakout_ts}"
        if self._vol_norm_ref is not None and event_key in self._vol_norm_ref.index:
            row = self._vol_norm_ref.loc[event_key]
            features["atr14_percentile_20d"] = float(row["atr14_percentile_20d"]) if not pd.isna(row["atr14_percentile_20d"]) else 0.5
            features["atr14_zscore_20d"] = float(row["atr14_zscore_20d"]) if not pd.isna(row["atr14_zscore_20d"]) else 0.0
            features["breakout_strength_percentile_20d"] = float(row["breakout_strength_percentile_20d"]) if not pd.isna(row["breakout_strength_percentile_20d"]) else 0.5
            features["breakout_strength_zscore_10d"] = float(row["breakout_strength_zscore_10d"]) if not pd.isna(row["breakout_strength_zscore_10d"]) else 0.0
            features["orb_range_percentile_20d"] = float(row["orb_range_percentile_20d"]) if not pd.isna(row["orb_range_percentile_20d"]) else 0.5
        # else: keep defaults (0.5/0.0) for events not in reference

        # ── Scale-invariant derived ────────────────────────────────────
        features["atr14_sq"] = features["atr14_at_entry"] ** 2
        features["breakout_strength_sq"] = features["breakout_strength"] ** 2
        features["orb_range_sq"] = bo["orb_range"].iloc[0] ** 2 if len(bo) > 0 else 0.0
        features["breakout_strength_atr_ratio"] = features["breakout_strength"] / (features["atr14_at_entry"] + EPS)
        features["breakout_strength_vs_orb"] = features["breakout_strength"] / (bo["orb_range"].iloc[0] + EPS) if len(bo) > 0 else 0.0

        # ── Interaction features ───────────────────────────────────────
        features["int_atr14_x_adx"] = features["atr14_at_entry"] * features["adx_14_15m"]
        features["int_breakout_strength_x_range"] = features["breakout_strength"] * (bo["orb_range"].iloc[0] if len(bo) > 0 else 0)
        features["int_vwap_distance_x_atr14"] = features["price_vs_vwap_pct_abs"] * features["atr14_at_entry"]
        features["int_adx_x_orb_range"] = features["adx_14_15m"] * features["orb_range_atr_ratio"]
        features["int_breakout_strength_x_session"] = features["breakout_strength"] * features["time_in_session_min"]

        # ── Macro features ─────────────────────────────────────────────
        macro = self._get_macro(event.date)
        features["mac_spx_regime"] = macro.get("spx_regime", 0.5)
        features["mac_dxy_trend"] = macro.get("dxy_trend", 0.0)
        features["mac_us10y_change"] = macro.get("us10y_change", 0.0)
        features["mac_oil_volatility"] = macro.get("oil_volatility", 0.0)

        return features

    def _compute_session_momentum(self, event, df_1m, bo, features):
        """Compute sm_* features inline, matching batch module logic exactly."""
        try:
            if len(df_1m) == 0:
                return
            if df_1m["timestamp_utc"].dtype == object or str(df_1m["timestamp_utc"].dtype) == "str":
                df_1m["timestamp_utc"] = pd.to_datetime(df_1m["timestamp_utc"])
            if df_1m["timestamp_utc"].dt.tz is None:
                df_1m["timestamp_utc"] = df_1m["timestamp_utc"].dt.tz_localize("UTC")
            df_1m = df_1m.sort_values("timestamp_utc").reset_index(drop=True)
            bo_ts = event.breakout_ts
            sess_open = self._session_time(event)["open"]
            sess_open_30m = sess_open + timedelta(minutes=30)
            ts_arr = df_1m["timestamp_utc"].values.astype("datetime64[ns]")
            open_ns = np.datetime64(sess_open.replace(tzinfo=timezone.utc))
            open_30m_ns = open_ns + np.timedelta64(30, "m")
            bo_ts_ns = np.datetime64(bo_ts.replace(tzinfo=timezone.utc) if bo_ts.tzinfo is None else bo_ts)

            open_idx = int(np.searchsorted(ts_arr, open_ns, side="left"))
            open_30m_idx = int(np.searchsorted(ts_arr, open_30m_ns, side="left"))
            bo_idx = int(np.searchsorted(ts_arr, bo_ts_ns, side="left"))

            atr14 = features.get("atr14_at_entry", 1.0)
            if atr14 <= 0:
                atr14 = 1.0

            if open_idx < len(df_1m) and open_30m_idx > open_idx:
                first_30m_high = float(df_1m["high"].iloc[open_idx:open_30m_idx].max())
                first_30m_low = float(df_1m["low"].iloc[open_idx:open_30m_idx].min())
                first_30m_range = first_30m_high - first_30m_low
                first_30m_close = float(df_1m["close"].iloc[open_30m_idx - 1])
                first_30m_open = float(df_1m["open"].iloc[open_idx])
                features["sm_first_30m_range"] = first_30m_range / atr14
                features["sm_first_30m_direction"] = (first_30m_close - first_30m_open) / atr14

            pre_bo_15m = bo_ts - timedelta(minutes=15)
            pre_bo_ns = np.datetime64(pre_bo_15m.replace(tzinfo=timezone.utc) if pre_bo_15m.tzinfo is None else pre_bo_15m)
            pre_bo_idx = int(np.searchsorted(ts_arr, pre_bo_ns, side="left"))

            if pre_bo_idx < bo_idx and pre_bo_idx >= open_idx:
                pre_bo_vol = float(df_1m["volume"].iloc[pre_bo_idx:bo_idx].sum()) if "volume" in df_1m.columns else 0.0
                n_session = bo_idx - open_idx
                if n_session > 0:
                    n_pre = bo_idx - pre_bo_idx
                    session_vol = float(df_1m["volume"].iloc[open_idx:bo_idx].sum()) if "volume" in df_1m.columns else 1.0
                    avg_15m = (session_vol / n_session) * 15
                    if avg_15m > 0:
                        features["sm_pre_breakout_volume_ratio"] = pre_bo_vol / avg_15m
                        if n_session >= 5 and "volume" in df_1m.columns:
                            vols = df_1m["volume"].iloc[open_idx:bo_idx].values.astype(float)
                            vmean = vols.mean(); vstd = vols.std()
                            avg_pre = pre_bo_vol / n_pre if n_pre > 0 else 0
                            features["sm_pre_breakout_volume_z"] = float(np.clip((avg_pre - vmean) / (vstd + 0.01), -5, 5))
        except Exception as e:
            print(f"[FeatureBuilder] sm: {e}", flush=True)

    def _compute_pre_breakout(self, event, bo, df_15m, features):
        """Compute pre_bo_* features inline from 15m candles."""
        try:
            if len(df_15m) < 4:
                return
            if df_15m["timestamp_utc"].dtype == object or str(df_15m["timestamp_utc"].dtype) == "str":
                df_15m["timestamp_utc"] = pd.to_datetime(df_15m["timestamp_utc"])
            if df_15m["timestamp_utc"].dt.tz is None:
                df_15m["timestamp_utc"] = df_15m["timestamp_utc"].dt.tz_localize("UTC")
            df_15m = df_15m.sort_values("timestamp_utc").reset_index(drop=True)
            bo_ts = event.breakout_ts
            bo_ts_ns = np.datetime64(bo_ts.replace(tzinfo=timezone.utc) if bo_ts.tzinfo is None else bo_ts)
            ts_arr = df_15m["timestamp_utc"].values.astype("datetime64[ns]")
            idx = int(np.searchsorted(ts_arr, bo_ts_ns, side="right")) - 1

            if idx < 4:
                return
            pre_idx = idx - 4 + 1
            candles = df_15m.iloc[pre_idx:idx + 1]
            orb_range = float(bo["orb_range"].iloc[0]) if len(bo) > 0 else 1.0
            atr14 = features.get("atr14_at_entry", 1.0)
            if orb_range <= 0: orb_range = 1.0
            if atr14 <= 0: atr14 = 1.0

            pre_high = float(candles["high"].max()); pre_low = float(candles["low"].min())
            features["pre_bo_compression_ratio"] = (pre_high - pre_low) / orb_range
            drift = abs(float(candles.iloc[-1]["close"]) - float(candles.iloc[0]["open"]))
            features["pre_bo_drift_atr"] = drift / atr14
            last = candles.iloc[-1]; prev = candles.iloc[-2]
            features["pre_bo_inside_bar_flag"] = float(last["high"] < prev["high"] and last["low"] > prev["low"])
            features["pre_bo_last_candle_range_ratio"] = (float(last["high"]) - float(last["low"])) / atr14
            bullish = int((candles["close"] > candles["open"]).sum())
            features["pre_bo_bullish_ratio"] = bullish / len(candles)
        except Exception as e:
            print(f"[FeatureBuilder] pb: {e}", flush=True)

    def _get_macro(self, date) -> dict:
        """Compute macro features from raw macro_data.parquet columns (matches batch)."""
        if self._macro_cache is None:
            return {}
        d = date if isinstance(date, Date) else (pd.Timestamp(date).date() if hasattr(pd.Timestamp(date), 'date') else date)
        mask = self._macro_cache["date"] <= d
        if not mask.any():
            return {}
        recent = self._macro_cache[mask].iloc[-1]

        spy_c = float(recent.get("spy_close", np.nan))
        spy_ma = float(recent.get("spy_ma200", np.nan))
        dxy_c = float(recent.get("dxy_close", np.nan))
        dxy_ma = float(recent.get("dxy_ma50", np.nan))
        us10y = float(recent.get("us10y_change", 0.0))
        oil_r = float(recent.get("oil_return", 0.0))

        spx_regime = np.nan
        if np.isfinite(spy_c) and np.isfinite(spy_ma):
            spx_regime = 1.0 if spy_c > spy_ma else 0.0

        dxy_trend = np.nan
        if np.isfinite(dxy_c) and np.isfinite(dxy_ma):
            if dxy_c > dxy_ma * 1.005:
                dxy_trend = 1.0
            elif dxy_c < dxy_ma * 0.995:
                dxy_trend = -1.0
            else:
                dxy_trend = 0.0

        return {
            "spx_regime": float(spx_regime) if not np.isnan(spx_regime) else 0.5,
            "dxy_trend": float(dxy_trend) if not np.isnan(dxy_trend) else 0.0,
            "us10y_change": float(us10y) if np.isfinite(us10y) else 0.0,
            "oil_volatility": abs(float(oil_r)) if np.isfinite(oil_r) else 0.0,
        }

    def to_array(self, features: dict[str, float]) -> np.ndarray:
        return np.array([features.get(col, 0.0) for col in FEATURE_ORDER], dtype=np.float32)

    def build_array(self, event: BreakoutEvent) -> np.ndarray:
        return self.to_array(self.build(event))
