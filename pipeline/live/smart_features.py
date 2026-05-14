
import numpy as np
import pandas as pd
from datetime import datetime, timezone

class SMARTFeatureBuilder:
    """Computes features for SMART_1 (Meta-v7/v8) in real-time from buffer."""
    
    def __init__(self, buffer):
        self.buffer = buffer

    def build_smart_features(self, symbol="MGC", now=None, st_val=None, entry_atr=None):
        """Build the feature vector for current market state."""
        # Load 120 bars of 5m data for indicator stability
        now = now or datetime.now(timezone.utc)
        start = (now - pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")
        
        df_1m = self.buffer.get(start, end)
        if df_1m.empty: return None
        
        df_1m["timestamp_utc"] = pd.to_datetime(df_1m["timestamp_utc"], utc=True)
        df_5m = df_1m.set_index("timestamp_utc").resample("5min", label="right", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(df_5m) < 50: return None
        
        # 1. Base Indicators
        c = df_5m["close"].values
        h = df_5m["high"].values
        l = df_5m["low"].values
        o = df_5m["open"].values
        
        # entry_adx (12 period)
        entry_adx = self._compute_adx(h, l, c, 12)
        # cci_abs (12 period)
        cci_val = self._compute_cci(h, l, c, 12)
        cci_abs = abs(cci_val)
        
        # 2. Regime Features (from 15m resample)
        df_15m = df_1m.set_index("timestamp_utc").resample("15min", label="right", closed="left").agg({
            "close": "last", "high": "max", "low": "min"
        }).dropna()
        
        efficiency_ratio = 0.5
        volatility_zscore = 0.0
        if len(df_15m) >= 20:
            c15 = df_15m["close"].values
            change = abs(c15[-1] - c15[-20])
            path = np.sum(np.abs(np.diff(c15[-20:])))
            efficiency_ratio = change / (path + 1e-9)
            
            raw_vol = (df_15m["high"] - df_15m["low"]) / df_15m["close"]
            if len(raw_vol) >= 100:
                mean_vol = raw_vol.rolling(100).mean().iloc[-1]
                std_vol = raw_vol.rolling(100).std().iloc[-1]
                volatility_zscore = (raw_vol.iloc[-1] - mean_vol) / (std_vol + 1e-9)

        # 3. Structural Features
        # st_gap_ratio
        st_gap_ratio = 0.0
        if st_val and entry_atr:
            st_gap_ratio = abs(c[-1] - st_val) / (entry_atr + 1e-9)
        
        # wick_ratio
        wick = max(0, h[-1] - max(o[-1], c[-1])) + max(0, min(o[-1], c[-1]) - l[-1])
        body = abs(c[-1] - o[-1])
        wick_ratio = wick / (body + body*0.1 + 1e-9) # slight offset for stability
        
        # candle_body_atr
        atr = self._compute_atr(h, l, c, 14)
        candle_body_atr = body / (atr + 1e-9)
        
        # vol_ratio
        vol = df_5m["volume"].values
        vol_ratio = vol[-1] / (np.mean(vol[-20:]) + 1e-9)
        
        # rsi_5
        rsi_5 = self._compute_rsi(c, 5)

        # st_slope (Diff 5)
        # Strategy needs to pass ST history or we recompute locally
        # For simplicity, we'll recompute ST here to get the slope
        st_vals, _ = self._compute_st(h, l, c, 4.0, 12)
        st_slope = st_vals[-1] - st_vals[-6] if len(st_vals) > 6 else 0.0
        
        # session_cluster
        hour = now.astimezone(timezone.utc).hour
        if 1 <= hour < 7: session_cluster = 0  # Tokyo
        elif 7 <= hour < 13: session_cluster = 1 # London
        else: session_cluster = 2 # US
        
        return {
            "entry_adx": entry_adx,
            "cci_abs": cci_abs,
            "efficiency_ratio": np.clip(efficiency_ratio, 0, 1),
            "volatility_zscore": np.clip(volatility_zscore, -3, 3),
            "wick_ratio": np.clip(wick_ratio, 0, 5),
            "candle_body_atr": np.clip(candle_body_atr, 0, 5),
            "vol_ratio": np.clip(vol_ratio, 0, 10),
            "rsi_5": rsi_5,
            "st_slope": st_slope,
            "st_gap_ratio": st_gap_ratio,
            "session_cluster": session_cluster
        }

    def _compute_st(self, h, l, c, factor, period):
        atr_val = pd.Series(self._compute_atr_array(h, l, c, period))
        hl2 = (h + l) / 2.0
        upper = hl2 + factor * atr_val
        lower = hl2 - factor * atr_val
        st = np.full(len(c), np.nan)
        direction = np.zeros(len(c), dtype=int)
        for i in range(len(c)):
            if i == 0:
                direction[i] = 1; st[i] = upper[i]
                continue
            if st[i-1] == upper[i-1]:
                direction[i] = 1 if c[i] <= upper[i-1] else -1
            else:
                direction[i] = -1 if c[i] >= lower[i-1] else 1
            st[i] = upper[i] if direction[i] == 1 else lower[i]
        return st, direction

    def _compute_atr_array(self, h, l, c, period):
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        return pd.Series(tr).ewm(span=period, adjust=False).mean().values

    def _compute_adx(self, h, l, c, period=14):
        if len(c) < period * 2: return 25.0
        up = h[1:] - h[:-1]
        dn = l[:-1] - l[1:]
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        a = 1.0 / period
        atr = pd.Series(tr).ewm(alpha=a, adjust=False).mean().values
        pdi = 100 * pd.Series(pdm).ewm(alpha=a, adjust=False).mean().values / (atr + 1e-9)
        ndi = 100 * pd.Series(ndm).ewm(alpha=a, adjust=False).mean().values / (atr + 1e-9)
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi + 1e-9)
        return float(pd.Series(dx).ewm(alpha=a, adjust=False).mean().values[-1])

    def _compute_cci(self, h, l, c, period=12):
        tp = (h + l + c) / 3.0
        ma = pd.Series(tp).rolling(period).mean().values
        md = pd.Series(tp).rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x)))).values
        return (tp[-1] - ma[-1]) / (0.015 * md[-1] + 1e-9)

    def _compute_atr(self, h, l, c, period=14):
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        return pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]

    def _compute_rsi(self, c, period=5):
        delta = np.diff(c)
        up = delta.copy(); up[up < 0] = 0
        dn = delta.copy(); dn[dn > 0] = 0; dn = abs(dn)
        ma_up = pd.Series(up).rolling(period).mean().iloc[-1]
        ma_dn = pd.Series(dn).rolling(period).mean().iloc[-1]
        if ma_dn == 0: return 100
        rs = ma_up / ma_dn
        return 100 - (100 / (1 + rs))
