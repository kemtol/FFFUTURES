"""
Level 2: Label setiap breakout event — apakah TP kena sebelum SL dalam X menit.

Fast approach: preconvert 1m data ke numpy arrays, pakai searchsorted untuk
O(log N) index lookup per breakout. Eliminasi pandas overhead di inner loop.

Output: data/Level_2_Datamart/training_datamart_orb.parquet
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT  = Path(__file__).parent.parent.parent
DB_1M = ROOT / "data/Level_0_Raw/MGC_1m.db"
BO_IN = ROOT / "data/Level_1_Features/breakout_events.parquet"
OUT   = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"

HORIZONS_MIN = [60, 120, 180, 240]


def load_1m_numpy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epoch_ms_array, high_array, low_array) sebagai numpy arrays sorted."""
    conn = sqlite3.connect(DB_1M)
    df = pd.read_sql(
        "SELECT epoch_ms, high, low FROM investing_ohlcv_1m "
        "WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    return df["epoch_ms"].values, df["high"].values, df["low"].values


def first_hit(highs: np.ndarray, lows: np.ndarray, tp: float, sl: float, side: int) -> int:
    """
    Cari apakah TP kena sebelum SL dalam window candles.
    side=1: long (TP=high>=tp, SL=low<=sl)
    side=-1: short (TP=low<=tp, SL=high>=sl)
    """
    if len(highs) == 0:
        return 0

    if side == 1:
        tp_hits = highs >= tp
        sl_hits = lows  <= sl
    else:
        tp_hits = lows  <= tp
        sl_hits = highs >= sl

    tp_idx = np.argmax(tp_hits) if tp_hits.any() else len(tp_hits)
    sl_idx = np.argmax(sl_hits) if sl_hits.any() else len(sl_hits)

    if not tp_hits.any():
        return 0
    return 1 if tp_idx <= sl_idx else 0


def label_breakouts(bo: pd.DataFrame, epochs: np.ndarray,
                    highs: np.ndarray, lows: np.ndarray) -> pd.DataFrame:
    MS_PER_MIN = 60_000
    records = []
    total = len(bo)

    for i, (_, row) in enumerate(bo.iterrows()):
        if i % 5000 == 0:
            print(f"  {i}/{total}...")

        entry_ep   = int(row["breakout_ts"].timestamp() * 1000)
        close_ep   = int(row["session_close_ts"].timestamp() * 1000)
        close60_ep = close_ep - 60 * MS_PER_MIN

        # Index pertama setelah entry
        i_start = np.searchsorted(epochs, entry_ep, side="right")

        for side_label, tp2, tp4, sl in [
            ("cont", row["tp2r_price_cont"], row["tp4r_price_cont"], row["sl_price_cont"]),
            ("rev",  row["tp2r_price_rev"],  row["tp4r_price_rev"],  row["sl_price_rev"]),
        ]:
            side_int = row["breakout_side"] if side_label == "cont" else -row["breakout_side"]

            record = {
                "date":              row["date"],
                "session":           row["session"],
                "orb_tf":            row["orb_tf"],
                "side":              side_label,
                "breakout_side":     row["breakout_side"],
                "breakout_ts":       row["breakout_ts"],
                "entry_price":       row["entry_price"],
                "orb_range":         row["orb_range"],
                "atr14_at_entry":    row["atr14_at_entry"],
                "sl_dist":           row["sl_dist"],
                "breakout_strength": row["breakout_strength"],
                "session_close_ts":  row["session_close_ts"],
            }

            for h in HORIZONS_MIN:
                end_ep = entry_ep + h * MS_PER_MIN
                i_end  = np.searchsorted(epochs, end_ep, side="right")
                h_arr  = highs[i_start:i_end]
                l_arr  = lows[i_start:i_end]
                record[f"y_1r2_{h}m"] = first_hit(h_arr, l_arr, tp2, sl, side_int)
                record[f"y_1r4_{h}m"] = first_hit(h_arr, l_arr, tp4, sl, side_int)

            # close - 60m
            if close60_ep > entry_ep:
                i_c = np.searchsorted(epochs, close60_ep, side="right")
                h_arr = highs[i_start:i_c]
                l_arr = lows[i_start:i_c]
                record["y_1r2_close60m"] = first_hit(h_arr, l_arr, tp2, sl, side_int)
                record["y_1r4_close60m"] = first_hit(h_arr, l_arr, tp4, sl, side_int)
            else:
                record["y_1r2_close60m"] = np.nan
                record["y_1r4_close60m"] = np.nan

            records.append(record)

    return pd.DataFrame(records)


def main(full_rebuild: bool = False) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not BO_IN.exists():
        raise FileNotFoundError(f"Run build_breakout_events.py first: {BO_IN}")

    bo = pd.read_parquet(BO_IN)

    existing_dates: set = set()
    existing_df = pd.DataFrame()
    if OUT.exists() and not full_rebuild:
        existing_df = pd.read_parquet(OUT)
        existing_dates = set(existing_df["date"].unique())
        print(f"Existing: {len(existing_df):,} rows, {len(existing_dates)} dates")

    new_bo = bo[~bo["date"].isin(existing_dates)] if existing_dates else bo
    if new_bo.empty:
        print("Already up to date.")
        return

    print(f"Labeling {len(new_bo):,} breakouts ({new_bo['date'].nunique()} dates)...")
    print("Loading 1m data as numpy arrays...")
    epochs, highs, lows = load_1m_numpy()

    print("Computing labels...")
    import time
    t0 = time.time()
    new_df = label_breakouts(new_bo, epochs, highs, lows)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(new_bo)*1000:.1f}ms per breakout)")

    final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    final_df = final_df.sort_values("breakout_ts").reset_index(drop=True)
    final_df.to_parquet(OUT, index=False)

    print(f"Saved: {len(final_df):,} total rows → {OUT}")


if __name__ == "__main__":
    import sys
    main(full_rebuild="--rebuild" in sys.argv)
