#!/usr/bin/env python3
"""
Live buffer data health check.

Phase 1: validates `data/Live/topstepx_buffer.db.ohlcv_1m` — the only data
source V8 router consumes at inference time. Read-only; no halt action
(per design decision D2 in plans/super_structure_data_health.md).

Six checks per run:
  1. Freshness — latest bar age vs threshold (10 min during market hours)
  2. Quantity — row count for last 24h matches expected
  3. Continuity — no gaps between consecutive 1m bars
  4. OHLC sanity — H >= max(O,C), L <= min(O,C), C > 0, V >= 0
  5. Price plausibility — close inside [1000, 10000] for MGC
  6. Duplicate timestamps — no dup timestamps in window

Output:
  - JSON: data/Live/health/data_health_{date}_{tz}.json
  - Markdown: same path + .md
  - Telegram push (per --telegram flag)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
BUFFER_DB = ROOT / "data" / "Live" / "topstepx_buffer.db"
OUT_DIR = ROOT / "data" / "Live" / "health"
TELEGRAM_ENV_PATH = ROOT / "data" / "Live" / "telegram.env"

# Thresholds — see plans/super_structure_data_health.md
FRESHNESS_THRESHOLD_S = 600          # 10 min
GAP_WARN_MIN = 5
GAP_CRITICAL_MIN = 30
PRICE_MIN = 1000.0
PRICE_MAX = 10000.0
QUANTITY_WARN_RATIO = 0.90           # of expected 1440 bars in 24h
WINDOW_HOURS = 24
SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"
TZ = "Asia/Jakarta"

# CME futures maintenance windows (US Central Time):
#   Daily: 16:00-17:00 CT  (Mon-Thu)
#   Weekly: Friday 16:00 CT → Sunday 17:00 CT
# We treat gaps that fall fully inside these windows as expected, not outages.
CME_DAILY_HALT_START_CT = (16, 0)
CME_DAILY_HALT_END_CT = (17, 0)
CME_WEEKEND_START_DAY = 4            # Friday
CME_WEEKEND_END_DAY = 6              # Sunday


# ── data loading ──────────────────────────────────────────────────────────


SNAPSHOT_DB = Path("/tmp/super_structure_health_snapshot.db")


def _snapshot_buffer() -> Path:
    """Copy the live buffer via sqlite3.backup() so we don't fight the live
    writer for a lock. Same pattern as `pipeline/live/rebuild_super_structure_ui.py`.
    """
    if SNAPSHOT_DB.exists():
        SNAPSHOT_DB.unlink()
    src_uri = f"file:{BUFFER_DB}?mode=ro"
    with sqlite3.connect(src_uri, uri=True, timeout=30) as src:
        with sqlite3.connect(str(SNAPSHOT_DB), timeout=30) as dst:
            src.backup(dst)
    return SNAPSHOT_DB


def load_recent_bars(hours: int = WINDOW_HOURS) -> pd.DataFrame:
    """Read from an isolated snapshot to avoid contending with the live writer."""
    if not BUFFER_DB.exists():
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(hours=hours)
    snapshot = _snapshot_buffer()
    con = sqlite3.connect(str(snapshot), timeout=30)
    try:
        df = pd.read_sql(
            "SELECT timestamp_utc, open, high, low, close, volume, epoch_ms "
            "FROM ohlcv_1m WHERE symbol = ? AND timeframe = ? "
            "AND timestamp_utc >= ? ORDER BY epoch_ms",
            con,
            params=[SYMBOL, TIMEFRAME, start.strftime("%Y-%m-%d %H:%M:%S")],
        )
    finally:
        con.close()
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


# ── individual checks ─────────────────────────────────────────────────────


def check_freshness(df: pd.DataFrame, now_utc: datetime) -> dict:
    if df.empty:
        return {"severity": "CRITICAL", "latest_bar_ts": None,
                "age_seconds": None,
                "threshold_seconds": FRESHNESS_THRESHOLD_S,
                "note": "buffer empty in window"}
    latest = df["timestamp_utc"].max()
    age = (now_utc - latest.to_pydatetime()).total_seconds()
    severity = "PASS" if age <= FRESHNESS_THRESHOLD_S else "CRITICAL"
    return {
        "severity": severity,
        "latest_bar_ts": latest.isoformat(),
        "age_seconds": round(age, 1),
        "threshold_seconds": FRESHNESS_THRESHOLD_S,
        "note": "" if severity == "PASS" else f"stale > {FRESHNESS_THRESHOLD_S}s",
    }


def check_quantity(df: pd.DataFrame, hours: int) -> dict:
    expected = hours * 60
    actual = len(df)
    ratio = (actual / expected) if expected else 0.0
    severity = "PASS" if ratio >= QUANTITY_WARN_RATIO else "WARN"
    return {
        "severity": severity,
        "actual_bars": int(actual),
        "expected_bars": int(expected),
        "ratio": round(ratio, 3),
        "threshold_ratio": QUANTITY_WARN_RATIO,
        "note": "" if severity == "PASS"
                else f"only {ratio*100:.1f}% of expected (market halt acceptable)",
    }


def _cme_maintenance_minutes_covered(from_ts: pd.Timestamp,
                                      to_ts: pd.Timestamp) -> float:
    """Return minutes of overlap between [from_ts, to_ts] and any known CME
    maintenance window. Handles daily 16:00-17:00 CT halt + weekend halt
    (Fri 16:00 CT → Sun 17:00 CT). DST handled by tz_convert.
    """
    from_ct = from_ts.tz_convert("America/Chicago")
    to_ct = to_ts.tz_convert("America/Chicago")
    # Walk minute-by-minute when gap < 3h (covers daily halt).
    # For longer gaps (weekend), compute span via day-of-week + hour rules.
    if (to_ts - from_ts) <= pd.Timedelta(hours=3):
        # Day-resolution overlap with each daily halt window.
        total = 0
        cur = from_ct.ceil("min")
        while cur < to_ct:
            if (cur.weekday() < CME_WEEKEND_START_DAY
                and CME_DAILY_HALT_START_CT
                <= (cur.hour, cur.minute)
                < CME_DAILY_HALT_END_CT):
                total += 1
            cur += pd.Timedelta(minutes=1)
        return float(total)
    # Long gap — weekend or multi-day outage. Treat overlap with weekend halt
    # (Fri 16:00 CT → Sun 17:00 CT) as expected.
    weekend_start = from_ct.normalize() - pd.Timedelta(days=from_ct.weekday())
    weekend_start += pd.Timedelta(days=CME_WEEKEND_START_DAY, hours=16)
    weekend_end = weekend_start + pd.Timedelta(days=2, hours=1)
    overlap_start = max(from_ct, weekend_start)
    overlap_end = min(to_ct, weekend_end)
    if overlap_end <= overlap_start:
        return 0.0
    return (overlap_end - overlap_start).total_seconds() / 60.0


def check_continuity(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {"severity": "PASS", "gap_count": 0, "max_gap_min": 0,
                "max_effective_gap_min": 0,
                "samples": [], "note": "not enough rows to evaluate"}
    df_sorted = df.sort_values("epoch_ms").reset_index(drop=True)
    deltas_ms = df_sorted["epoch_ms"].diff().dropna()
    # Gaps > 60s + 5s slack = anything > 65000ms is a "skip"
    gap_threshold_ms = 65_000
    gaps = deltas_ms[deltas_ms > gap_threshold_ms]
    if gaps.empty:
        return {"severity": "PASS", "gap_count": 0, "max_gap_min": 0.0,
                "max_effective_gap_min": 0.0,
                "samples": [], "note": ""}
    samples = []
    max_effective = 0.0
    for idx, ms in gaps.items():
        prev_ts = df_sorted.loc[idx - 1, "timestamp_utc"]
        cur_ts = df_sorted.loc[idx, "timestamp_utc"]
        gap_min = float(ms) / 60_000
        maint_min = _cme_maintenance_minutes_covered(prev_ts, cur_ts)
        effective_min = max(0.0, gap_min - maint_min)
        if effective_min > max_effective:
            max_effective = effective_min
        samples.append({
            "from": prev_ts.isoformat(),
            "to": cur_ts.isoformat(),
            "gap_min": round(gap_min, 2),
            "cme_halt_min": round(maint_min, 2),
            "effective_gap_min": round(effective_min, 2),
        })
    samples.sort(key=lambda s: -s["effective_gap_min"])
    samples = samples[:5]
    # Severity uses EFFECTIVE gap (gap minus known CME halts).
    severity = "PASS"
    if max_effective >= GAP_CRITICAL_MIN:
        severity = "CRITICAL"
    elif max_effective >= GAP_WARN_MIN:
        severity = "WARN"
    max_gap_min = float(gaps.max()) / 60_000
    return {
        "severity": severity,
        "gap_count": int(len(gaps)),
        "max_gap_min": round(max_gap_min, 2),
        "max_effective_gap_min": round(max_effective, 2),
        "samples": samples,
        "note": "" if severity == "PASS"
                else f"largest unscheduled gap {max_effective:.1f}min",
    }


def check_ohlc_sanity(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"severity": "PASS", "violations": 0, "samples": [],
                "note": "no rows to check"}
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    violations = (
        (h < o) | (h < c) |
        (l > o) | (l > c) |
        (h < l) |
        (c <= 0) | (o <= 0) |
        (v < 0)
    )
    count = int(violations.sum())
    severity = "PASS" if count == 0 else "CRITICAL"
    samples = []
    if count > 0:
        bad = df[violations].head(5)
        for _, row in bad.iterrows():
            samples.append({
                "ts": row["timestamp_utc"].isoformat(),
                "o": float(row["open"]), "h": float(row["high"]),
                "l": float(row["low"]), "c": float(row["close"]),
                "v": float(row["volume"]),
            })
    return {
        "severity": severity,
        "violations": count,
        "total_rows": len(df),
        "samples": samples,
        "note": "" if count == 0 else f"{count} row(s) with invalid OHLCV",
    }


def check_price_plausibility(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"severity": "PASS", "latest_close": None, "note": "no rows"}
    latest_close = float(df["close"].iloc[-1])
    in_range = PRICE_MIN <= latest_close <= PRICE_MAX
    severity = "PASS" if in_range else "WARN"
    return {
        "severity": severity,
        "latest_close": latest_close,
        "range": [PRICE_MIN, PRICE_MAX],
        "note": "" if in_range else f"close {latest_close} outside [{PRICE_MIN}, {PRICE_MAX}]",
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"severity": "PASS", "duplicate_count": 0, "samples": [],
                "note": "no rows"}
    dup_mask = df["timestamp_utc"].duplicated(keep=False)
    count = int(dup_mask.sum())
    severity = "PASS" if count == 0 else "CRITICAL"
    samples = []
    if count > 0:
        dup_ts = df[dup_mask]["timestamp_utc"].drop_duplicates().head(5)
        samples = [ts.isoformat() for ts in dup_ts]
    return {
        "severity": severity,
        "duplicate_count": count,
        "samples": samples,
        "note": "" if count == 0 else f"{count} row(s) share timestamps",
    }


# ── aggregation + report ──────────────────────────────────────────────────


def aggregate_summary(checks: dict) -> dict:
    severities = [c["severity"] for c in checks.values()]
    crit = sum(1 for s in severities if s == "CRITICAL")
    warn = sum(1 for s in severities if s == "WARN")
    passed = sum(1 for s in severities if s == "PASS")
    if crit:
        status = "CRITICAL"
    elif warn:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "checks_run": len(checks),
        "checks_passed": passed,
        "checks_warn": warn,
        "checks_critical": crit,
    }


def build_report() -> dict:
    now_utc = datetime.now(timezone.utc)
    df = load_recent_bars(WINDOW_HOURS)
    checks = {
        "freshness": check_freshness(df, now_utc),
        "quantity": check_quantity(df, WINDOW_HOURS),
        "continuity": check_continuity(df),
        "ohlc_sanity": check_ohlc_sanity(df),
        "price_plausibility": check_price_plausibility(df),
        "duplicate_timestamps": check_duplicates(df),
    }
    summary = aggregate_summary(checks)
    return {
        "generated_at_utc": now_utc.isoformat(),
        "timezone": TZ,
        "buffer_path": str(BUFFER_DB),
        "window_hours": WINDOW_HOURS,
        "summary": summary,
        **checks,
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    status_emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}[s["status"]]
    now_local = datetime.fromisoformat(report["generated_at_utc"]).astimezone(
        ZoneInfo(report["timezone"])
    )
    lines = [
        f"# Live Buffer Data Health — {now_local:%Y-%m-%d %H:%M %Z}",
        "",
        f"**Status: {status_emoji} {s['status']}**  "
        f"({s['checks_passed']}/{s['checks_run']} pass, "
        f"{s['checks_warn']} warn, {s['checks_critical']} critical)",
        "",
        f"Buffer: `{report['buffer_path']}`",
        f"Window: last `{report['window_hours']}h`",
        "",
        "## Checks",
        "",
        "| Check | Severity | Detail |",
        "| --- | --- | --- |",
    ]
    detail_map = {
        "freshness":
            lambda c: f"latest @ {c['latest_bar_ts'] or '—'} ({c['age_seconds']}s ago)",
        "quantity":
            lambda c: f"{c['actual_bars']}/{c['expected_bars']} bars ({c['ratio']*100:.1f}%)",
        "continuity":
            lambda c: (f"{c['gap_count']} gap(s), max {c['max_gap_min']}min "
                       f"(effective {c['max_effective_gap_min']}min after CME halt)"
                       if c["gap_count"] else "no gaps"),
        "ohlc_sanity":
            lambda c: f"{c['violations']}/{c['total_rows']} invalid",
        "price_plausibility":
            lambda c: f"close {c['latest_close']} (range {c['range'][0]}-{c['range'][1]})",
        "duplicate_timestamps":
            lambda c: f"{c['duplicate_count']} dup row(s)",
    }
    emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}
    for key in ("freshness", "quantity", "continuity", "ohlc_sanity",
                "price_plausibility", "duplicate_timestamps"):
        c = report[key]
        detail = detail_map[key](c)
        note = f" — {c['note']}" if c.get("note") else ""
        lines.append(f"| {key} | {emoji[c['severity']]} {c['severity']} | {detail}{note} |")
    return "\n".join(lines) + "\n"


def render_telegram(report: dict) -> str:
    s = report["summary"]
    status_emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}[s["status"]]
    now_local = datetime.fromisoformat(report["generated_at_utc"]).astimezone(
        ZoneInfo(report["timezone"])
    )
    lines = [
        f"🩺 *Data Health* `{now_local:%Y-%m-%d %H:%M %Z}`",
        f"Status: *{status_emoji} {s['status']}*  "
        f"({s['checks_passed']}/{s['checks_run']} ok)",
        "",
    ]
    emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}
    one_liners = {
        "freshness": lambda c: f"Fresh: `{c['age_seconds']}s` ago",
        "quantity": lambda c: f"Qty: `{c['actual_bars']}` bars "
                              f"({c['ratio']*100:.1f}%)",
        "continuity": lambda c: (f"Cont: `{c['gap_count']}` gap(s), "
                                 f"max `{c['max_gap_min']}` / eff `{c['max_effective_gap_min']}` min"),
        "ohlc_sanity": lambda c: f"OHLC: `{c['violations']}` invalid "
                                 f"/ `{c['total_rows']}` rows",
        "price_plausibility": lambda c: f"Price: `{c['latest_close']}`",
        "duplicate_timestamps": lambda c: f"Dup: `{c['duplicate_count']}` row(s)",
    }
    for key in ("freshness", "quantity", "continuity", "ohlc_sanity",
                "price_plausibility", "duplicate_timestamps"):
        c = report[key]
        lines.append(f"{emoji[c['severity']]} {one_liners[key](c)}")
        if c.get("note") and c["severity"] != "PASS":
            lines.append(f"  _{c['note']}_")
    return "\n".join(lines)


# ── Telegram helpers (re-use parity pattern) ──────────────────────────────


def _load_telegram_env() -> tuple[str, str]:
    if not TELEGRAM_ENV_PATH.exists():
        return "", ""
    token, chat_id = "", ""
    for line in TELEGRAM_ENV_PATH.read_text().strip().split("\n"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "TELEGRAM_BOT_TOKEN":
            token = v.strip()
        elif k == "TELEGRAM_CHAT_ID":
            chat_id = v.strip()
    return token, chat_id


def _telegram_send(token: str, chat_id: str, text: str) -> bool:
    import urllib.parse
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()
    try:
        resp = urllib.request.urlopen(url, data, timeout=10)
        result = json.loads(resp.read())
        return bool(result.get("ok"))
    except Exception as exc:
        print(f"telegram send error: {exc}", file=sys.stderr)
        return False


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--telegram", choices=("always", "never", "critical-only"),
                   default="always",
                   help="Telegram push mode (default: always)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory for JSON + Markdown reports")
    args = p.parse_args()

    report = build_report()
    summary = report["summary"]

    now_local = datetime.fromisoformat(report["generated_at_utc"]).astimezone(
        ZoneInfo(TZ)
    )
    date_str = now_local.strftime("%Y-%m-%d")
    tz_slug = TZ.replace("/", "-")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"data_health_{date_str}_{tz_slug}.json"
    md_path = args.out_dir / f"data_health_{date_str}_{tz_slug}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))

    print(f"Status: {summary['status']} "
          f"({summary['checks_passed']}/{summary['checks_run']} pass, "
          f"{summary['checks_warn']} warn, {summary['checks_critical']} critical)")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")

    push = False
    if args.telegram == "always":
        push = True
    elif args.telegram == "critical-only":
        push = summary["status"] == "CRITICAL"

    if push:
        token, chat_id = _load_telegram_env()
        if token and chat_id:
            ok = _telegram_send(token, chat_id, render_telegram(report))
            print(f"Telegram push: {'OK' if ok else 'FAILED'}")
        else:
            print("Telegram push: SKIPPED (env missing)")

    # Exit code: 0 = PASS/WARN, 2 = CRITICAL — so timer log catches CRIT runs.
    return 0 if summary["status"] != "CRITICAL" else 2


if __name__ == "__main__":
    sys.exit(main())
