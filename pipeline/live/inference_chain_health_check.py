#!/usr/bin/env python3
"""
Inference Chain Health Check (Phase 2).

Validates that the data + artifacts that feed into V8 router decisions
haven't silently drifted. Three categories per design
(plans/inference_chain_health.md):

  B. Model artifact integrity — sha256 of model files + config JSON
     validation. Detects silent model swap / corruption.
  C. L0 Raw ↔ L0 Live sync — last 6h overlap between MGC_1m.db and
     topstepx_buffer.db. Detects raw/live drift for research integrity.
  D. Datamart anchor sterility — v3=3643 rows + v1.12=1471 rows, sha256
     baseline. Detects accidental overwrites of sim source-of-truth.

Cadence: every 1 hour (sterile/slow-changing targets; 15min = noise).
Severity action: Telegram alert only (no auto-halt) — matches Phase 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "Live" / "health"
BASELINES_PATH = OUT_DIR / "inference_chain_baselines.json"
TELEGRAM_ENV_PATH = ROOT / "data" / "Live" / "telegram.env"

# B. Model artifact paths (sterile)
MODEL_PATHS_REL = [
    "model/SUPER_STRUCTURE/meta_v7/inference_model.txt",
    "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json",
    "model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt",
]
CONS_BRAIN_LIVE = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"
CONS_BRAIN_LEGACY = ROOT / "model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt"
REFINED_CONFIG = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json"

# C. L0 sync
RAW_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"
LIVE_DB = ROOT / "data" / "Live" / "topstepx_buffer.db"
SYNC_OVERLAP_HOURS = 6
SYNC_PRICE_TOLERANCE = 0.5      # 5 MGC ticks
SYNC_ROW_MISMATCH_WARN = 0.05   # 5%
SYNC_ROW_MISMATCH_CRITICAL = 0.30
SYNC_PRICE_DRIFT_WARN = 0.01    # 1% of rows drift
SYNC_PRICE_DRIFT_CRITICAL = 0.05

# D. Datamart anchors (sterile)
DATAMART_REL = "data/Level_2_Datamart/super_structure_ml"
DATAMART_ANCHORS = {
    f"{DATAMART_REL}/v3_final_training.parquet": 3643,
    f"{DATAMART_REL}/v1_12_training_datamart.parquet": 1471,
}

TZ = "Asia/Jakarta"


# ── helpers ───────────────────────────────────────────────────────────────


def sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_baselines() -> dict:
    if not BASELINES_PATH.exists():
        return {}
    return json.loads(BASELINES_PATH.read_text())


def save_baselines(d: dict) -> None:
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(d, indent=2))


def capture_baselines() -> dict:
    out = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_by": "first run",
        "model_artifacts": {},
        "datamart_anchors": {},
    }
    for rel in MODEL_PATHS_REL:
        p = ROOT / rel
        if p.exists():
            out["model_artifacts"][rel] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
    for rel, anchor_rows in DATAMART_ANCHORS.items():
        p = ROOT / rel
        if p.exists():
            df = pd.read_parquet(p)
            out["datamart_anchors"][rel] = {
                "rows": int(len(df)),
                "anchor_rows": int(anchor_rows),
                "sha256": sha256_file(p),
            }
    return out


# ── B. Model Artifact Integrity ───────────────────────────────────────────


def check_model_artifact(baselines: dict) -> dict:
    rows: list[dict] = []
    crit = 0
    warn = 0

    bm = baselines.get("model_artifacts", {})

    # Per-file existence + sha256 vs baseline
    for rel in MODEL_PATHS_REL:
        p = ROOT / rel
        item = {"path": rel, "exists": p.exists()}
        if not p.exists():
            item["severity"] = "CRITICAL"
            item["note"] = "file missing"
            crit += 1
            rows.append(item)
            continue
        sha = sha256_file(p)
        size = p.stat().st_size
        item["sha256"] = sha
        item["size_bytes"] = size
        base = bm.get(rel)
        if not base:
            item["severity"] = "PASS"
            item["note"] = "baseline captured this run"
        elif sha == base.get("sha256"):
            item["severity"] = "PASS"
            item["note"] = ""
        else:
            item["severity"] = "CRITICAL"
            item["note"] = f"sha256 drift vs baseline {base.get('sha256')[:12]}..."
            crit += 1
        rows.append(item)

    # cross-equality: inference_model.txt == conservative_brain.txt
    equality = {"check": "inference_model_eq_conservative_brain"}
    try:
        live_sha = sha256_file(CONS_BRAIN_LIVE)
        legacy_sha = sha256_file(CONS_BRAIN_LEGACY)
        if live_sha == legacy_sha:
            equality["severity"] = "PASS"
            equality["note"] = "byte-identical"
        else:
            equality["severity"] = "CRITICAL"
            equality["note"] = "meta_v7 ≠ SMART_1 conservative_brain"
            crit += 1
    except Exception as exc:
        equality["severity"] = "CRITICAL"
        equality["note"] = f"comparison failed: {exc}"
        crit += 1

    # refined config JSON structural validation
    cfg_check = {"check": "refined_config_structure"}
    try:
        cfg = json.loads(REFINED_CONFIG.read_text())
        thresholds = cfg.get("thresholds")
        ok = (
            isinstance(thresholds, dict)
            and all(k in thresholds for k in ("0", "1", "2"))
            and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
                    for v in thresholds.values())
        )
        if ok:
            cfg_check["severity"] = "PASS"
            cfg_check["thresholds"] = {k: float(v) for k, v in thresholds.items()}
        else:
            cfg_check["severity"] = "CRITICAL"
            cfg_check["note"] = "thresholds missing keys 0/1/2 or out of [0,1]"
            crit += 1
    except Exception as exc:
        cfg_check["severity"] = "CRITICAL"
        cfg_check["note"] = f"config JSON parse failed: {exc}"
        crit += 1

    # lightgbm loadability
    lgb_check = {"check": "lightgbm_loadable"}
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(CONS_BRAIN_LIVE))
        lgb_check["severity"] = "PASS"
        lgb_check["num_features"] = booster.num_feature()
        lgb_check["num_trees"] = booster.num_trees()
    except Exception as exc:
        lgb_check["severity"] = "CRITICAL"
        lgb_check["note"] = f"LightGBM load failed: {exc}"
        crit += 1

    severity = "CRITICAL" if crit else ("WARN" if warn else "PASS")
    return {
        "severity": severity,
        "critical": crit,
        "warn": warn,
        "files": rows,
        "equality": equality,
        "config": cfg_check,
        "lightgbm": lgb_check,
    }


# ── C. L0 Raw ↔ L0 Live sync ──────────────────────────────────────────────


def _snapshot_db(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    src_uri = f"file:{src}?mode=ro"
    with sqlite3.connect(src_uri, uri=True, timeout=30) as s:
        with sqlite3.connect(str(dst), timeout=30) as d:
            s.backup(d)


def _detect_ohlcv_table(con: sqlite3.Connection) -> str | None:
    """MGC_1m.db uses `investing_ohlcv_1m`; topstepx_buffer.db uses `ohlcv_1m`.
    Pick the first table whose name starts with a known OHLC prefix and has
    the expected columns.
    """
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    candidates = [t for t in tables if "ohlcv_1m" in t and "yfinance" not in t]
    return candidates[0] if candidates else None


def _read_bars(db: Path, hours: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(hours=hours)
    con = sqlite3.connect(str(db), timeout=30)
    try:
        table = _detect_ohlcv_table(con)
        if not table:
            return pd.DataFrame()
        df = pd.read_sql(
            f"SELECT timestamp_utc, open, high, low, close, volume "
            f"FROM {table} WHERE timestamp_utc >= ? ORDER BY timestamp_utc",
            con, params=[start.strftime("%Y-%m-%d %H:%M:%S")],
        )
    finally:
        con.close()
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def check_l0_live_sync() -> dict:
    if not RAW_DB.exists() or not LIVE_DB.exists():
        return {
            "severity": "WARN",
            "note": "one of L0 Raw / L0 Live DBs missing",
            "raw_exists": RAW_DB.exists(),
            "live_exists": LIVE_DB.exists(),
        }
    raw_snap = Path("/tmp/health_l0_raw_snapshot.db")
    live_snap = Path("/tmp/health_l0_live_snapshot.db")
    try:
        _snapshot_db(RAW_DB, raw_snap)
        _snapshot_db(LIVE_DB, live_snap)
    except Exception as exc:
        return {"severity": "WARN", "note": f"snapshot failed: {exc}"}

    raw = _read_bars(raw_snap, SYNC_OVERLAP_HOURS)
    live = _read_bars(live_snap, SYNC_OVERLAP_HOURS)
    if raw.empty:
        # Research raw DB (MGC_1m.db) is batch-ingested — staleness vs the
        # 6h window is expected unless someone re-ran a Databento ingest.
        # Not a live-trading issue, so PASS with note.
        return {"severity": "PASS",
                "note": "raw DB stale (batch-ingested, expected)",
                "raw_rows": 0, "live_rows": int(len(live))}
    if live.empty:
        # Live empty IS a problem — but Phase 1 freshness check owns that.
        return {"severity": "WARN", "note": "live DB empty in overlap window",
                "raw_rows": int(len(raw)), "live_rows": 0}

    # Overlap = intersection of timestamp_utc sets.
    merged = raw.merge(
        live, on="timestamp_utc", how="outer",
        suffixes=("_raw", "_live"), indicator=True,
    )
    both = merged[merged["_merge"] == "both"]
    raw_only = (merged["_merge"] == "left_only").sum()
    live_only = (merged["_merge"] == "right_only").sum()
    total = int(len(merged))
    mismatch_ratio = float((raw_only + live_only) / total) if total else 0.0

    drift_count = 0
    if not both.empty:
        for col in ("open", "high", "low", "close"):
            delta = (both[f"{col}_raw"] - both[f"{col}_live"]).abs()
            drift_count = max(drift_count, int((delta > SYNC_PRICE_TOLERANCE).sum()))
    drift_ratio = (drift_count / len(both)) if len(both) else 0.0

    severity = "PASS"
    notes = []
    if mismatch_ratio >= SYNC_ROW_MISMATCH_CRITICAL:
        severity = "CRITICAL"
        notes.append(f"row mismatch {mismatch_ratio*100:.1f}% (≥ {SYNC_ROW_MISMATCH_CRITICAL*100:.0f}%)")
    elif mismatch_ratio >= SYNC_ROW_MISMATCH_WARN:
        severity = "WARN"
        notes.append(f"row mismatch {mismatch_ratio*100:.1f}%")
    if drift_ratio >= SYNC_PRICE_DRIFT_CRITICAL:
        severity = "CRITICAL"
        notes.append(f"OHLC drift in {drift_ratio*100:.1f}% rows (≥ {SYNC_PRICE_DRIFT_CRITICAL*100:.0f}%)")
    elif drift_ratio >= SYNC_PRICE_DRIFT_WARN and severity != "CRITICAL":
        severity = "WARN"
        notes.append(f"OHLC drift in {drift_ratio*100:.1f}% rows")

    return {
        "severity": severity,
        "window_hours": SYNC_OVERLAP_HOURS,
        "raw_rows": int(len(raw)),
        "live_rows": int(len(live)),
        "overlap_rows": int(len(both)),
        "raw_only": int(raw_only),
        "live_only": int(live_only),
        "mismatch_ratio": round(mismatch_ratio, 4),
        "ohlc_drift_count": int(drift_count),
        "ohlc_drift_ratio": round(drift_ratio, 4),
        "tolerance": SYNC_PRICE_TOLERANCE,
        "note": "; ".join(notes),
    }


# ── D. Datamart Anchor Sterility ──────────────────────────────────────────


def check_datamart_anchors(baselines: dict) -> dict:
    rows = []
    crit = 0
    bm = baselines.get("datamart_anchors", {})
    for rel, anchor_rows in DATAMART_ANCHORS.items():
        p = ROOT / rel
        item = {"path": rel, "anchor_rows": int(anchor_rows)}
        if not p.exists():
            item["severity"] = "CRITICAL"
            item["note"] = "file missing"
            crit += 1
            rows.append(item)
            continue
        df = pd.read_parquet(p)
        actual_rows = int(len(df))
        sha = sha256_file(p)
        item["actual_rows"] = actual_rows
        item["sha256"] = sha
        base = bm.get(rel) or {}
        if actual_rows != anchor_rows:
            item["severity"] = "CRITICAL"
            item["note"] = f"rows {actual_rows} != anchor {anchor_rows}"
            crit += 1
        elif base.get("sha256") and base["sha256"] != sha:
            item["severity"] = "CRITICAL"
            item["note"] = f"sha256 drift (rows still {anchor_rows})"
            crit += 1
        else:
            item["severity"] = "PASS"
            item["note"] = ""
        rows.append(item)
    severity = "CRITICAL" if crit else "PASS"
    return {"severity": severity, "critical": crit, "files": rows}


# ── aggregation + report ──────────────────────────────────────────────────


def build_report() -> dict:
    now_utc = datetime.now(timezone.utc)
    baselines = load_baselines()
    first_run = not baselines
    if first_run:
        baselines = capture_baselines()
        save_baselines(baselines)

    model = check_model_artifact(baselines)
    sync = check_l0_live_sync()
    dm = check_datamart_anchors(baselines)

    severities = [model["severity"], sync["severity"], dm["severity"]]
    if "CRITICAL" in severities:
        status = "CRITICAL"
    elif "WARN" in severities:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "generated_at_utc": now_utc.isoformat(),
        "timezone": TZ,
        "first_run": first_run,
        "baselines_captured_at": baselines.get("captured_at_utc"),
        "summary": {
            "status": status,
            "first_run": first_run,
            "model_severity": model["severity"],
            "l0_sync_severity": sync["severity"],
            "datamart_severity": dm["severity"],
        },
        "model_artifact": model,
        "l0_live_sync": sync,
        "datamart_anchors": dm,
    }


def render_markdown(r: dict) -> str:
    s = r["summary"]
    emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}
    now_local = datetime.fromisoformat(r["generated_at_utc"]).astimezone(
        ZoneInfo(r["timezone"])
    )
    lines = [
        f"# Inference Chain Health — {now_local:%Y-%m-%d %H:%M %Z}",
        "",
        f"**Status: {emoji[s['status']]} {s['status']}**"
        + (" (first run — baselines captured)" if r["first_run"] else ""),
        "",
        f"Baselines captured: `{r['baselines_captured_at']}`",
        "",
        "## B. Model Artifact Integrity",
        f"Severity: {emoji[r['model_artifact']['severity']]} "
        f"{r['model_artifact']['severity']}",
        "",
        "| File | Severity | Note |",
        "| --- | --- | --- |",
    ]
    for f in r["model_artifact"]["files"]:
        lines.append(f"| `{f['path']}` | {emoji[f['severity']]} {f['severity']} | {f.get('note', '')} |")
    for k in ("equality", "config", "lightgbm"):
        c = r["model_artifact"][k]
        lines.append(f"| _{c.get('check', k)}_ | {emoji[c['severity']]} {c['severity']} | {c.get('note', '')} |")
    cfg = r["model_artifact"]["config"]
    if cfg.get("thresholds"):
        lines.append(f"")
        lines.append(f"Refined thresholds: `{cfg['thresholds']}`")

    lines += [
        "",
        "## C. L0 Raw ↔ L0 Live Sync",
        f"Severity: {emoji[r['l0_live_sync']['severity']]} "
        f"{r['l0_live_sync']['severity']}",
        "",
    ]
    sync = r["l0_live_sync"]
    if "overlap_rows" in sync:
        lines.append(f"- Raw rows: `{sync['raw_rows']}`, Live rows: `{sync['live_rows']}`, "
                     f"overlap: `{sync['overlap_rows']}`")
        lines.append(f"- Mismatched: `{sync['mismatch_ratio']*100:.1f}%` "
                     f"(raw-only {sync['raw_only']}, live-only {sync['live_only']})")
        lines.append(f"- OHLC drift > {sync['tolerance']}: `{sync['ohlc_drift_count']}` rows "
                     f"({sync['ohlc_drift_ratio']*100:.1f}%)")
    elif "raw_rows" in sync:
        lines.append(f"- Raw rows: `{sync['raw_rows']}`, Live rows: `{sync['live_rows']}` "
                     f"(no overlap to compare)")
    if sync.get("note"):
        lines.append(f"- Note: {sync['note']}")

    lines += [
        "",
        "## D. Datamart Anchor Sterility",
        f"Severity: {emoji[r['datamart_anchors']['severity']]} "
        f"{r['datamart_anchors']['severity']}",
        "",
        "| File | Anchor | Actual | Severity | Note |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for f in r["datamart_anchors"]["files"]:
        lines.append(f"| `{f['path']}` | {f['anchor_rows']} | {f.get('actual_rows', '—')} | "
                     f"{emoji[f['severity']]} {f['severity']} | {f.get('note', '')} |")

    return "\n".join(lines) + "\n"


def render_telegram(r: dict) -> str:
    s = r["summary"]
    emoji = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}
    now_local = datetime.fromisoformat(r["generated_at_utc"]).astimezone(
        ZoneInfo(r["timezone"])
    )
    lines = [
        f"🔗 *Inference Chain Health* `{now_local:%Y-%m-%d %H:%M %Z}`",
        f"Status: *{emoji[s['status']]} {s['status']}*"
        + (" — first run" if r["first_run"] else ""),
        "",
        f"{emoji[s['model_severity']]} Model artifact: {s['model_severity']}",
        f"{emoji[s['l0_sync_severity']]} L0 Raw↔Live sync: {s['l0_sync_severity']}",
        f"{emoji[s['datamart_severity']]} Datamart anchors: {s['datamart_severity']}",
    ]
    if s["status"] != "PASS":
        if r["model_artifact"]["severity"] != "PASS":
            for f in r["model_artifact"]["files"]:
                if f["severity"] != "PASS":
                    lines.append(f"  _model {f['path'].split('/')[-1]}: {f['note']}_")
            for k in ("equality", "config", "lightgbm"):
                c = r["model_artifact"][k]
                if c["severity"] != "PASS":
                    lines.append(f"  _{c.get('check', k)}: {c['note']}_")
        if r["l0_live_sync"]["severity"] != "PASS":
            lines.append(f"  _L0 sync: {r['l0_live_sync'].get('note', '')}_")
        if r["datamart_anchors"]["severity"] != "PASS":
            for f in r["datamart_anchors"]["files"]:
                if f["severity"] != "PASS":
                    lines.append(f"  _{f['path'].split('/')[-1]}: {f['note']}_")
    return "\n".join(lines)


# ── Telegram helper (re-use Phase 1 pattern) ──────────────────────────────


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
                   default="always")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    report = build_report()
    summary = report["summary"]

    now_local = datetime.fromisoformat(report["generated_at_utc"]).astimezone(
        ZoneInfo(TZ)
    )
    date_str = now_local.strftime("%Y-%m-%d")
    tz_slug = TZ.replace("/", "-")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"inference_chain_{date_str}_{tz_slug}.json"
    md_path = args.out_dir / f"inference_chain_{date_str}_{tz_slug}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))

    print(f"Status: {summary['status']} (model={summary['model_severity']}, "
          f"l0_sync={summary['l0_sync_severity']}, "
          f"datamart={summary['datamart_severity']})")
    if report["first_run"]:
        print(f"First run — baselines saved: {BASELINES_PATH}")
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

    return 0 if summary["status"] != "CRITICAL" else 2


if __name__ == "__main__":
    sys.exit(main())
