#!/usr/bin/env python3
"""Send a Telegram alert when MNQ L0 automation fails."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "data/Live/telegram.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--log", default=str(ROOT / "_LOG/mnq_l0_update.log"))
    return parser.parse_args()


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return "(log missing)"
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def main() -> int:
    args = parse_args()
    env = load_env(ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram env missing; cannot send MNQ L0 failure alert", file=sys.stderr)
        return 1

    log_tail = tail(Path(args.log))
    if len(log_tail) > 2500:
        log_tail = log_tail[-2500:]
    message = (
        "*MNQ L0 update failed*\n"
        f"exit_code: `{args.exit_code}`\n\n"
        "```text\n"
        f"{log_tail}\n"
        "```"
    )
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }
    ).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    with urllib.request.urlopen(url, data=payload, timeout=15) as response:
        body = response.read().decode("utf-8", errors="replace")
    result = json.loads(body)
    if not result.get("ok"):
        raise SystemExit(body)
    print("MNQ L0 failure alert sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
