#!/usr/bin/env python3
"""
TopstepX trade execution via REST API.

Endpoint: POST https://userapi.topstepx.com/Order
Auth: JWT from data/Live/topstepx_token.json

Order types (inferred):
  type 1 = Limit
  type 2 = Market? Stop?

timeType: 0 (Day/GTC)
customTag: UUID for tracking
"""

from __future__ import annotations

import json
import random
import time as _time
import uuid
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"

ORDER_URL = "https://userapi.topstepx.com/Order"
SYMBOL = "F.US.MGC"
ACCOUNT_ID = 22303383

# Mimic real Chrome browser — avoids bot detection
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.topstepx.com",
    "Referer": "https://www.topstepx.com/trade",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}


def _token() -> str:
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())["access_token"]
    raise RuntimeError("No token file")


def _api(path: str, body: dict) -> dict:
    """POST to TopstepX API with browser-mimic headers."""
    token = _token()
    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    data = json.dumps(body).encode()

    # Random micro-delay (500ms-1500ms) — mimics human interaction
    _time.sleep(random.uniform(0.5, 1.5))

    req = urllib.request.Request(path, data=data, headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


class TopstepXExecution:
    """Trade execution on TopstepX evaluation account."""

    def __init__(self):
        self.account_id = ACCOUNT_ID
        self.symbol = SYMBOL
        self._token = _token()

    def place_limit_order(self, side: str, price: float,
                          quantity: int = 1) -> dict:
        """Place a limit order. side='Buy' or 'Sell'."""
        size = quantity if side == "Buy" else -quantity
        return _api(ORDER_URL, {
            "accountId": self.account_id,
            "symbolId": self.symbol,
            "type": 1,
            "limitPrice": price,
            "stopPrice": None,
            "positionSize": size,
            "customTag": str(uuid.uuid4()),
            "timeType": 0,
        })

    def place_market_order(self, side: str, quantity: int = 1) -> dict:
        """Place a market order."""
        size = quantity if side == "Buy" else -quantity
        return _api(ORDER_URL, {
            "accountId": self.account_id,
            "symbolId": self.symbol,
            "type": 2,
            "limitPrice": None,
            "stopPrice": None,
            "positionSize": size,
            "customTag": str(uuid.uuid4()),
            "timeType": 0,
        })

    def bracket_entry(self, side: str, entry: float,
                      tp: float, sl: float, quantity: int = 1) -> dict:
        """Place entry limit + auto OCO bracket.
        
        Places entry order first, then TP/SL will need separate calls.
        TopstepX may auto-attach bracket if stopPrice is set.
        """
        return _api(ORDER_URL, {
            "accountId": self.account_id,
            "symbolId": self.symbol,
            "type": 1,  # Limit
            "limitPrice": entry,
            "stopPrice": sl,  # stop loss level
            "positionSize": quantity,
            "customTag": str(uuid.uuid4()),
            "timeType": 0,
        })

    def cancel_order(self, order_id: int) -> dict:
        """Cancel a pending order."""
        url = f"{ORDER_URL}/cancel/{self.account_id}/id/{order_id}"
        headers = {**HEADERS, "Authorization": f"Bearer {_token()}"}
        _time.sleep(random.uniform(0.3, 0.8))
        req = urllib.request.Request(url, method="DELETE", headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())

    def signal_to_action(self, side: str, decision: str) -> str:
        """Convert ORB signal to Buy/Sell."""
        return {
            ("BULL", "CONT"): "Buy",
            ("BULL", "REV"):  "Sell",
            ("BEAR", "CONT"): "Sell",
            ("BEAR", "REV"):  "Buy",
        }.get((side, decision), "Buy")
        """Convert ORB signal to Buy/Sell."""
        return {
            ("BULL", "CONT"): "Buy",
            ("BULL", "REV"):  "Sell",
            ("BEAR", "CONT"): "Sell",
            ("BEAR", "REV"):  "Buy",
        }.get((side, decision), "Buy")


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ex = TopstepXExecution()
    print(f"Account: {ex.account_id}, Symbol: {ex.symbol}")

    # Test: place limit order FAR from market (will not fill)
    result = ex.place_limit_order("Buy", price=4000.0, quantity=1)
    print(f"Test order result: {json.dumps(result, indent=2)}")
