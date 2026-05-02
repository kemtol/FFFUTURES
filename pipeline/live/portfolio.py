#!/usr/bin/env python3
class PortfolioTracker:
    """Tracks open positions and trade history for paper trading."""

    def __init__(self, risk_per_r: float = 100.0):
        self.risk_per_r = risk_per_r
        self.open_positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.balance = 0.0
        self.start_balance = 0.0
        self._load_state()

    def _state_file(self):
        return ROOT / "data" / "Live" / "portfolio_state.json"

    def _load_state(self):
        f = self._state_file()
        if f.exists():
            try:
                data = json.loads(f.read_text())
                self.closed_trades = data.get("closed_trades", [])
                self.balance = data.get("balance", 0.0)
                self.start_balance = data.get("start_balance", 0.0)
            except Exception:
                pass
        if self.start_balance == 0.0:
            self.start_balance = 50000.0  # Topstep starting balance
        if self.balance == 0.0:
            self.balance = self.start_balance

    def _save_state(self):
        f = self._state_file()
        try:
            data = {
                "closed_trades": self.closed_trades[-500:],
                "balance": self.balance,
                "start_balance": self.start_balance,
                "open_positions": self.open_positions,
            }
            f.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def open(self, signal: dict) -> dict | None:
        """Open a position from a signal. Closes opposite position if exists."""
        direction = "SHORT" if (signal["side"] == "BULL" and signal["decision"] == "REV") or \
                               (signal["side"] == "BEAR" and signal["decision"] == "CONT") else "LONG"

        # Close opposite positions first
        for pos in list(self.open_positions):
            if pos["direction"] != direction:
                self._close_position(pos, 0.0, "REVERSED")

        # Open new position
        pos = {
            "id": len(self.closed_trades) + len(self.open_positions) + 1,
            "direction": direction,
            "session": signal["session"],
            "entry": signal["entry"],
            "tp": signal["tp"],
            "sl": signal["sl"],
            "rr": signal["rr_ratio"],
            "open_time": str(signal["ts"]),
            "open_price": signal["entry"],
            "current_price": signal["entry"],
            "unrealized_pnl": 0.0,
        }
        self.open_positions.append(pos)
        self._save_state()
        return pos

    def update(self, current_price: float) -> None:
        """Update unrealized PnL and check TP/SL for all open positions."""
        for pos in list(self.open_positions):
            pos["current_price"] = current_price
            if pos["direction"] == "LONG":
                pnl = (current_price - pos["entry"]) * self.risk_per_r / (abs(pos["entry"] - pos["sl"]) + 0.01)
            else:
                pnl = (pos["entry"] - current_price) * self.risk_per_r / (abs(pos["entry"] - pos["sl"]) + 0.01)
            pos["unrealized_pnl"] = round(pnl, 2)

            # Check TP/SL
            if pos["direction"] == "LONG":
                if current_price >= pos["tp"]:
                    self._close_position(pos, pos["rr"] * self.risk_per_r, "TP")
                elif current_price <= pos["sl"]:
                    self._close_position(pos, -self.risk_per_r, "SL")
            else:
                if current_price <= pos["tp"]:
                    self._close_position(pos, pos["rr"] * self.risk_per_r, "TP")
                elif current_price >= pos["sl"]:
                    self._close_position(pos, -self.risk_per_r, "SL")

    def _close_position(self, pos: dict, pnl: float, reason: str) -> None:
        self.open_positions.remove(pos)
        trade = {
            **pos,
            "pnl": round(pnl - 3.0, 2),  # $3 commission
            "close_reason": reason,
            "close_time": str(datetime.now(timezone.utc)),
        }
        self.closed_trades.append(trade)
        self.balance += trade["pnl"]
        self._save_state()

    def stats(self) -> str:
        total_trades = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        wr = wins / total_trades if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        open_pnl = sum(p["unrealized_pnl"] for p in self.open_positions)
        return (
            f"💰 *Portfolio*\n\n"
            f"Balance: `${self.balance:,.0f}`\n"
            f"Start: `${self.start_balance:,.0f}`\n"
            f"Total PnL: `${total_pnl:+,.0f}` (closed)\n"
            f"Open PnL: `${open_pnl:+,.0f}` ({len(self.open_positions)} pos)\n"
            f"Trades: `{total_trades}` | Win rate: `{wr:.1%}`"
        )

    def pnl_summary(self) -> str:
        lines = []
        for pos in self.open_positions:
            lines.append(
                f"{'🟢' if pos['direction'] == 'LONG' else '🔴'} {pos['session'].upper()} "
                f"{pos['direction']} @ `${pos['entry']:.1f}` | "
                f"PnL: `${pos['unrealized_pnl']:+,.0f}` | "
                f"TP: `${pos['tp']:.1f}` SL: `${pos['sl']:.1f}`"
            )
        if self.closed_trades:
            last = self.closed_trades[-3:]
            lines.append("\n*Last 3 closed:*")
            for t in last:
                emoji = "✅" if t["pnl"] > 0 else "❌"
                lines.append(f"{emoji} {t['session'].upper()} {t['direction']} "
                           f"`${t['pnl']:+,.0f}` ({t['close_reason']})")
        return "\n".join(lines) if lines else "No positions"



