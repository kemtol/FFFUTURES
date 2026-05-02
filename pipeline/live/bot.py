#!/usr/bin/env python3
class TelegramBot:
    def __init__(self):
        self.token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = _os.environ.get("TELEGRAM_CHAT_ID", "")
        env_file = ROOT / "data" / "Live" / "telegram.env"
        if env_file.exists():
            for line in env_file.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "TELEGRAM_BOT_TOKEN" and not self.token:
                        self.token = v
                    elif k == "TELEGRAM_CHAT_ID" and not self.chat_id:
                        self.chat_id = v
        self._last_update_id = self._fetch_latest_update_id()
        self.enabled = bool(self.token and self.chat_id)

    def _fetch_latest_update_id(self) -> int:
        """Get the latest update_id from Telegram so we only process new messages."""
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            resp = urllib.request.urlopen(url, timeout=5)
            result = json.loads(resp.read())
            if result.get("ok") and result.get("result"):
                return result["result"][-1]["update_id"]
        except Exception:
            pass
        return 0

    def send(self, msg: str) -> None:
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": msg,
                "parse_mode": "Markdown",
            }).encode()
            urllib.request.urlopen(url, data, timeout=5)
        except Exception:
            pass

    def poll(self, signal_runner) -> None:
        """Check for incoming commands once. Call periodically."""
        if not self.enabled:
            return
        try:
            offset = self._last_update_id + 1

            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = urllib.parse.urlencode({
                "offset": max(1, offset),
                "timeout": 0,  # no long-poll — return immediately
            })
            full_url = f"{url}?{params}"
            req = urllib.request.Request(full_url)
            resp = urllib.request.urlopen(req, timeout=5)
            result = json.loads(resp.read())
            if result.get("ok") and result.get("result"):
                for update in result["result"]:
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {}).get("text", "")
                    chat = update.get("message", {}).get("chat", {}).get("id")
                    if chat and msg:
                        print(f"[Telegram] Command: {msg}", flush=True)
                        self._handle_command(msg, chat, signal_runner)
        except Exception as e:
            print(f"[Telegram] Poll error: {e}", flush=True)

    def _handle_command(self, msg: str, chat_id: int, runner) -> None:
        msg = msg.strip().lower()

        if msg == "/status" or msg == "/health":
            stats = runner.stats()
            latest = runner.buffer.latest()
            now = datetime.now(timezone.utc)
            data_age = (now - latest["ts"]).seconds // 60 if latest else 999
            gold_price = f"${latest['close']:.1f}" if latest else "N/A"

            # Current session info
            from pipeline.live.orb_detector import SESSIONS
            current_sess = None
            prev_sess = None
            for i, s in enumerate(SESSIONS):
                t = now.time()
                if s.open_utc <= t <= s.close_utc:
                    current_sess = s
                    break
                elif t > s.close_utc:
                    prev_sess = s

            if current_sess:
                current_str = current_sess.name.upper()
            elif prev_sess:
                current_str = f"IDLE (after {prev_sess.name.upper()})"
            else:
                current_str = "IDLE"

            # Find next session
            next_sess = ""
            for s in SESSIONS:
                if now.time() < s.open_utc:
                    next_open = datetime.combine(now.date(), s.open_utc, tzinfo=timezone.utc)
                    delta = next_open - now
                    next_sess = f"{s.name.upper()} in {delta.seconds//3600}h {(delta.seconds%3600)//60}m"
                    break
            if not next_sess:
                next_sess = "TOKYO tomorrow"

            self.send(
                f"📊 *ORB v2.0 Status*\n\n"
                f"Session: `{current_str}`\n"
                f"Next: `{next_sess}`\n"
                f"Gold: `{gold_price}`\n"
                f"1m candles: `{data_age}m old` (yfinance)\n"
                f"Events: `{runner._total_events}` | Signals: `{runner._total_signals}`\n"
                f"Portfolio: `${runner.portfolio.balance:,.0f}` "
                f"({len(runner.portfolio.open_positions)} open)"
            )

        elif msg == "/last":
            # Show from portfolio (persisted) first, then live signals
            total = len(runner.portfolio.closed_trades) + len(runner.portfolio.open_positions)
            if total > 0:
                lines = [f"📡 *Today's Activity* ({total} trades/positions)\n"]
                for pos in runner.portfolio.open_positions:
                    lines.append(f"🔵 OPEN {pos['direction']} @ `${pos['entry']:.1f}` | PnL: `${pos['unrealized_pnl']:+,.0f}`")
                for t in runner.portfolio.closed_trades[-5:]:
                    emoji = "✅" if t["pnl"] > 0 else "❌"
                    lines.append(f"{emoji} {t['direction']} `${t['pnl']:+,.0f}` ({t['close_reason']})")
                self.send("\n".join(lines))
            elif runner.signals:
                last = runner.signals[-1]
                ts = pd.Timestamp(last["ts"]).strftime("%H:%M UTC")
                self.send(
                    f"📡 *Last Signal*\n\n"
                    f"Time: `{ts}`\n"
                    f"Session: *{last['session'].upper()}*\n"
                    f"Decision: *{last['decision']}*\n"
                    f"Entry: `${last['entry']:.1f}`"
                )
            else:
                self.send("📡 *No signals yet today.*")

        elif msg == "/health":
            self.send("Use /status for health info.")

        elif msg == "/pnl":
            self.send(runner.portfolio.pnl_summary() or "No positions")

        elif msg == "/portfolio":
            self.send(runner.portfolio.stats())
        elif msg == "/features":
            if runner.signals:
                last = runner.signals[-1]
                self.send(
                    f"📐 *Last Signal Features*\n\n"
                    f"Prob(Rev): `{last['prob_rev']:.3f}`\n"
                    f"Prob(Cont): `{last['prob_cont']:.3f}`\n"
                    f"Session: `{last['session']}` | Side: `{last['side']}`\n"
                    f"Entry: `${last['entry']:.1f}`"
                )
            else:
                latest = runner.buffer.latest()
                self.send(
                    f"📐 *Current Market*\n\n"
                    f"Gold: `${latest['close']:.1f}`\n"
                    f"Data: yfinance (~10-15m delay)\n"
                    f"Features: 42 total, 7 modules"
                )
        elif msg.startswith("/"):
            self.send("Commands: /status /last /pnl /portfolio /features")


