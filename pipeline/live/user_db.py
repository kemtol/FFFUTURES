#!/usr/bin/env python3
"""
User database — lightweight SQLite for user profiles, chat channels,
and strategy subscriptions.

No class, just module-level functions. Thread-safe via WAL mode.

Schema:
  users:          user_id, name, created_at
  user_chats:     user_id, platform, chat_id
  subscriptions:  user_id, strategy, active

Usage:
  from pipeline.live.user_db import get_subscribers, subscribe, upsert_user
  chat_ids = get_subscribers("tv_strategy")  # → ["7980136995", ...]
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "Live" / "users.db"

_lock = threading.Lock()


def _init_db() -> None:
    with _lock:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_chats (
                user_id    TEXT NOT NULL,
                platform   TEXT NOT NULL,
                chat_id    TEXT NOT NULL,
                PRIMARY KEY (user_id, platform),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id    TEXT NOT NULL,
                strategy   TEXT NOT NULL,
                active     INTEGER NOT NULL DEFAULT 1,
                subscribed_at TEXT NOT NULL,
                PRIMARY KEY (user_id, strategy),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)
        conn.commit()
        conn.close()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def upsert_user(user_id: str, name: str = "") -> None:
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET name = excluded.name",
            [user_id, name, datetime.now(timezone.utc).isoformat()],
        )
        conn.commit()
        conn.close()


def upsert_chat(user_id: str, platform: str, chat_id: str) -> None:
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT INTO user_chats (user_id, platform, chat_id) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, platform) DO UPDATE SET chat_id = excluded.chat_id",
            [user_id, platform, chat_id],
        )
        conn.commit()
        conn.close()


def subscribe(user_id: str, strategy: str) -> None:
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT OR IGNORE INTO subscriptions (user_id, strategy, subscribed_at) "
            "VALUES (?, ?, ?)",
            [user_id, strategy, datetime.now(timezone.utc).isoformat()],
        )
        conn.execute(
            "UPDATE subscriptions SET active = 1 WHERE user_id = ? AND strategy = ?",
            [user_id, strategy],
        )
        conn.commit()
        conn.close()


def unsubscribe(user_id: str, strategy: str) -> None:
    with _lock:
        conn = _db()
        conn.execute(
            "UPDATE subscriptions SET active = 0 WHERE user_id = ? AND strategy = ?",
            [user_id, strategy],
        )
        conn.commit()
        conn.close()


def subscribe_by_chat(chat_id: str, strategy: str, platform: str = "telegram") -> None:
    """Subscribe via chat_id (resolves user_id internally)."""
    with _lock:
        conn = _db()
        row = conn.execute(
            "SELECT user_id FROM user_chats WHERE chat_id = ? AND platform = ?",
            [chat_id, platform],
        ).fetchone()
        conn.close()
    if row:
        subscribe(row["user_id"], strategy)


def unsubscribe_by_chat(chat_id: str, strategy: str, platform: str = "telegram") -> None:
    """Unsubscribe via chat_id (resolves user_id internally)."""
    with _lock:
        conn = _db()
        row = conn.execute(
            "SELECT user_id FROM user_chats WHERE chat_id = ? AND platform = ?",
            [chat_id, platform],
        ).fetchone()
        conn.close()
    if row:
        unsubscribe(row["user_id"], strategy)


def get_subscribers(strategy: str, platform: str = "telegram") -> list[str]:
    """Return list of chat_ids subscribed to a strategy on a given platform."""
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT uc.chat_id FROM user_chats uc "
            "JOIN subscriptions s ON s.user_id = uc.user_id "
            "WHERE s.strategy = ? AND s.active = 1 AND uc.platform = ?",
            [strategy, platform],
        ).fetchall()
        conn.close()
        return [r["chat_id"] for r in rows]


def get_user_subs(user_id: str) -> list[dict]:
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT strategy, active FROM subscriptions WHERE user_id = ? ORDER BY strategy",
            [user_id],
        ).fetchall()
        conn.close()
        return [{"strategy": r["strategy"], "active": bool(r["active"])} for r in rows]


def list_subscribers(strategy: str) -> list[dict]:
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT u.user_id, u.name FROM users u "
            "JOIN subscriptions s ON s.user_id = u.user_id "
            "WHERE s.strategy = ? AND s.active = 1",
            [strategy],
        ).fetchall()
        conn.close()
        return [{"user_id": r["user_id"], "name": r["name"]} for r in rows]


def get_subscriptions_by_chat(chat_id: str, platform: str = "telegram") -> list[str]:
    """Return list of active strategy names for a given chat_id."""
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT s.strategy FROM subscriptions s "
            "JOIN user_chats uc ON uc.user_id = s.user_id "
            "WHERE uc.chat_id = ? AND uc.platform = ? AND s.active = 1 "
            "ORDER BY s.strategy",
            [chat_id, platform],
        ).fetchall()
        conn.close()
        return [r["strategy"] for r in rows]


# Initialize on import
_init_db()
