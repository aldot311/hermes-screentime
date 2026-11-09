"""Run index: every Measure, Teardown and Compare result, so follow-up questions cost nothing.

Reports are stored as JSON. A run belongs to the chat that made it (``owner``, the session routing key); lookups from
another chat do not see it.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from typing import Any, Optional

from .models import to_dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    owner TEXT NOT NULL,
    created_at REAL NOT NULL,
    video_sha256 TEXT,
    product TEXT,
    summary TEXT,
    report TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_owner ON runs(owner, created_at);
CREATE INDEX IF NOT EXISTS runs_video ON runs(video_sha256, mode);
"""


def new_run_id(prefix: str = "st") -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


class RunStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.executescript(SCHEMA)
        conn.commit()

    def save(self, report: Any, *, owner: str, summary: str = "") -> None:
        data = to_dict(report)
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, mode, owner, created_at, video_sha256, product, summary, report) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                data["run_id"],
                data.get("mode", "?"),
                owner or "local",
                time.time(),
                data.get("video_sha256") or (data.get("yours") or {}).get("video_sha256"),
                data.get("product"),
                summary[:500],
                json.dumps(data),
            ),
        )
        self.conn.commit()

    def get(self, run_id: str, *, owner: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT report FROM runs WHERE run_id = ? AND owner = ?", (run_id, owner or "local")
        ).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, *, owner: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT run_id, mode, created_at, product, summary FROM runs WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
            (owner or "local", int(limit)),
        ).fetchall()
        return [
            {"run_id": r[0], "mode": r[1], "created_at": r[2], "product": r[3], "summary": r[4]} for r in rows
        ]

    def latest_measure_for(self, video_sha256: str, *, owner: str) -> Optional[dict]:
        """The newest Measure report of the same file in this chat (Teardown reuses its product timing)."""
        row = self.conn.execute(
            "SELECT report FROM runs WHERE video_sha256 = ? AND mode = 'measure' AND owner = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (video_sha256, owner or "local"),
        ).fetchone()
        return json.loads(row[0]) if row else None
