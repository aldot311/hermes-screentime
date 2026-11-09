"""Spend caps for paid Meta Model API calls, held in SQLite so they survive restarts and hold across sessions.

Every paid call reserves its estimated cost BEFORE it is made. A reservation that would take a run past its per-video cap
or the day past its per-day cap is refused, and no request is sent. After the call, the reservation is settled to the
actual cost when it is known (SAM bills by frames), otherwise it stays at the estimate. Estimates are upper bounds; they
are not Meta's bill.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional

from .errors import ScreentimeError

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    day TEXT NOT NULL,
    purpose TEXT NOT NULL,
    reserved_usd REAL NOT NULL,
    actual_usd REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_run ON spend(run_id);
CREATE INDEX IF NOT EXISTS spend_day ON spend(day);
"""


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@dataclass
class Reservation:
    id: int
    run_id: str
    usd: float


class Budget:
    def __init__(self, conn: sqlite3.Connection, *, per_video_usd: float, per_day_usd: float):
        if per_video_usd < 0 or per_day_usd < 0:
            raise ValueError("caps must be >= 0")
        self.conn = conn
        self.per_video_usd = float(per_video_usd)
        self.per_day_usd = float(per_day_usd)
        conn.executescript(SCHEMA)
        conn.commit()

    _SUM = {
        "run_id": "SELECT COALESCE(SUM(COALESCE(actual_usd, reserved_usd)), 0) FROM spend WHERE run_id = ?",
        "day": "SELECT COALESCE(SUM(COALESCE(actual_usd, reserved_usd)), 0) FROM spend WHERE day = ?",
    }

    def _spent(self, where: str, arg: str) -> float:
        row = self.conn.execute(self._SUM[where], (arg,)).fetchone()
        return float(row[0] or 0.0)

    def spent_run(self, run_id: str) -> float:
        return self._spent("run_id", run_id)

    def spent_today(self) -> float:
        return self._spent("day", _today())

    def reserve(self, run_id: str, usd: float, purpose: str) -> Reservation:
        """Reserve ``usd`` for one paid call, or raise ScreentimeError("budget") without reserving anything."""
        usd = max(0.0, float(usd))
        day = _today()
        # BEGIN IMMEDIATE takes the write lock first, so two runs at the same moment cannot both pass the check.
        if self.conn.in_transaction:
            self.conn.commit()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            run_total = self._spent("run_id", run_id)
            day_total = self._spent("day", day)
            if run_total + usd > self.per_video_usd + 1e-9:
                raise ScreentimeError(
                    "budget",
                    f"This video would cost more than its ${self.per_video_usd:.2f} cap "
                    f"(${run_total:.3f} so far + ${usd:.3f} for {purpose}).",
                    fix="Raise max_usd_per_video in the plugin settings, or measure a shorter clip.",
                )
            if day_total + usd > self.per_day_usd + 1e-9:
                raise ScreentimeError(
                    "budget",
                    f"Today's ${self.per_day_usd:.2f} cap would be passed (${day_total:.3f} spent today).",
                    fix="Raise max_usd_per_day in the plugin settings, or wait until tomorrow (UTC).",
                )
            cur = self.conn.execute(
                "INSERT INTO spend(run_id, day, purpose, reserved_usd, actual_usd, created_at) VALUES (?,?,?,?,NULL,?)",
                (run_id, day, purpose[:200], usd, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return Reservation(int(cur.lastrowid), run_id, usd)

    def settle(self, res: Reservation, actual_usd: Optional[float]) -> None:
        """Record what the call actually cost (None keeps the estimate)."""
        if actual_usd is None:
            return
        self.conn.execute("UPDATE spend SET actual_usd = ? WHERE id = ?", (max(0.0, float(actual_usd)), res.id))
        self.conn.commit()

    def release(self, res: Reservation) -> None:
        """Free a reservation for a call that was refused before anything was billed (4xx, no request sent)."""
        self.conn.execute("UPDATE spend SET actual_usd = 0 WHERE id = ?", (res.id,))
        self.conn.commit()


# Estimates (upper bounds) used for reservations. SAM bills $0.20 per 1,000 frames with a 50-frame minimum per request.
SAM_USD_PER_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50
SPARK_CHECK_RESERVE_USD = 0.02  # the app's flat reservation per Muse Spark call
TRANSCRIBE_RESERVE_USD = 0.01  # the app's flat reservation per Voice Transcribe call


def sam_estimate(frames: int) -> float:
    return max(int(frames), SAM_MIN_FRAMES) * SAM_USD_PER_FRAME


def measure_estimate(frames: int, *, objects_to_check: int = 3, transcribe: bool = True, spark_checks: bool = True) -> float:
    """Rough upper bound for one Measure run, shown to the user before anything is billed."""
    usd = sam_estimate(frames)
    if spark_checks:
        usd += objects_to_check * SPARK_CHECK_RESERVE_USD
    if transcribe:
        usd += TRANSCRIBE_RESERVE_USD
    return round(usd, 3)
