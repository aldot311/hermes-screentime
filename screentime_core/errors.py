"""Expected failures, each with a stable code the tool layer turns into one plain sentence."""

from __future__ import annotations

from typing import Optional

CODES = {
    "ffmpeg_missing",
    "bad_input",
    "too_long",
    "no_audio",
    "engine_unavailable",
    "auth",
    "billing",
    "model_not_found",
    "rate_limited",
    "server",
    "timeout",
    "refused",
    "incomplete",
    "budget",
    "verifier_unavailable",
}


class ScreentimeError(Exception):
    """An expected failure. ``code`` is one of CODES; ``fix`` tells the user what to do."""

    def __init__(self, code: str, message: str, *, fix: Optional[str] = None, retryable: bool = False):
        if code not in CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix
        self.retryable = retryable

    def to_dict(self) -> dict:
        out = {"error": self.code, "message": self.message}
        if self.fix:
            out["fix"] = self.fix
        return out
