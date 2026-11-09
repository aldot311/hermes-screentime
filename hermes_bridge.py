"""Everything that touches Hermes: paths, settings, the Meta key, the host model. Imported only inside Hermes.

Hermes imports are local to each function, so `import hermes_bridge` never fails outside Hermes and Plugin Doctor's
registration check (which blocks sockets) never sees network access.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PLUGIN_ID = "screentime"
AUX_TASK = "screentime_verify"
KEY_ENVS = ("META_MODEL_API_KEY", "MODEL_API_KEY", "META_API_KEY")  # same names Hermes's meta-ai provider reads
PLUGIN_DIR = Path(__file__).resolve().parent
DEMO_DIR = PLUGIN_DIR / "fixtures" / "demo"

_CTX: Any = None
_AUX_REGISTERED = False


def bind(ctx: Any, *, aux_registered: bool) -> None:
    global _CTX, _AUX_REGISTERED
    _CTX = ctx
    _AUX_REGISTERED = aux_registered


def data_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir

    return Path(plugin_data_dir(PLUGIN_ID))


def hermes_home() -> Path:
    # <HERMES_HOME>/plugin-data/<id>: derived from the documented storage helper, not a private core import.
    return data_dir().parent.parent


def work_dir() -> Path:
    d = data_dir() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(run_id: str) -> Path:
    """Deliverables live in the generated-media cache, a trusted delivery root on every gateway."""
    d = hermes_home() / "cache" / "generated" / PLUGIN_ID / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def db() -> sqlite3.Connection:
    from plugins.plugin_storage import plugin_db

    return plugin_db(PLUGIN_ID)


def setting(key: str, default: Any) -> Any:
    try:
        val = _CTX.get_config(key, default=default) if _CTX is not None else default
    except Exception:
        val = default
    if val is None:
        return default
    if isinstance(default, bool):
        return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        try:
            return type(default)(val)
        except (TypeError, ValueError):
            return default
    return val


def meta_key() -> Optional[str]:
    """The Meta Model API key through Hermes's secret scope only (catalog rule 11). Never logged or returned."""
    from agent.secret_scope import get_secret

    for name in KEY_ENVS:
        try:
            v = get_secret(name)
        except Exception:
            v = None
        if v and v.strip():
            return v.strip()
    return None


def _agent_cwd() -> Path:
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return Path(resolve_agent_cwd())
    except Exception:  # outside Hermes (tests, plain CLI use)
        return Path.cwd()


def allowed_roots() -> list[Path]:
    """Where a tool may read user files: the session's folder, Hermes's home (chat attachments land in its cache),
    and the user's home directory. System folders (/etc, /var, /proc, ...) are never read."""
    roots = [_agent_cwd(), Path.home()]
    try:
        roots.append(hermes_home())
    except Exception as exc:  # plugin storage unavailable outside Hermes
        log.debug("screentime: no Hermes home (%s)", exc)
    return [r.resolve() for r in roots]


def resolve_path(p: str) -> Path:
    """A user-named local file inside allowed_roots(). Relative paths resolve against the session's folder; URLs,
    option-like names and media directives are refused; symlinks are followed and judged by where they land."""
    from .screentime_core.errors import ScreentimeError

    raw = (p or "").strip().strip('"').strip("'")
    if not raw or "://" in raw or raw.startswith("-") or "MEDIA:" in raw.upper() or "\x00" in raw:
        raise ScreentimeError("bad_input", "Give a local file path (an attachment's path works).",
                              fix="Attach the video again, or give its full path.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _agent_cwd() / path
    path = path.resolve()
    if not any(path.is_relative_to(root) for root in allowed_roots()):
        raise ScreentimeError("bad_input", f"{path.name} is outside the folders Screen-Time reads.",
                              fix="Attach the file in the chat, or move it into your home or working folder.")
    if not path.is_file():
        raise ScreentimeError("bad_input", f"File not found: {path.name}", fix="Attach it again or give the full path.")
    return path


def owner_key(session_key: str = "") -> Optional[str]:
    """The chat a run belongs to. 'local' for the plain CLI (one user); None inside a gateway chat whose identity is
    unknown, in which case results are neither stored nor read (fail closed, never a shared bucket)."""
    if session_key:
        return session_key
    try:
        from gateway.session_context import get_session_env
    except ImportError:  # not running inside Hermes
        return "local"
    platform = (get_session_env("HERMES_SESSION_PLATFORM") or "").strip()
    for name in ("HERMES_SESSION_KEY", "HERMES_SESSION_CHAT_ID"):
        v = (get_session_env(name) or "").strip()
        if v:
            return v if name == "HERMES_SESSION_KEY" else f"{platform or 'chat'}:{v}"
    return None if platform and platform not in ("cli", "local") else "local"


def hermes_structured_call():
    """A callable for HermesLlmVerifier: the user's model (or the screentime_verify aux model) with image blocks."""
    from agent.plugin_llm import PluginLlmImageInput

    def call(instructions: str, images: list, json_schema: dict, schema_name: str) -> dict:
        blocks = [PluginLlmImageInput(data=data, mime_type=mime) for data, mime in images]
        res = _CTX.llm.complete_structured(
            instructions=instructions,
            input=blocks,
            json_schema=json_schema,
            schema_name=schema_name,
            task=AUX_TASK if _AUX_REGISTERED else None,
            purpose="screentime product check",
            max_tokens=800,
            timeout=90,
        )
        usage = getattr(res, "usage", None)
        return {
            "parsed": getattr(res, "parsed", None),
            "text": getattr(res, "text", ""),
            "provider": getattr(res, "provider", ""),
            "model": getattr(res, "model", ""),
            "usd": getattr(usage, "cost_usd", None) if usage is not None else None,
        }

    return call
