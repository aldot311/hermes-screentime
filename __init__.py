"""Screen-Time for Hermes: registration only. No network, no ffmpeg, no file writes here."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def register(ctx) -> None:
    from . import cli, commands, hermes_bridge, schemas, tools

    registered: set[str] = set()
    tasks = (
        (hermes_bridge.AUX_TASK, "Screen-Time product check",
         "Checks crops of each tracked object against the product photo, so look-alikes are not counted."),
        (hermes_bridge.TRACK_TASK, "Screen-Time tracker",
         "Boxes the product's kind of object in sampled video frames when no Meta key is set."),
    )
    if hasattr(ctx, "register_auxiliary_task"):
        for key, name, desc in tasks:
            try:
                # Low reasoning: each call is a narrow look-and-box question, and tracking makes 20-30 of them.
                ctx.register_auxiliary_task(key, display_name=name, description=desc, inherit_from="vision",
                                            defaults={"reasoning_effort": "low"})
                registered.add(key)
            except Exception as exc:  # an older Hermes, or a name clash: fall back to the main model
                log.warning("screentime: auxiliary task %s not registered (%s)", key, exc)
    hermes_bridge.bind(ctx, aux_registered=hermes_bridge.AUX_TASK in registered, aux_tasks=registered)

    ctx.register_tool(name="screentime_measure", toolset="screentime", schema=schemas.MEASURE, handler=tools.measure)
    ctx.register_tool(name="screentime_compare", toolset="screentime", schema=schemas.COMPARE, handler=tools.compare)
    ctx.register_tool(name="screentime_runs", toolset="screentime", schema=schemas.RUNS, handler=tools.runs)
    ctx.register_command(
        "screentime",
        handler=commands.handle,
        description="Screen-Time: demo, selftest, status, last",
        args_hint="help | demo | selftest | status | last",
    )
    ctx.register_cli_command(
        name="screentime",
        help="Screen-Time: measure product screen time, compare edits",
        setup_fn=cli.setup,
        handler_fn=cli.handle,
    )
    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.is_dir():
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                ctx.register_skill(child.name, child / "SKILL.md")
