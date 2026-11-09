"""Screen-Time for Hermes: registration only. No network, no ffmpeg, no file writes here."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def register(ctx) -> None:
    from . import cli, commands, hermes_bridge, schemas, tools

    aux = False
    if hasattr(ctx, "register_auxiliary_task"):
        try:
            ctx.register_auxiliary_task(
                hermes_bridge.AUX_TASK,
                display_name="Screen-Time product check",
                description="Checks crops of each tracked object against the product photo, so look-alikes are not counted.",
                inherit_from="vision",
            )
            aux = True
        except Exception as exc:  # an older Hermes, or a name clash: fall back to the main model
            log.warning("screentime: auxiliary task not registered (%s)", exc)
    hermes_bridge.bind(ctx, aux_registered=aux)

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
