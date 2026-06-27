"""targets/ orchestrator — per-frame construction of the cibles.

Composes the pure ops (style + interaction's pose/eval/transport) into ``FrameTargets``.
Instrumentation lives HERE via ``prof`` spans, NEVER inside the pure ops. ``trace_frame``
is the instrumented twin (keeps intermediates) for ``viz``. See docs/TARGETS.md, OBS.md.
"""
from __future__ import annotations

from ..obs import NULL


def process_frame(grounded, ctx, config, f, prof=NULL):
    """One frame -> FrameTargets.  Span plan (filled when targets/ is implemented):

        with prof.span("frame", f=f):
            with prof.span("pose"):                    pose = frame_pose(grounded, f)
            with prof.span("style"):                   style = style.build(pose, ...)
            with prof.span("interaction.pose"):        clouds = pose_cloud(...)
            with prof.span("interaction.eval", n_channels=C, n_points=P): fields = eval_fields(...)
            with prof.span("interaction.transport"):   robot = transport(...)
    """
    raise NotImplementedError


def trace_frame(grounded, ctx, config, f, prof=NULL):
    """Same pure ops as process_frame, but returns a FrameTrace (intermediates kept) for viz."""
    raise NotImplementedError


def run_sequence(grounded, ctx, config, prof=NULL):
    """Drive all frames: online loop ``for f: process_frame`` OR a vectorised batch over T.

        with prof.span("sequence", T=grounded.n_frames):
            for f in range(grounded.n_frames):
                ... process_frame(grounded, ctx, config, f, prof)
    """
    raise NotImplementedError
