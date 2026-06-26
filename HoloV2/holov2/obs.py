"""Observability for HoloV2 — nested timing spans + structured events.

Opt-in and **no-op when disabled** (the fast path pays ~nothing). Used ONLY at the
orchestrator seams (``prepare/runner``, ``targets/pipeline``) — NEVER inside the pure ops
they call, so the compute stays clean. This is the timing/log analogue of ``viz`` reading
``FrameTrace``: observability is a wrapper, not an embedded hook.

Usage
-----
    prof = Profile()                      # enabled
    with prof.span("frame", f=12):
        with prof.span("eval", n_points=3925, n_channels=3):
            ...
        prof.event("sdf cache hit", item="obj0")
    print(prof.render())                  # indented tree with durations + %

    process_frame(..., prof=NULL)         # default everywhere -> ~zero overhead
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    meta: dict
    duration: float = 0.0                       # seconds; 0 for point events
    is_event: bool = False
    children: list["Span"] = field(default_factory=list)


class Profile:
    """Collects a nested span tree. ``enabled=False`` makes ``span``/``event`` no-ops.

    Pass an optional ``logger`` to also emit one structured debug line per span on exit.
    """

    def __init__(self, enabled: bool = True, logger=None) -> None:
        self.enabled = enabled
        self._logger = logger
        self._root = Span("root", {})
        self._stack: list[Span] = [self._root]

    @contextmanager
    def span(self, name: str, **meta):
        if not self.enabled:
            yield None
            return
        node = Span(name, meta)
        self._stack[-1].children.append(node)
        self._stack.append(node)
        t0 = time.perf_counter()
        try:
            yield node
        finally:
            node.duration = time.perf_counter() - t0
            self._stack.pop()
            if self._logger is not None:
                pad = "  " * (len(self._stack) - 1)
                self._logger.debug("%s%s %.2fms %s", pad, name, node.duration * 1e3, meta or "")

    def event(self, name: str, **meta) -> None:
        if not self.enabled:
            return
        self._stack[-1].children.append(Span(name, meta, is_event=True))
        if self._logger is not None:
            self._logger.debug("· %s %s", name, meta or "")

    def tree(self) -> Span:
        return self._root

    def render(self) -> str:
        """Indented flame-tree: name, duration (ms), % of parent, and meta."""
        lines: list[str] = []

        def walk(span: Span, depth: int, parent_dur: float) -> None:
            for c in span.children:
                pad = "  " * depth
                if c.is_event:
                    lines.append(f"{pad}· {c.name} {c.meta or ''}".rstrip())
                else:
                    pct = f" {100 * c.duration / parent_dur:4.0f}%" if parent_dur > 0 else ""
                    meta = f" {c.meta}" if c.meta else ""
                    lines.append(f"{pad}{c.name}  {c.duration * 1e3:8.2f}ms{pct}{meta}")
                    walk(c, depth + 1, c.duration)

        walk(self._root, 0, 0.0)
        return "\n".join(lines)


# Shared disabled singleton — the default everywhere. Safe to share: when disabled,
# span/event return immediately and never mutate state.
NULL = Profile(enabled=False)
