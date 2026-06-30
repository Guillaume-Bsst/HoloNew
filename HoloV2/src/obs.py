"""Observabilité pour HoloV2 — spans de temps imbriqués + événements structurés.

Opt-in et **no-op quand désactivé** (le chemin rapide coûte ~rien). Utilisé UNIQUEMENT aux
seams de l'orchestrateur (``prepare/runner``, ``targets/pipeline``) — JAMAIS à l'intérieur des ops purs
qu'ils appellent, donc le calcul reste propre. C'est l'analogue timing/log de ``viz`` lisant
``FrameTrace`` : l'observabilité est un wrapper, pas un hook imbriqué.

Utilisation
-----------
    prof = Profile()                      # activé
    with prof.span("frame", f=12):
        with prof.span("eval", n_points=3925, n_channels=3):
            ...
        prof.event("sdf cache hit", item="obj0")
    print(prof.render())                  # arbre indenté avec durées + %

    process_frame(..., prof=NULL)         # défaut partout → ~zéro surcharge
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    meta: dict
    duration: float = 0.0                       # secondes ; 0 pour les événements ponctuels
    is_event: bool = False
    children: list["Span"] = field(default_factory=list)


class Profile:
    """Collecte un arbre de spans imbriqué. ``enabled=False`` rend ``span``/``event`` des no-ops.

    Passer un ``logger`` optionnel pour émettre aussi une ligne de débogage structurée par span à la sortie.
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
        """Arbre enflammé indenté : nom, durée (ms), % du parent, et métadonnées."""
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


# Singleton partagé désactivé — le défaut partout. Sûr à partager : quand désactivé,
# span/event retournent immédiatement et ne mutent jamais l'état.
NULL = Profile(enabled=False)
