"""Panel ``cost_dashboard`` (roadmap #2) — coût total + ``cost_by_term`` empilé sur TOUS les frames
résolus, avec marqueurs status/n_iters pour les frames non-convergées. Rendu avec matplotlib (Agg)
dans une image viser (plotly est absent de l'env, donc ``gui.add_plotly`` n'est PAS utilisé).
Lit ``frame.solved`` sur toute la séquence (fournie par la Source au setup).
matplotlib/viser confinés à ce module."""
from __future__ import annotations

import numpy as np

# Statuts considérés comme convergés
_OK = ("optimal", "optimal_inaccurate")
# Nom du folder viser pour ce panel
_FOLDER = "Cost dashboard"


def stack_cost_terms(frames):
    """Empile les coûts par terme sur la séquence. PUR (testable, sans matplotlib/viser).

    Retourne ``(names, matrix (F, K), total (F,), status (F,), n_iters (F,))`` :
    - ``names`` = union des clés ``cost_by_term`` dans l'ordre de première apparition ;
    - un frame avec ``solved is None`` -> ligne nulle + status vide ;
    - un terme absent d'un frame -> 0.
    """
    # Récupère le champ solved de chaque frame (None si absent)
    solved = [getattr(fr, "solved", None) for fr in frames]

    # Construction de l'union des noms de termes, ordre de première apparition
    names: list[str] = []
    for s in solved:
        if s is not None:
            for k in s.cost_by_term:
                if k not in names:
                    names.append(k)

    F, K = len(frames), len(names)
    mat = np.zeros((F, K), np.float64)
    total = np.zeros(F, np.float64)
    status: list[str] = [""] * F
    n_iters = np.zeros(F, np.int64)

    for i, s in enumerate(solved):
        if s is None:
            # Frame non résolue : ligne nulle, status/n_iters vides
            continue
        total[i] = float(s.cost) if np.isfinite(s.cost) else 0.0
        status[i] = str(s.status)
        n_iters[i] = int(s.n_iters)
        for j, k in enumerate(names):
            mat[i, j] = float(s.cost_by_term.get(k, 0.0))

    return tuple(names), mat, total, tuple(status), n_iters


def _render_chart(names, mat, total, status) -> np.ndarray:
    """Rendu matplotlib (Agg) : aire empilée par terme + courbe total + marqueurs non-convergées.

    Retourne une image RGB uint8 (H, W, 3) passable à ``gui.add_image``."""
    # Import différé : matplotlib n'est pas nécessaire au niveau module (tests purs headless)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = mat.shape[0]
    x = np.arange(F)
    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=100)

    if names and F > 0:
        # Aires empilées par terme (légende = noms des termes)
        ax.stackplot(x, mat.T, labels=list(names))

    # Courbe du coût total
    ax.plot(x, total, color="k", lw=1.5, label="total")

    # Marqueurs rouges pour les frames non-convergées
    bad = [i for i, st in enumerate(status) if st not in _OK]
    if bad:
        ax.scatter(bad, total[np.asarray(bad)], color="r", zorder=5, s=20, label="non-converged")

    ax.set_xlabel("frame")
    ax.set_ylabel("cost (Σ residuals²)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()

    # Conversion canvas -> tableau numpy RGB uint8
    rgba = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4) uint8
    plt.close(fig)
    return rgba[..., :3].copy()                   # (H, W, 3) RGB


def _summary_md(status, n_iters) -> str:
    """Résumé markdown des frames non-convergées (tableau frame/status/n_iters ou message OK)."""
    bad = [
        (i, st, int(n))
        for i, (st, n) in enumerate(zip(status, n_iters))
        if st not in _OK and st
    ]
    if not bad:
        return "**solve** — all frames converged (optimal)."
    lines = [
        "**solve — non-converged frames**",
        "",
        "| frame | status | n_iters |",
        "|---|---|---|",
    ]
    lines += [f"| {i} | {st} | {n} |" for i, st, n in bad]
    return "\n".join(lines)


class CostDashboard:
    """Panel 2D. ``setup`` une fois avec la séquence entière -> image de coût + résumé non-convergées."""

    folder = _FOLDER

    def setup(self, server, gui, frames) -> None:
        """Construit et affiche le panel dans le folder viser dédié.

        Paramètres
        ----------
        server : serveur viser (non utilisé directement, passé par convention)
        gui    : handle GUI viser (``add_folder``, ``add_image``, ``add_markdown``)
        frames : séquence de VizFrame (``frame.solved`` peut être None)
        """
        names, mat, total, status, n_iters = stack_cost_terms(frames)
        img = _render_chart(names, mat, total, status)
        with gui.add_folder(self.folder):
            gui.add_image(img, label="cost / cost_by_term")
            gui.add_markdown(_summary_md(status, n_iters))
