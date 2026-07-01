"""Smoke solve headless : RobotLayer + CostDashboard sur un VRAI ViserServer (solve=True,
max_frames=1). Ferme les chemins B2/B3 différés :

- ``ViserUrdf.update_cfg`` / ``show_visual`` appelés sur handles viser réels (RobotLayer.setup
  + update avec un SolvedFrame réel).
- ``CostDashboard.setup`` : matplotlib Agg → ``gui.add_image`` panel viser réel.

Skip si données HODome / SMPL-X / corr_neutral.npz absents (données présentes en local
→ le test tourne). Aucune boucle bloquante : pas d'appel à ``Player.run()``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.viz.core.layer import UiState
from src.viz.layers.fields import FieldsLayer
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.ground import GroundLayer
from src.viz.layers.human_cloud import HumanCloudLayer
from src.viz.layers.objects import ObjectsLayer
from src.viz.layers.robot import RobotLayer
from src.viz.layers.skeleton import SkeletonLayer
from src.viz.layers.style import StyleLayer
from src.viz.panels.cost_dashboard import CostDashboard
from src.viz.sources import BakeSource
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX

# Chemins internes au dépôt (stables, pas dans paths.toml)
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
_HOLOV2 = Path(__file__).resolve().parent.parent


def _pick() -> Path | None:
    """Sélectionne la première séquence HODome commune smplx + object."""
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
# Skipif : données absentes → skip (données présentes en local → le test tourne)
_SKIP_DATA = pytest.mark.skipif(
    _SEQ is None,
    reason="HODome data / SMPL-X model / corr_neutral.npz / g1_29dof.urdf absent",
)


def _robot() -> RobotSpec:
    """RobotSpec réel G1 (URDF 29 dof — même spec que les autres tests de données)."""
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


def _spec(cache: Path) -> SceneSpec:
    """Construit le SceneSpec HODome de démo (même pattern que test_viz_app_smoke)."""
    return SceneSpec(
        dataset="hodome",
        motion_path=_SEQ,
        robot=_robot(),
        smpl_model_dir=_SMPLX,
        cache_dir=cache,
    )


# ---------------------------------------------------------------------------
# Fixture cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Cache temporaire avec corr_neutral.npz copié (même pattern que test_viz_app_smoke)."""
    c = tmp_path_factory.mktemp("viz_solve_smoke_cache")
    (c / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, c / "correspondence" / "corr_neutral.npz")
    return c


# ---------------------------------------------------------------------------
# Smoke — ViserUrdf réel + CostDashboard réel
# ---------------------------------------------------------------------------

@_SKIP_DATA
def test_viz_solve_smoke_robot_layer_cost_dashboard(cache):
    """Smoke headless solve=True : setup + update de RobotLayer + CostDashboard sur un VRAI
    ViserServer (port 18083).

    Construit BakeSource(solve=True, max_frames=1, frame_step=50) pour limiter le temps de solve
    SQP (1 frame suffit à exercer les APIs). Appelle :

    - ``layer.setup(srv, srv.gui, ctx)`` sur les 8 couches (dont RobotLayer → ViserUrdf) ;
    - ``panel.setup(srv, srv.gui, source.frames)`` sur CostDashboard (matplotlib → add_image) ;
    - ``layer.update(frame, ui)`` pour le frame résolu (ViserUrdf.update_cfg / show_visual).

    Aucune exception = les vraies APIs ViserUrdf + gui.add_image fonctionnent headless.
    """
    import viser  # import différé : viser démarre des threads daemon

    # Solve sur 1 frame (grand frame_step pour tomber en milieu de séquence, loin des bords)
    source = BakeSource(_spec(cache), PrepareConfig(), solve=True, frame_step=50, max_frames=1)
    ctx = source.context

    # UiState nominal : canal ground, heatmap distance
    ui = UiState(channel=ctx.channel_names[0], color_mode="distance", point_size=0.01)

    # Les 8 couches identiques à run_app (ordre canonique, RobotLayer en dernier)
    layers: list = [
        GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
        ObjectsLayer(), FieldsLayer(), StyleLayer(), RobotLayer(),
    ]
    # Panel coût : agrège solved.cost_by_term sur toute la séquence
    panel = CostDashboard()

    # Port haut arbitraire pour éviter les collisions (18083 = B3 + solve)
    srv = viser.ViserServer(port=18083)
    try:
        # --- setup couches : crée les handles persistants (ViserUrdf pour RobotLayer) ---
        for layer in layers:
            layer.setup(srv, srv.gui, ctx)

        # --- setup panel : matplotlib Agg → chart empilée → gui.add_image ---
        panel.setup(srv, srv.gui, source.frames)

        # --- update × 1 frame : exerce ViserUrdf.update_cfg / show_visual ---
        frame = source.get(0)
        for layer in layers:
            layer.update(frame, ui)  # aucune exception ne doit être levée

    finally:
        # Destruction propre : threads daemon → pas de blocage
        del srv
