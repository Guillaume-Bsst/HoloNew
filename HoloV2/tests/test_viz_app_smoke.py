"""Smoke test headless de l'app viz — deux garanties complémentaires :

1. ``import src.viz`` / ``import src.viz.app`` N'importe PAS viser (viser est différé dans
   ``Player.run()`` ; la logique dispatch + les couches doivent rester testables sans écran).
   Vérifié dans un sous-processus frais pour isoler sys.modules.

2. Setup + update des 7 couches portées sur un VRAI ``viser.ViserServer`` (vrais handles
   viser, pas des fakes).  Exerce les setters ``.points/.colors/.visible`` sur les handles
   réels — ferme le risque différé « LineSegmentsHandle.points setter non testé contre vrais
   handles ».  Aucune boucle bloquante : pas d'appel à ``Player.run()`` / ``while True`` ; le
   serveur est détruit après le test (threads daemon).

Gabarit de skip/spec identique à ``test_viz_bake_source.py`` (données HODome requises).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
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
from src.viz.layers.skeleton import SkeletonLayer
from src.viz.layers.style import StyleLayer
from src.viz.sources import BakeSource
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX

# Chemins internes au dépôt (stables, pas dans paths.toml)
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"

# Racine HoloV2/ (cwd pour le sous-processus et pour importer src.*)
_HOLOV2 = Path(__file__).resolve().parent.parent


def _pick() -> Path | None:
    """Sélectionne la première séquence HODome commune smplx + object."""
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
# Skipif : données absentes → skip (données présentes en local → le test tourne)
_SKIP_DATA = pytest.mark.skipif(
    _SEQ is None,
    reason="HODome data / SMPL-X model / corr_neutral.npz absent",
)


def _robot() -> RobotSpec:
    """RobotSpec réel (même URDF que les autres tests de données)."""
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


def _spec(cache: Path) -> SceneSpec:
    """Construit le SceneSpec HODome de démo (même pattern que test_viz_bake_source)."""
    return SceneSpec(
        dataset="hodome",
        motion_path=_SEQ,
        robot=_robot(),
        smpl_model_dir=_SMPLX,
        cache_dir=cache,
    )


# ---------------------------------------------------------------------------
# Garantie 1 — import léger (viser non chargé à l'import de src.viz / src.viz.app)
# ---------------------------------------------------------------------------

def test_viz_app_import_ne_charge_pas_viser():
    """``import src.viz`` et ``import src.viz.app`` ne doivent PAS charger viser dans sys.modules.
    Testé dans un sous-processus isolé pour éviter la contamination de session pytest."""
    script = (
        "import sys; "
        "import src.viz; import src.viz.app; "
        "dans = 'viser' in sys.modules; "
        "assert not dans, "
        "f\"viser chargé à l'import de src.viz.app — modules: {list(sys.modules)}\""
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_HOLOV2),
    )
    assert result.returncode == 0, (
        f"viser importé à tort lors de 'import src.viz.app' :\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Garantie 2 — smoke réel : setup + update des 7 couches sur vrais handles viser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Cache temporaire avec corr_neutral.npz copié (même pattern que test_viz_bake_source)."""
    c = tmp_path_factory.mktemp("viz_smoke_cache")
    (c / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, c / "correspondence" / "corr_neutral.npz")
    return c


@_SKIP_DATA
def test_viz_smoke_setup_update_sept_couches_vrais_handles(cache):
    """Smoke headless : setup + update des 7 couches portées sur un VRAI ViserServer.

    Construit BakeSource (2 frames, frame_step=8), démarre un ViserServer sur le port 18081
    (sans aucune boucle bloquante), appelle ``layer.setup`` puis ``layer.update`` sur chaque
    couche × frame — exerce les setters .points/.colors/.visible sur les vrais handles viser.
    Le serveur est détruit proprement (threads daemon) à la sortie du bloc try/finally.
    """
    import viser  # import différé : viser est un effet de bord (daemon threads)

    # Construction de la source avec données réelles, peu de frames pour la rapidité
    source = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=2)
    ctx = source.context

    # UiState nominal : canal ground, heatmap distance
    ui = UiState(channel=ctx.channel_names[0], color_mode="distance", point_size=0.01)

    # Les 7 couches identiques à run_app (ordre canonique)
    layers: list = [
        GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
        ObjectsLayer(), FieldsLayer(), StyleLayer(),
    ]

    # Port haut arbitraire pour éviter les collisions avec les outils locaux
    srv = viser.ViserServer(port=18081)
    try:
        # --- setup : crée les handles persistants via les wrappers viser_ops ---
        for layer in layers:
            layer.setup(srv, srv.gui, ctx)

        # --- update × frames : exerce les setters sur vrais handles viser ---
        for i in range(source.n_frames):
            frame = source.get(i)
            for layer in layers:
                layer.update(frame, ui)  # aucune exception ne doit être levée

    finally:
        # Destruction propre : threads daemon, pas de while True actif → pas de blocage
        del srv
