"""Application viz prod — le viewer unifié. Construit un BakeSource, un Player et les sept couches
portées, puis sert. Remplace le legacy ``viewer.py`` (un god-class ~370 lignes) par des couches
composables : ajouter une couche de roadmap = un fichier dans ``layers/`` + une ligne ici.
Consommateur pur : viser reste dans ``core/viser_ops`` + les couches + le Player.

Lancement :
    fuser -k 8080/tcp   # libérer le port D'ABORD (ne jamais pkill -f ce script : il se tue seul)
    python -m src.viz.app --motion-path <smplx.npz> --model-dir <smplx_models> \\
        [--dataset hodome --port 8080 --frame-step 2 --max-frames 200]
"""
from __future__ import annotations

import argparse

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from .debug._args import add_scene_args, scene_from_args
from .core.player import Player
from .layers.contacts import ContactsLayer
from .layers.correspondence import CorrespondenceLayer
from .layers.fields import FieldsLayer
from .layers.geodesic import GeodesicLayer
from .layers.ghost import GhostLayer
from .layers.ground import GroundLayer
from .layers.human_cloud import HumanCloudLayer
from .layers.objects import ObjectsLayer
from .layers.robot import RobotLayer
from .layers.sdf_iso import SdfIsoLayer
from .layers.skeleton import SkeletonLayer
from .layers.style import StyleLayer
from .panels.cost_dashboard import CostDashboard
from .sources import BakeSource


def run_app(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200,
            solve: bool = False) -> None:
    """Construit BakeSource -> Player -> les 12 couches portées (7 + RobotLayer + 4 interaction) -> sert.

    ``solve=True`` (phase B) :
    - la source cuit ``SolvedFrame`` pour chaque frame (BakeSource exécute le solveur SQP) ;
    - ``RobotLayer`` affiche le robot résolu (elle se masque d'elle-même si ``solved is None``) ;
    - ``CostDashboard`` est ajouté comme panel et agrège les coûts sur toute la séquence.

    ``solve=False`` : comportement identique à la phase A (7 couches + RobotLayer masquée,
    aucun panel coût), non régressé. Les couches solve-gated (contacts, correspondance) se masquent
    d'elles-mêmes quand ``frame.solved is None``."""
    source = BakeSource(spec, PrepareConfig(), solve=solve, frame_step=frame_step,
                        max_frames=max_frames)
    # RobotLayer ajoutée toujours : elle se masque si frame.solved is None (solve désactivé)
    layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
              ObjectsLayer(), FieldsLayer(), StyleLayer(), RobotLayer(),
              ContactsLayer(),          # roadmap #3 — contact cible vs atteint (solve-gated)
              CorrespondenceLayer(),    # roadmap #4 — lignes SMPL↔G1 (solve-gated)
              SdfIsoLayer(),            # roadmap #6 — bande iso ≈ surface des SDF
              GeodesicLayer()]          # roadmap #7 — champ géodésique des canaux
    # CostDashboard seulement utile avec solve (lit solved.* sur toute la séquence)
    panels = [CostDashboard()] if solve else []
    Player(source, layers, port=port, panels=panels).run()


def main() -> None:
    """Point d'entrée CLI — parse les args de scène standard et lance run_app."""
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument("--solve", action="store_true", help="cuit le côté robot résolu (phase B)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    run_app(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames, solve=a.solve)


if __name__ == "__main__":
    main()
