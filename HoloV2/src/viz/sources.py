"""Sources — les fournisseurs du vue-modèle. Protocole ``Source`` (``get`` / ``n_frames`` /
``context``) ; ``BakeSource`` exécute ``prepare`` une fois et cuit un ``VizFrame`` par frame affichée
(playback offline fluide) ; ``LiveSource`` (on-the-fly) est différé par la refonte.

numpy-only (pas de viser) : pilote la surface publique du pipeline (``prepare.runner`` +
``targets.pipeline.trace_frame``) et emballe le résultat dans le vue-modèle. L'espace géométrique
est fixé ICI en un seul endroit (la décision de seam de la refonte). Les imports lourds
(``prepare.runner``, ``targets.pipeline``) sont différés DANS ``__init__`` pour que
``import src.viz.sources`` reste léger (viser-free, torch-free au niveau module)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from .model import VizContext, VizFrame


@runtime_checkable
class Source(Protocol):
    """Protocole commun des fournisseurs de vue-modèle : ``get(i)`` retourne un ``VizFrame``,
    ``n_frames`` leur nombre total, ``context`` le contexte statique de la scène."""

    context: VizContext
    n_frames: int

    def get(self, i: int) -> VizFrame: ...


class BakeSource:
    """Cuit chaque ``VizFrame`` à l'avance (offline) pour un playback fluide. ``solve=False``
    laisse ``VizFrame.solved = None`` (pré-solveur) ; ``solve=True`` (``SolveTrajectory``
    + ``Evaluator`` → ``SolvedFrame``) arrive en phase B — lève ``NotImplementedError`` ici."""

    def __init__(
        self,
        spec: SceneSpec,
        config: PrepareConfig,
        *,
        solve: bool = False,
        frame_step: int = 2,
        max_frames: int = 200,
    ) -> None:
        if solve:
            raise NotImplementedError(
                "solve baking (SolvedFrame) arrives in phase B; use solve=False"
            )

        # Imports différés : ``prepare.runner`` tire torch/smplx/trimesh/pinocchio ;
        # ``targets.pipeline`` tire les ops cibles. Aucun des deux n'importe viser.
        # Différés ici (pas au niveau module) pour que l'import de ce module reste léger.
        from ..prepare.runner import prepare
        from ..targets.pipeline import trace_frame

        grounded, ctx = prepare(spec, config)
        body = grounded.body
        if body is None:
            raise ValueError(
                "the bake source needs a parametric body (SMPL params)"
            )

        # Sélectionner les frames à afficher : pas de frame_step=0
        shown = list(range(0, grounded.n_frames, max(1, frame_step)))[:max_frames]

        # FrameTrace par frame — même chemin que trace_frame (intermédiaires gardés)
        traces = [trace_frame(grounded, ctx, spec.robot, f) for f in shown]

        # Sommets SMPL posés pour le rendu mesh du sujet (nuage est déjà dans la trace)
        verts = [
            body.posed_vertices(grounded.smpl_params, f).astype(np.float32)
            for f in shown
        ]

        # Contexte statique de la scène — construit UNE FOIS depuis grounded/ctx
        self._context = VizContext(
            channel_names=ctx.channel_names,
            margin=float(ctx.margin),
            style_link_names=traces[0].targets.style.link_names,
            smpl_faces=np.asarray(body.faces),
            smpl_parents=np.asarray(body.parents),  # type: ignore[attr-defined]
            n_objects=grounded.n_objects,
            robot_urdf_path=spec.robot.urdf_path,
            has_solve=False,
            ground_sdf=ctx.channels[0].sdf,
        )

        # Frames cuits : VizFrame gelé par frame affichée
        self._frames: list[VizFrame] = [
            VizFrame(
                pose=tr.pose,
                smpl_verts_world=v,
                human_cloud_world=np.asarray(tr.human_cloud_world, np.float32),
                object_clouds_world=tuple(
                    np.asarray(oc, np.float32) for oc in tr.object_clouds_world
                ),
                human_field=tr.human_field,
                targets=tr.targets,
                solved=None,
            )
            for tr, v in zip(traces, verts)
        ]

    @property
    def context(self) -> VizContext:
        """Contexte statique de la scène (remis à chaque couche lors du setup)."""
        return self._context

    @property
    def n_frames(self) -> int:
        """Nombre de frames cuits disponibles via ``get``."""
        return len(self._frames)

    def get(self, i: int) -> VizFrame:
        """Retourne le ``VizFrame`` cuit à l'index ``i``."""
        return self._frames[int(i)]


class LiveSource:
    """Différé par la refonte viz ("LiveSource … plus tard") : exécuterait ``trace_frame``
    (+ solve) à la volée derrière la même interface ``Source`` pour la téléopération live.
    Non construit dans cette phase — lève ``NotImplementedError``."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "LiveSource is deferred by the viz redesign; use BakeSource"
        )
