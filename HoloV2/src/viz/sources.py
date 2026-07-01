"""Sources — les fournisseurs du vue-modèle. Protocole ``Source`` (``get`` / ``n_frames`` /
``context``) ; ``BakeSource`` exécute ``prepare`` une fois et cuit un ``VizFrame`` par frame affichée
(playback offline fluide) ; ``LiveSource`` (on-the-fly) est différé par la refonte.

numpy-only (pas de viser) : pilote la surface publique du pipeline (``prepare.runner`` +
``targets.pipeline.trace_frame``) et emballe le résultat dans le vue-modèle. L'espace géométrique
est fixé ICI en un seul endroit (la décision de seam de la refonte). Les imports lourds
(``prepare.runner``, ``targets.pipeline``, ``solve.runner``) sont différés DANS ``__init__`` pour que
``import src.viz.sources`` reste léger (viser-free, torch-free au niveau module).

Phase B — ``build_solved_frame`` : construit un ``SolvedFrame`` pur (numpy-only) à partir
d'une ``SolveTrajectory`` + d'un ``Evaluator`` déjà construit ; câblé dans ``BakeSource(solve=True)``."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation as _R

from ..prepare.config import PrepareConfig
from ..prepare.contracts import InteractionContext, SceneSpec
from ..solve.contracts import SolveTrajectory
from ..targets import Evaluator, pose_cloud
from .model import SolvedFrame, VizContext, VizFrame


def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions wxyz -> (N, 3, 3) rotations. Réordonne wxyz -> xyzw pour scipy.
    Identique à la conversion interne de ``solve.loop.evaluate`` (qu'on NE importe pas — interne solve)."""
    q = np.asarray(quat_wxyz, np.float64).reshape(-1, 4)
    if q.shape[0] == 0:
        return np.zeros((0, 3, 3), np.float64)
    return _R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def build_solved_frame(traj: SolveTrajectory, ev: Evaluator, ctx: InteractionContext,
                       f: int) -> SolvedFrame:
    """Construit le ``SolvedFrame`` post-solve pour la frame ``f``. PUR : lit seulement la
    ``SolveTrajectory`` et RÉUTILISE le ``targets.Evaluator`` (+ FK ``ctx.robot``) — aucune nouvelle
    logique de retargeting.

    - ``q`` / ``object_poses`` / diagnostics de coût : directement depuis ``traj``.
    - ``style_achieved`` = ``ev.style(q)`` (le recompute prouvé, cf. test_solve_runner).
    - ``contact_achieved`` = ``ev.contacts(q, object_rot, object_pos)`` (object_poses wxyz → matrices).
    - ``robot_points_world`` : les M points de contrôle placés par FK @ q (``pose_cloud(robot_cloud, R, t)``).
    - ``link_transforms`` : (L, 4, 4) placements homogènes monde des liens robot @ q."""
    q = np.asarray(traj.qpos[f], np.float64)                       # (nq,)
    poses = np.asarray(traj.object_poses[f], np.float64)           # (N, 7) pos + quat wxyz
    info = traj.info[f]                                             # FrameInfo

    # « atteint » — RÉUTILISE l'Evaluator (pas de recompute retargeting).
    style_achieved = ev.style(q)                                    # StyleEval
    object_pos = poses[:, :3]                                       # (N, 3)
    object_rot = _quat_wxyz_to_R(poses[:, 3:7])                    # (N, 3, 3)
    contact_achieved = ev.contacts(q, object_rot, object_pos)      # ContactEval

    # FK : transforms des liens (L, 4, 4) + points de contrôle robot (M, 3).
    rot, pos = ctx.robot.link_transforms(q)                        # (L, 3, 3), (L, 3)
    L = rot.shape[0]
    link_tf = np.zeros((L, 4, 4), np.float64)
    link_tf[:, :3, :3] = rot
    link_tf[:, :3, 3] = pos
    link_tf[:, 3, 3] = 1.0
    robot_points_world = pose_cloud(ctx.robot_cloud, rot, pos)     # (M, 3)

    return SolvedFrame(
        q=q,
        object_poses=poses,
        robot_points_world=np.asarray(robot_points_world, np.float64),
        link_transforms=link_tf,
        style_achieved=style_achieved,
        contact_achieved=contact_achieved,
        cost=float(info.cost),
        cost_by_term=dict(info.cost_by_term),
        n_iters=int(info.n_iters),
        status=str(info.status),
    )


@runtime_checkable
class Source(Protocol):
    """Protocole commun des fournisseurs de vue-modèle : ``get(i)`` retourne un ``VizFrame``,
    ``n_frames`` leur nombre total, ``context`` le contexte statique de la scène."""

    context: VizContext
    n_frames: int

    def get(self, i: int) -> VizFrame: ...


class BakeSource:
    """Cuit chaque ``VizFrame`` à l'avance (offline) pour un playback fluide. ``solve=False``
    laisse ``VizFrame.solved = None`` (pré-solveur) ; ``solve=True`` exécute le solveur SQP 1× et
    remplit ``VizFrame.solved`` (``SolvedFrame``) par frame — câblé en phase B.

    ``solve_config`` : ``SolveConfig`` optionnelle (défaut = ``SolveConfig()`` = hyperparamètres
    standard). Les imports lourds (``prepare.runner``, ``targets.pipeline``, ``solve.runner``,
    ``cvxpy``) sont tous différés DANS ``__init__`` ; l'import du module reste léger (viser-free)."""

    def __init__(
        self,
        spec: SceneSpec,
        config: PrepareConfig,
        *,
        solve: bool = False,
        solve_config=None,
        frame_step: int = 2,
        max_frames: int = 200,
    ) -> None:
        # Imports différés : ``prepare.runner`` tire torch/smplx/trimesh/pinocchio ;
        # ``targets.pipeline`` tire les ops cibles. Aucun des deux n'importe viser.
        # Différés ici (pas au niveau module) pour que l'import de ce module reste léger.
        from ..prepare.runner import prepare, load_object_meshes
        from ..targets.pipeline import trace_frame

        grounded, ctx = prepare(spec, config)
        # Géométrie mesh des objets — point d'entrée PUBLIC offline/viz (le solve n'en a pas besoin).
        object_meshes = load_object_meshes(grounded)
        body = grounded.body
        if body is None:
            raise ValueError(
                "the bake source needs a parametric body (SMPL params)"
            )

        # Sélectionner les frames à afficher : pas de frame_step=0
        shown = list(range(0, grounded.n_frames, max(1, frame_step)))[:max_frames]
        n = len(shown)

        # FrameTrace par frame — même chemin que trace_frame (intermédiaires gardés)
        traces = [trace_frame(grounded, ctx, spec.robot, f) for f in shown]

        # Sommets SMPL posés pour le rendu mesh du sujet (nuage est déjà dans la trace)
        verts = [
            body.posed_vertices(grounded.smpl_params, f).astype(np.float32)
            for f in shown
        ]

        # Contexte statique de la scène — construit UNE FOIS depuis grounded/ctx.
        # channels + correspondence : simple ré-exposition depuis l'InteractionContext (pas de
        # recompute — règle d'or : consommateur pur). Channel/CorrespondenceTable sont numpy-only,
        # donc ce module reste viser/torch-free.
        self._context = VizContext(
            channel_names=ctx.channel_names,
            margin=float(ctx.margin),
            style_link_names=traces[0].targets.style.link_names,
            smpl_faces=np.asarray(body.faces),
            smpl_parents=np.asarray(body.parents),  # type: ignore[attr-defined]
            n_objects=grounded.n_objects,
            channels=ctx.channels,                 # Channel complets (sdf + geodesic + object_idx)
            correspondence=ctx.correspondence,     # appariement SMPL<->robot statique
            robot_urdf_path=spec.robot.urdf_path,
            has_solve=solve,
            ground_sdf=ctx.channels[0].sdf,
            object_meshes=object_meshes,           # mesh source translucide (ObjectsLayer)
        )

        # Solve optionnel (phase B) : exécute le solveur SQP sur les frames cuits.
        # Imports lazy (solve.runner tire cvxpy qui est lourd) — restent locaux à ce bloc.
        self._solved: list = [None] * n
        if solve:
            from ..solve.runner import solve as _solve_seq   # lazy : cvxpy n'arrive qu'ici
            from ..solve.config import SolveConfig

            # Réutilise les FrameTargets DÉJÀ calculés par le bake trace (pas de recompute)
            frame_targets = [traces[i].targets for i in range(n)]
            traj = _solve_seq(grounded, ctx, frame_targets,
                              solve_config or SolveConfig(), robot_name=spec.robot.name)
            ev = Evaluator(ctx, spec.robot.name)
            self._solved = [build_solved_frame(traj, ev, ctx, i) for i in range(n)]

        # Frames cuits : VizFrame gelé par frame affichée (solved=None si solve=False)
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
                solved=self._solved[i],
            )
            for i, (tr, v) in enumerate(zip(traces, verts))
        ]

    @property
    def context(self) -> VizContext:
        """Contexte statique de la scène (remis à chaque couche lors du setup)."""
        return self._context

    @property
    def n_frames(self) -> int:
        """Nombre de frames cuits disponibles via ``get``."""
        return len(self._frames)

    @property
    def frames(self) -> list[VizFrame]:
        """Séquence complète des frames cuits — utilisée par les panels qui lisent toute la
        séquence (ex. CostDashboard agrège ``solved.cost_by_term`` sur tous les frames)."""
        return list(self._frames)

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
