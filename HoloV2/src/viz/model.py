"""Contrats vue-modèle DÉTENUS par viz — la jointure entre la Source et les couches. Gelés, numpy-only
(pas de viser, pas d'import torch): importables/testables sans écran. ``VizContext`` = ressources
statiques par scène (remises à chaque couche lors de setup); ``VizFrame`` = un frame montré
(Source.get); ``SolvedFrame`` = le bundle post-résolution (None jusqu'à résolu). Les couches lisent
SEULEMENT ces contrats — jamais les contrats du pipeline.

``ground_sdf`` étend le design des 8 champs VizContext pour que la couche ``ground`` rend le vrai
canal ground (plan/terrain) au lieu d'une boîte plate codée en dur (migration design: "sol box plat
-> couche ground lit le SDF"). ``SDF`` est un contrat prepare numpy-only, donc ce module reste
viser/torch-free."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..prepare.contracts import Channel, CorrespondenceTable, ObjectMesh, SDF
from ..targets.contracts import (ContactEval, FramePose, FrameTargets, MultiChannelField, StyleEval)


@dataclass(frozen=True)
class VizContext:
    """Ressources statiques par scène remises à chaque couche lors de ``setup``."""

    channel_names: tuple[str, ...]       # ground + N objets, dans l'ordre d'éval
    margin: float                        # bande d'activation du champ (m) — échelle heatmap distance
    style_link_names: tuple[str, ...]    # les L liens de style suivis (ordre StyleTargets)
    smpl_faces: np.ndarray               # (F, 3) int — topologie mesh SMPL (couche ghost)
    smpl_parents: np.ndarray             # (J,) int — parents d'os SMPL (couche skeleton)
    n_objects: int
    # --- assets statiques d'interaction (consommés par les couches contacts/correspondence/sdf_iso/
    # geodesic) : on porte les Channel COMPLETS (chaque Channel tient son SDF, sa GeodesicTable et son
    # object_idx -> binding de pose), plus l'appariement SMPL<->robot. channel_names en reste dérivable.
    channels: tuple[Channel, ...]           # ground (object_idx=None) + un par objet, ordre prepare
    correspondence: CorrespondenceTable     # appariement statique SMPL<->robot (smpl_idx / link_idx)
    robot_urdf_path: Path                # G1 URDF (couche robot, phase B)
    has_solve: bool                      # True une fois que la Source cuit solve -> SolvedFrame
    ground_sdf: SDF                      # SDF du canal ground (channels[0]) — surface couche ground
    # Géométrie MESH des objets (consommée par ObjectsLayer pour le mesh source translucide). Défaut
    # ``()`` : champ purement OFFLINE/viz, rétro-compatible (les constructions sans mesh restent valides,
    # et une source non-BakeSource peut l'omettre). Chargé à la demande via ``prepare.runner.load_object_meshes``.
    object_meshes: tuple[ObjectMesh, ...] = ()

    def __post_init__(self) -> None:
        if self.smpl_faces.ndim != 2 or self.smpl_faces.shape[1] != 3:
            raise ValueError(f"smpl_faces must be (F, 3), got {self.smpl_faces.shape}")
        if self.smpl_parents.ndim != 1:
            raise ValueError(f"smpl_parents must be (J,), got {self.smpl_parents.shape}")
        if len(self.channel_names) != self.n_objects + 1:
            raise ValueError(
                f"channel_names ({len(self.channel_names)}) must be n_objects+1 ({self.n_objects + 1})")


@dataclass(frozen=True)
class SolvedFrame:
    """Bundle post-résolution pour UN frame (construit par ``BakeSource`` à partir de
    ``SolveTrajectory`` + ``targets.Evaluator``). Ensemble complet de champs défini maintenant;
    PHASE A laisse ``VizFrame.solved = None`` (BakeSource le remplit en phase B). Il LIT
    ``SolveTrajectory`` et RÉUTILISE les 'achieved' de l'Evaluator — aucune logique retargeting
    nouvelle."""

    q: np.ndarray                    # (nq,)     config robot résolu (SolveTrajectory.qpos[f])
    object_poses: np.ndarray         # (N, 7)    poses objets résolues (SolveTrajectory.object_poses[f])
    robot_points_world: np.ndarray   # (M, 3)    points de correspondance placés par FK robot @ q
    link_transforms: np.ndarray      # (L, 4, 4) placements des liens (FK) — couches correspondance/contact
    style_achieved: "StyleEval | None"      # ev.style(q)
    contact_achieved: "ContactEval | None"  # ev.contacts(q, object_rot, object_pos)
    cost: float
    cost_by_term: dict               # {nom_terme: résidu²} (FrameInfo)
    n_iters: int
    status: str


@dataclass(frozen=True)
class VizFrame:
    """Un frame montré (``Source.get(i)``) — gelé, numpy-only. L'espace géométrique est fixé UNE
    FOIS par la Source (échelle scène quand solve est présent); les couches lisent ces arrays world
    tels quels."""

    pose: FramePose                              # os + objet (R, t)           (de FrameTrace)
    smpl_verts_world: np.ndarray | None          # (V, 3) f32 mesh SMPL posé (None si non-paramétrique)
    human_cloud_world: np.ndarray                # (N, 3) f32 nuage humain posé
    object_clouds_world: tuple[np.ndarray, ...]  # par objet, (P_i, 3) f32
    human_field: MultiChannelField               # champ sur le nuage humain (PRE-transport)
    targets: FrameTargets                        # références style + robot + env
    solved: "SolvedFrame | None"                 # None tant que non résolu -> couches solve no-op

    def __post_init__(self) -> None:
        if self.human_cloud_world.ndim != 2 or self.human_cloud_world.shape[1] != 3:
            raise ValueError(f"human_cloud_world must be (N, 3), got {self.human_cloud_world.shape}")
        if self.smpl_verts_world is not None and (
                self.smpl_verts_world.ndim != 2 or self.smpl_verts_world.shape[1] != 3):
            raise ValueError(f"smpl_verts_world must be (V, 3)|None, got {self.smpl_verts_world.shape}")
