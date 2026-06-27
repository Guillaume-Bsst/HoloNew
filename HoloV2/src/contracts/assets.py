"""Build-once geometry assets for the interaction treatment: surface clouds, the SMPL<->robot
correspondence, and the assembled ``InteractionContext`` that bundles them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    """Surface samples carrying their own SPARSE SKINNING, posed from part transforms alone
    (mesh-free, torch-free), uniformly for every part kind:
      - object: K=1, weight 1, part = the rigid body.
      - robot : K=1, weight 1, part = the link (posed by FK).
      - human : K~4, LBS-on-cloud blend over the dominant SMPL bones (closes joint creases).

    Posing one frame, given each part's world transform ``T[j] = (R_j, t_j)``:
        p_world[i] = sum_k weights[i,k] * (R[parts[i,k]] @ offsets[i,k] + t[parts[i,k]])
    ``offsets`` are in each part's REST-local frame (skinning baked once offline)."""

    parts: np.ndarray     # (P, K) int    part/bone index per influence
    weights: np.ndarray   # (P, K) float  rows sum to 1 (K=1 => rigid)
    offsets: np.ndarray   # (P, K, 3)     point in part k's rest-local frame
    sampling_id: str = "" # identity of the sampling (density/seed/topology) — binds to the
                          # correspondence built against it (see CorrespondenceTable)

    @property
    def n_points(self) -> int:
        return self.parts.shape[0]

    @property
    def n_influences(self) -> int:
        return self.parts.shape[1]


@dataclass(frozen=True)
class CorrespondenceTable:
    """Fixed SMPL <-> robot surface correspondence (built once by optimal transport, OT).

    Pairs M points: human side (``smpl_idx`` into the SMPL cloud) and robot side
    (``link_idx`` + ``offset_local`` in that link's frame). Transport copies the human field
    at ``smpl_idx[m]`` onto robot point m. VALID ONLY for the SMPL cloud whose
    ``sampling_id == smpl_sampling_id`` (assert at assembly)."""

    smpl_idx: np.ndarray         # (M,) index into the SMPL PointCloud's point order
    link_idx: np.ndarray         # (M,) robot link index (into link_names)
    offset_local: np.ndarray     # (M, 3) robot point in that link's frame
    link_names: tuple[str, ...]  # (L,)
    smpl_sampling_id: str = ""   # the human-cloud sampling this was built against

    @property
    def n_points(self) -> int:
        return self.smpl_idx.shape[0]


@dataclass(frozen=True)
class InteractionContext:
    """All build-once assets for the interaction treatment, passed explicitly (no globals).

    Invariants (checked at assembly):
    - ``channels[0]`` is the GROUND (static; a plane SDF by default, or a terrain SDF);
      the rest are object channels with ``object_idx`` aligned to ``object_clouds`` and the
      scene's object order.
    - ``human_cloud.sampling_id == correspondence.smpl_sampling_id``."""

    channels: tuple[Channel, ...]          # ground (static) + one per object
    human_cloud: PointCloud                # on the SMPL surface
    object_clouds: tuple[PointCloud, ...]  # one per object (object_clouds[i] <-> channel object_idx=i)
    correspondence: CorrespondenceTable    # SMPL -> robot (STATIC binding)
    margin: float                          # field activation margin (m)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)
