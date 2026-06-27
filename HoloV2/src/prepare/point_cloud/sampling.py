"""The shared surface-sampling identity that binds the human cloud to the correspondence.

``CorrespondenceTable.smpl_idx`` indexes the human cloud's point ORDER. That correspondence is
built once on a NEUTRAL template body, while the runtime cloud is the SUBJECT's; for ``smpl_idx`` to
stay valid both must share the exact same ``(tri_idx, bary)`` sampling. So the sampling is the
canonical identity: the correspondence carries it (embedded in ``corr_neutral.npz``), and the human
cloud REUSES it rather than resampling — it only recomputes the per-subject skinning on top.

``SurfaceSampling`` is a build-only intermediate (it never reaches ``targets``/``solve``), so it
lives here, local to ``point_cloud/``, not in the ``contracts/`` package. ``sampling_id`` is the stable hash
stamped onto both the cloud (``PointCloud.sampling_id``) and the table
(``CorrespondenceTable.smpl_sampling_id``); the runner asserts they match.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceSampling:
    """A fixed set of surface samples as barycentric locations on a mesh's triangles, ORDER-stable.
    Topology-bound (``tri_idx`` references the SMPL faces), so it is betas-independent: the same
    ``(tri_idx, bary)`` names a valid point on any SMPL-X mesh of the same topology."""

    tri_idx: np.ndarray   # (N,)    triangle index per sample
    bary: np.ndarray      # (N, 3)  barycentric weights, rows sum to 1
    sampling_id: str      # stable identity (see ``sampling_id``)

    @property
    def n_points(self) -> int:
        return self.tri_idx.shape[0]


def sampling_id(tri_idx: np.ndarray, bary: np.ndarray) -> str:
    """Stable hash of a ``(tri_idx, bary)`` sampling — the binding key between cloud and table.

    Depends only on the sampled locations (triangle + barycentric weights), so it is identical for
    the neutral template and every subject that reuses the same sampling, and changes whenever the
    sampling does. Hashed in canonical dtypes so the value is reproducible across machines."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(tri_idx, np.int64).tobytes())
    h.update(np.ascontiguousarray(bary, np.float32).tobytes())
    return h.hexdigest()[:16]
