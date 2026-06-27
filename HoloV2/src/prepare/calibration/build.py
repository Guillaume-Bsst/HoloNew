"""prepare/calibration — grounds the whole scene (human + objects) and characterises the subject.

Produces the per-(subject, take) ``Calibration`` (``contracts.Calibration``): a subject stature, a
human grounding offset, a shared object grounding offset and a root frame — computed offline once
and cached. ROBOT-FREE by design — the human->robot scale is a
(human, robot) quantity, owned and applied by the correspondence + transport layer (the module
whose job IS the human<->robot pairing, where both surfaces meet); calibration only supplies the
robot-free subject ``human_stature`` it composes with the robot height.

  - ``human_stature`` = the subject's rest-pose stature (m), from betas-FK (vertical extent of the
                        rest mesh). The REAL subject's size — feeds ``scale = robot_height / stature``
                        downstream. Uniform across datasets: every parametric source has a SMPL-X
                        ``BodyModel``, so there is no per-dataset stature path.
  - ``human_offset``  = world z-shift grounding the human. The ``CalibrationConfig.foot_percentile``
                        percentile (default 50 = median) of the lower mocap FOOT-JOINT height over
                        the clip. The foot joint (not the SMPL sole) is used on purpose: the SMPL
                        foot frequently penetrates BELOW the rest level during toe articulation, so
                        chasing the lowest surface point over-lifts the human; the percentile of the
                        joint targets the RESTING/contact level. Mocap demo joints => no SMPL forward
                        and one path for parametric AND positions-only sources.
  - ``object_offset`` = ONE world z-shift shared by ALL objects: grounds the lowest-reaching object
                        (the one that touches the floor) just above z=0, via a low percentile of the
                        posed object points. Shared (not per-object) so inter-object geometry is kept.
                        TODO: a finer per-object / inter-object calibration could optimise the
                        object<->object & object<->floor contacts jointly.
  - ``root_frame``    = a (4,4) framing of the root. Identity for now (provisional hook; the V1
                        kept XY raw). Generalise only when a non-trivial framing is actually needed.

Decisions (confirmed): the human->robot scale is NOT computed here (calibration stays robot-free,
cacheable per subject); the scene stays at HUMAN scale, where the interaction pipeline lives (human
cloud + object are size-consistent, and the SMPL->robot transport absorbs the size difference).
Grounding is ALWAYS recomputed (uniform): an already-grounded clip (SFU) simply yields
``human_offset ~= 0``.

Dataset-agnostic by construction: every difference between datasets lives in the DATA the loader
already produced (``smpl_params`` present -> betas-FK stature, else the default; ``joint_names`` ->
foot-joint indices for the human offset; object meshes -> shared object offset, empty list -> 0),
never in a branch here.

Ported from HoloNew: ``holosoma/preprocess.ground_to_floor`` (foot-joint grounding),
``data_loaders/omomo.omomo_height_from_betas`` (betas-FK stature).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import BodyModel, Calibration, RawMotion, SmplParams
from config_types import CalibrationConfig


# =============================================================================
# Pure functions (no I/O, no mutation) — the calibration math
# =============================================================================
def human_stature(body: BodyModel, params: SmplParams) -> float:
    """Rest-pose stature (m) = vertical extent of the subject's rest mesh, from betas-FK.

    ``rest_vertices`` is in the model's NATIVE Y-up frame, so the stature is the Y-extent (the same
    quantity the previous HoloNew measured for both the SMPL-X and the OMOMO betas paths)."""
    rv = np.asarray(body.rest_vertices(params), np.float64)   # (V, 3) native Y-up
    return float(rv[:, 1].max() - rv[:, 1].min())


def foot_floor_offset(joint_pos: np.ndarray, foot_idx: list[int], percentile: float) -> float:
    """Human floor offset = ``percentile`` of the lower mocap FOOT-JOINT height over the clip.

    Per frame, the lower of the foot joints gives the foot height; the percentile over frames targets
    the RESTING/contact level. The mocap foot joint is used (not the SMPL sole): the SMPL foot
    frequently penetrates below the rest level during toe articulation, so the lowest surface point
    sits below the floor and chasing it (a min / low percentile) over-lifts the human. ``percentile``
    = 50 is the median; higher pushes the human down (toward the planted level), lower lifts it.
    Validated visually on HODome / OMOMO / SFU. Returns the value to SUBTRACT from world z."""
    z = np.asarray(joint_pos)[:, foot_idx, 2].min(axis=1)     # (T,) lower foot per frame
    return float(np.percentile(z, percentile))


def object_floor_offset(object_verts: list[np.ndarray], object_poses: list[np.ndarray],
                        percentile: float, vert_cap: int = 4000, frame_cap: int = 150) -> float:
    """ONE shared floor offset for ALL objects = ``percentile`` of the lowest posed object point over
    the clip, pooled across objects. Grounds the object that touches the floor to ~z=0 (a low
    percentile = its "lowest reach", i.e. when it is set down); the SAME offset is then applied to
    every object so inter-object geometry is preserved. Returns the value to SUBTRACT from world z,
    or 0.0 if there are no objects.

    TODO: a finer per-object / inter-object calibration could ground each object and jointly optimise
    the object<->object and object<->floor contacts; for now one shared offset (floor-touching object
    just above the floor) is enough. ``object_verts``: list of (V,3) local; ``object_poses``: list of
    (T,7) pos-first wxyz. Verts/frames are subsampled (caps) to bound cost on dense scans."""
    if not object_verts:
        return 0.0
    rng = np.random.default_rng(0)
    lows = []
    for verts, poses in zip(object_verts, object_poses):
        v = np.asarray(verts, np.float64)
        if v.shape[0] > vert_cap:
            v = v[rng.choice(v.shape[0], vert_cap, replace=False)]
        poses = np.asarray(poses, np.float64)
        fidx = np.unique(np.linspace(0, poses.shape[0] - 1, min(poses.shape[0], frame_cap)).astype(int))
        rz = R.from_quat(poses[fidx][:, [4, 5, 6, 3]]).as_matrix()[:, 2, :]   # (F,3) world-z row, wxyz->xyzw
        world_z = np.einsum("fj,vj->fv", rz, v) + poses[fidx, 2][:, None]     # (F, V) world z of each vertex
        lows.append(world_z.min(axis=1))                                      # (F,) lowest object point/frame
    return float(np.percentile(np.concatenate(lows), percentile))


def _foot_indices(joint_names: tuple[str, ...]) -> list[int]:
    """Indices of the foot joints (case-insensitive 'foot', else 'ankle'). Raises if neither."""
    feet = [i for i, n in enumerate(joint_names) if "foot" in n.lower()]
    if not feet:
        feet = [i for i, n in enumerate(joint_names) if "ankle" in n.lower()]
    if not feet:
        raise ValueError(f"no foot/ankle joint to ground on in {joint_names!r}")
    return feet


# =============================================================================
# CalibrationBuilder — the AssetBuilder for this deliverable (build / cache)
# =============================================================================
class CalibrationBuilder:
    """``AssetBuilder`` producing the ``Calibration`` for a (subject, take). Scoped to that pair
    (NOT a geometry cache): the floor offset depends on the whole motion, the stature on the
    subject. ROBOT-FREE (no robot input -> caches per subject). The runner wraps ``build``/``load``
    in a ``prof.span("calibration")``."""

    def cache_key(self, config: CalibrationConfig, raw: RawMotion) -> str:
        """Stable hash of everything the calibration depends on: the knobs (``foot_percentile``,
        ``object_floor_pct``, ``fallback_stature``) + the demo joints (foot-joint floor offset), the
        subject shape (betas/gender -> stature), and the object meshes + poses (the shared object floor
        offset). No robot term — calibration is robot-free."""
        h = hashlib.sha1()
        h.update(f"{config.foot_percentile}|{config.object_floor_pct}|{config.fallback_stature}".encode())
        h.update(np.ascontiguousarray(raw.joint_pos, np.float32).tobytes())   # drives the foot offset
        p = raw.smpl_params
        if p is not None:                                                     # drives the stature
            h.update(str(p.gender).encode())
            h.update(np.ascontiguousarray(p.betas, np.float32).tobytes())
        for path, pose in zip(raw.object_mesh_paths, raw.object_poses_raw):   # drive the object offset
            h.update(str(path).encode())
            h.update(np.ascontiguousarray(pose, np.float32).tobytes())
        return h.hexdigest()

    def build(self, config: CalibrationConfig, raw: RawMotion,
              body: BodyModel | None = None, smpl_model_dir: Path | None = None) -> Calibration:
        """Compute the ``Calibration``. The floor offset is mocap-joint based (no body needed); the
        ``body`` is used only for the betas-FK STATURE of a parametric source — reuse it if supplied,
        else build it from ``raw.smpl_params`` + ``smpl_model_dir`` (one is required; the model dir
        lives on ``SceneSpec``, not ``RawMotion``). A positions-only clip uses the default stature."""
        if raw.is_parametric and body is None:
            if smpl_model_dir is None:
                raise ValueError("a parametric scene needs a BodyModel or smpl_model_dir to "
                                 "compute the betas-FK stature")
            from ..load.smpl import build_body_model
            body = build_body_model(raw.smpl_params, smpl_model_dir)

        # Stature: betas-FK from the real subject when parametric; the configured fallback otherwise.
        stature = human_stature(body, raw.smpl_params) if body is not None else config.fallback_stature

        # Human floor offset: a percentile of the lower foot-joint height (mocap demo joints).
        human = foot_floor_offset(raw.joint_pos, _foot_indices(raw.joint_names), config.foot_percentile)

        # Shared object floor offset: ground the lowest-reaching object just above z=0 (the floor it
        # touches), applied to ALL objects so inter-object geometry is preserved. Needs the meshes.
        if raw.object_mesh_paths:
            from ..load.mesh import load_mesh
            obj_verts = [load_mesh(p)[0] for p in raw.object_mesh_paths]
            object_offset = object_floor_offset(obj_verts, list(raw.object_poses_raw), config.object_floor_pct)
        else:
            object_offset = 0.0

        return Calibration(human_stature=stature, human_offset=human,
                           object_offset=object_offset, root_frame=np.eye(4))

    def save(self, calib: Calibration, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), human_stature=np.float64(calib.human_stature),
                 human_offset=np.float64(calib.human_offset),
                 object_offset=np.float64(calib.object_offset),
                 root_frame=np.asarray(calib.root_frame, np.float64))

    def load(self, path: Path) -> Calibration:
        d = np.load(str(path))
        return Calibration(human_stature=float(d["human_stature"]), human_offset=float(d["human_offset"]),
                           object_offset=float(d["object_offset"]),
                           root_frame=np.asarray(d["root_frame"], np.float64))


def build_calibration(raw: RawMotion, config: CalibrationConfig, body: BodyModel | None = None,
                      smpl_model_dir: Path | None = None) -> Calibration:
    """Convenience wrapper around ``CalibrationBuilder.build`` (the common call site)."""
    return CalibrationBuilder().build(config, raw, body=body, smpl_model_dir=smpl_model_dir)
