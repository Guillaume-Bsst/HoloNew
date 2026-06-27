"""OMOMO loader (InterMimic ``.pt`` motion + OMOMO subject pickle) -> RawMotion.

OMOMO object-manipulation sequences come from two places:
  - the InterMimic ``.pt`` (``spec.motion_path``): a ``(T, 591)`` state tensor carrying per-joint
    SMPL-H joint positions and global orientations (MuJoCo order, Z-up) plus the object's world
    pose;
  - the original OMOMO release (``spec.dataset_root``): joblib pickles keyed by sequence name with
    the subject ``betas``/``gender`` and per-frame ``obj_scale``, plus the captured unit object
    meshes under ``data/captured_objects/``.

Simplification over the previous HoloNew: the body is **SMPL-X** (16 betas), so we reuse the
SMPL-X ``BodyModel`` directly -- the .pt's 52 body+hand joints map cleanly onto SMPL-X's
``body_pose`` + ``left/right_hand_pose`` (the 3 face joints stay at zero). No SMPL-H body model
is needed. The .pt stores GLOBAL orientations (like SFU), already Z-up (PHC/IsaacGym), so we
reconstruct the local ``SmplParams`` with the shared ``local_rotvecs_from_global`` and slice
body/hands out of the SMPL-H 52-joint layout; ``BodyModel`` re-applies Q (Y->Z) to those params.
Scale/grounding is NOT done here -- it belongs to ``prepare/calibration/`` (as for SFU/HODome).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..smpl import SMPLX_BODY_JOINTS, local_rotvecs_from_global, rest_body_model

_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_omomo_meshes"

# InterMimic .pt column layout (the stored per-frame state vector).
_PT_JOINTS = slice(162, 162 + 52 * 3)   # (52,3) joint positions, MuJoCo order, Z-up
_PT_OBJECT = slice(318, 325)            # [tx,ty,tz, qx,qy,qz,qw] object world pose
_PT_QUATS = slice(383, 383 + 52 * 4)    # (52,4) per-joint global orientations, MuJoCo order, xyzw

# Right-multiplier undoing InterMimic's PHC "upright_start" twist baked into the stored global
# quats (interact2mimic writes global_rot * Q^-1, xyzw); applying it yields true SMPL-X globals.
_UPRIGHT_FIX_XYZW = np.array([0.5, 0.5, 0.5, 0.5])

# smpl_2_mujoco (from InterAct): SMPL_2_MUJOCO[mujoco_idx] = smpl_idx. The .pt arrays are MuJoCo
# order; scattering by this index puts each joint into its SMPL-H 52-joint slot (body 0-21, left
# hand 22-36, right hand 37-51).
_SMPL_2_MUJOCO = np.array([
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

# SMPL-H 52-joint parent tree (body 0-21 identical to SMPL-X; left hand 22-36 / right hand 37-51
# are the MANO sub-tree off the wrists 20/21). Turns the scattered globals into parent-relative
# locals; the 15 hand locals are layout-invariant, so they feed SMPL-X's left/right_hand_pose
# (SMPL-X hand slots 25-39 / 40-54) unchanged. Validated to <1e-6 m against the .pt joints.
_SMPLH_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
     20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
     21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50], dtype=np.int64)


def _object_token(seq: str) -> str:
    """Object name = 2nd '_'-segment of the sequence (sub16_largetable_008 -> largetable)."""
    parts = seq.split("_")
    return parts[1] if len(parts) >= 2 else seq


def _load_pt(path: Path):
    """Read an InterMimic .pt -> (joints (T,52,3) MuJoCo Z-up, object_pose (T,7) pos-first wxyz,
    quats (T,52,4) MuJoCo wxyz with the upright-start twist undone)."""
    import torch

    d = torch.load(str(path), map_location="cpu").detach().numpy()
    joints = d[:, _PT_JOINTS].reshape(-1, 52, 3).astype(np.float64)
    obj = d[:, _PT_OBJECT]                                       # [tx,ty,tz, qx,qy,qz,qw]
    object_pose = obj[:, [0, 1, 2, 6, 3, 4, 5]].astype(np.float32)   # -> pos-first wxyz
    q_xyzw = d[:, _PT_QUATS].reshape(-1, 52, 4)
    t = q_xyzw.shape[0]
    fixed = (R.from_quat(q_xyzw.reshape(-1, 4)) * R.from_quat(_UPRIGHT_FIX_XYZW)).as_quat()
    quats_wxyz = fixed.reshape(t, 52, 4)[:, :, [3, 0, 1, 2]]
    return joints, object_pose, quats_wxyz


def _subject_meta(dataset_root: Path | None, seq: str):
    """``(betas (16,)|None, gender, obj_scale|None)`` for ``seq`` from the OMOMO pickle (train then
    test). Degrades to ``(None, "neutral", None)`` when the release / sequence is absent -> neutral
    mean shape, native object size."""
    if dataset_root is None:
        return None, "neutral", None
    import joblib

    for split in ("train", "test"):
        p = Path(dataset_root) / "data" / f"{split}_diffusion_manip_seq_joints24.p"
        if not p.exists():
            continue
        data = joblib.load(str(p))
        for entry in data.values():
            if str(entry.get("seq_name", "")) == seq:
                betas = np.asarray(entry["betas"], np.float32).reshape(-1)
                gender = str(entry.get("gender", "neutral"))
                scale = float(np.asarray(entry["obj_scale"]).mean()) if "obj_scale" in entry else None
                return betas, gender, scale
    return None, "neutral", None


def _object_mesh(token: str, dataset_root: Path | None, scale: float | None, cache_dir: Path):
    """Captured unit mesh -> recentred + per-sequence-scaled mesh in the frame the .pt poses
    expect (centroid at the origin), cached. Returns None when the mesh or its scale is missing
    (then the sequence degrades to body/ground-only, as the V1 builder does)."""
    if dataset_root is None or scale is None:
        return None
    import trimesh

    captured = Path(dataset_root) / "data" / "captured_objects" / f"{token}_cleaned_simplified.obj"
    if not captured.exists():
        return None
    out = cache_dir / f"{token}_{scale:.6f}.obj"
    if out.exists():
        return out
    m = trimesh.load(str(captured), force="mesh", process=False)
    v = np.asarray(m.vertices, np.float64)
    v = (v - v.mean(0)) * scale
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False).export(str(out))
    return out


@register_loader("omomo")
class OmomoLoader:
    """SceneSpec -> RawMotion for an OMOMO/InterMimic manipulation sequence (one object)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("OMOMO needs spec.smpl_model_dir (the SMPL-X model directory)")
        seq = Path(spec.motion_path).stem
        joints_mj, object_pose, quats_mj = _load_pt(Path(spec.motion_path))
        T = joints_mj.shape[0]

        # MuJoCo -> SMPL-H 52-joint order (body 0-21, left hand 22-36, right hand 37-51).
        joints = np.empty_like(joints_mj); joints[:, _SMPL_2_MUJOCO] = joints_mj
        quats = np.empty_like(quats_mj);   quats[:, _SMPL_2_MUJOCO] = quats_mj

        betas, gender, scale = _subject_meta(spec.dataset_root, seq)
        if betas is None:
            betas = np.zeros(16, np.float32)                        # neutral mean shape fallback
        rest = rest_body_model(betas, gender, Path(spec.smpl_model_dir))

        local, transl = local_rotvecs_from_global(
            quats, joints[:, 0], _SMPLH_PARENTS, rest.rest_joints[0])
        params = SmplParams(
            betas=betas, global_orient=local[:, 0],
            body_pose=local[:, 1:22].reshape(T, -1),
            left_hand_pose=local[:, 22:37].reshape(T, -1),
            right_hand_pose=local[:, 37:52].reshape(T, -1),
            transl=transl, gender=gender, model_type="smplx")

        cache_dir = Path(spec.cache_dir) / "omomo_meshes" if spec.cache_dir else _MESH_CACHE
        mesh = spec.object_mesh_paths[0] if spec.object_mesh_paths else \
            _object_mesh(_object_token(seq), spec.dataset_root, scale, cache_dir)
        object_poses = (object_pose,) if mesh is not None else ()
        object_meshes = (Path(mesh),) if mesh is not None else ()

        return RawMotion(
            joint_pos=joints[:, :len(SMPLX_BODY_JOINTS)].astype(np.float32), joint_names=SMPLX_BODY_JOINTS,
            fps=30.0, source_format="omomo", object_poses_raw=object_poses,
            object_mesh_paths=object_meshes, smpl_params=params)
