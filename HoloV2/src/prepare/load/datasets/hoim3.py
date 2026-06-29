"""HOI-M3 loader (EasyMocap SMPL, multi-person / multi-object) -> RawMotion (as SMPL-X).

Each sequence is a pair under ``mocap_ground/``:
    <seq>_human.npz   smpl_params[frame] = list of persons {poses(69), shapes(10), Rh(3), Th(3),
                      id}  (EasyMocap SMPL) + model/gender/mocap_frame_rate
    <seq>_object.npz  object_params[frame] = {object_name: {object_R(3,3), object_T(1,3)}}
Object meshes live at ``scanned_object/<obj>/<obj>_simplified_transformed.obj`` (the centred frame
the R/T poses expect). The capture is Y-up @ 60fps. ``spec.motion_path`` points at the human npz.

We retarget ONE person (the pipeline is single-human) and keep ALL objects. To stay homogeneous
with the other loaders (everything SMPL-X) the SMPL params are converted to SMPL-X:
  - SHAPE: transferred once via ``smpl_betas_to_smplx`` (deformation transfer; shape is constant);
  - POSE: remaps kinematically -- SMPL body joints 1-21 == SMPL-X body joints 1-21, so
    ``body_pose = poses[:, :63]``; SMPL's two single hand joints are dropped (flat SMPL-X hands);
  - PLACEMENT: EasyMocap rotates the canonical body about the origin then adds Th, i.e.
    ``v_world = R(Rh) @ v_canonical + Th``. SMPL-X rotates about the pelvis, so
    ``global_orient = Rh`` and ``transl = Th + (R(Rh) - I) @ J0`` reproduce the same world pose.
``BodyModel`` then applies Q (Y->Z); object R/T are rotated to the Z-up world the same way.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..frames import object_pose_zup
from ..smpl import SMPLX_BODY_JOINTS, rest_body_model
from ..smpl2smplx import smpl_betas_to_smplx, smpl_rest_pelvis

_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_hoim3_meshes"


def _centered_mesh(src: Path, cache_dir: Path) -> Path:
    """The object_R/T poses are calibrated against the mesh CENTRED on its vertex mean (the HOI-M3
    toolbox does ``vertices -= vertices.mean(0)`` before posing). Centre + cache, return the path."""
    import trimesh

    out = cache_dir / src.name
    if out.exists():
        return out
    m = trimesh.load(str(src), force="mesh", process=False, skip_materials=True)
    v = np.asarray(m.vertices, np.float64)
    v = v - v.mean(0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False).export(str(out))
    return out


def _resolve_assets(smplx_dir: Path, gender: str, smplh_dir: Path | None = None,
                    deftrafo_pkl: Path | None = None):
    """Locate the SMPL-X npz, the SMPL-H npz (SMPL body template) and the SMPL->SMPL-X deftrafo.

    Explicit overrides (from paths.toml, threaded via SceneSpec) win; otherwise derive by
    convention from the SMPL-X model dir (``.../models/<release>/models/smplx``)."""
    smplx_npz = smplx_dir / f"SMPLX_{gender.upper()}.npz"
    models_root = smplx_dir.parents[2]                      # .../models
    smplh_root = Path(smplh_dir) if smplh_dir is not None else models_root / "smplh"
    smplh_npz = smplh_root / gender / "model.npz"
    deftrafo = (Path(deftrafo_pkl) if deftrafo_pkl is not None
                else models_root / "model_transfer" / "smpl2smplx_deftrafo_setup.pkl")
    for p, what in ((smplx_npz, "SMPL-X npz"), (smplh_npz, "SMPL-H npz (SMPL body template)"),
                    (deftrafo, "SMPL->SMPL-X deftrafo (smpl2smplx_deftrafo_setup.pkl)")):
        if not p.exists():
            raise FileNotFoundError(f"HOI-M3 needs the {what} at {p}")
    return smplx_npz, smplh_npz, deftrafo


def _person_series(smpl_params: np.ndarray, target_id: int):
    """``(Rh (T,3), Th (T,3), poses (T,69), betas10)`` for ``target_id``, holding the last seen
    frame on a dropout (multi-person mocap can briefly lose a subject)."""
    T = len(smpl_params)
    Rh, Th, poses = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros((T, 69))
    betas, last = None, None
    for f in range(T):
        ent = next((p for p in smpl_params[f] if int(np.asarray(p["id"])) == target_id), None)
        ent = last if ent is None else ent
        if ent is None:
            continue                                        # leading dropout: filled once seen
        last = ent
        Rh[f] = np.asarray(ent["Rh"], np.float64).reshape(3)
        Th[f] = np.asarray(ent["Th"], np.float64).reshape(3)
        poses[f] = np.asarray(ent["poses"], np.float64).reshape(69)
        if betas is None:
            betas = np.asarray(ent["shapes"], np.float64).reshape(-1)[:10]
    if betas is None:
        raise ValueError(f"person id {target_id} not present in the sequence")
    return Rh, Th, poses, betas


def _objects(object_npz: Path, scanned_dir: Path, cache_dir: Path, override: tuple[Path, ...],
             keep: tuple[str, ...] | None):
    """Objects -> (poses tuple of (T,7), mesh-path tuple). ``keep`` selects a subset by name
    (None => all). Meshes are centred (the frame the poses expect); objects whose mesh is absent
    are dropped; the last seen pose is held on a per-frame dropout."""
    op = np.load(str(object_npz), allow_pickle=True)["object_params"]
    T = len(op)
    names = list(op[0].keys())
    if keep is not None:
        unknown = [n for n in keep if n not in names]
        if unknown:
            raise ValueError(f"object_names {unknown} not in the scene; available: {names}")
        names = [n for n in names if n in keep]
    poses: list[np.ndarray] = []
    meshes: list[Path] = []
    for i, name in enumerate(names):
        if override:
            mesh = override[i] if i < len(override) else None
        else:
            src = scanned_dir / name / f"{name}_simplified_transformed.obj"
            mesh = _centered_mesh(src, cache_dir) if src.exists() else None
        if mesh is None or not Path(mesh).exists():
            continue
        Rs, Ts, last = np.zeros((T, 3, 3)), np.zeros((T, 3)), None
        for f in range(T):
            v = op[f].get(name, last)
            if v is None:
                continue
            last = v
            Rs[f] = np.asarray(v["object_R"], np.float64)
            Ts[f] = np.asarray(v["object_T"], np.float64).reshape(3)
        poses.append(object_pose_zup(Rs, Ts))
        meshes.append(Path(mesh))
    return tuple(poses), tuple(meshes)


def build_person_params(smpl_params: np.ndarray, target_id: int, gender: str, smplx_dir: Path,
                        smplh_dir: Path | None = None, deftrafo_pkl: Path | None = None):
    """EasyMocap SMPL person ``target_id`` -> ``(SmplParams (SMPL-X), BodyModel)``.

    Shared by ``load()`` (one retargeted person) and the multi-person debug view. EasyMocap places
    the body as ``R(Rh) @ J_canonical + Th`` about the SMPL pelvis, so the SMPL-X pelvis is re-placed
    on the SMPL one: ``transl = Th + R @ J0_smpl - J0_smplx`` (rest pelvis heights differ ~14cm)."""
    Rh, Th, poses, betas10 = _person_series(smpl_params, target_id)
    T = len(smpl_params)
    smplx_npz, smplh_npz, deftrafo = _resolve_assets(Path(smplx_dir), gender, smplh_dir, deftrafo_pkl)
    betas_x = smpl_betas_to_smplx(betas10, smplh_npz, smplx_npz, deftrafo)
    body = rest_body_model(betas_x, gender, Path(smplx_dir))
    Rmat = R.from_rotvec(Rh).as_matrix()                              # (T,3,3)
    transl = Th + np.einsum("tij,j->ti", Rmat, smpl_rest_pelvis(betas10, smplh_npz)) - body.rest_joints[0]
    z = np.zeros((T, 45), np.float32)
    params = SmplParams(
        betas=betas_x, global_orient=Rh.astype(np.float32), body_pose=poses[:, :63].astype(np.float32),
        left_hand_pose=z, right_hand_pose=z, transl=transl.astype(np.float32),
        gender=gender, model_type="smplx")
    return params, body


@register_loader("hoim3")
class HoiM3Loader:
    """SceneSpec -> RawMotion for a HOI-M3 sequence (one retargeted person + all objects)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("HOI-M3 needs spec.smpl_model_dir (the SMPL-X model directory)")
        human_npz = Path(spec.motion_path)
        hd = np.load(str(human_npz), allow_pickle=True)
        if str(hd["model"]) != "smpl":
            raise ValueError(f"HOI-M3 loader expects EasyMocap SMPL, got model={hd['model']!r}")
        gender = str(hd["gender"])
        fps = float(np.asarray(hd["mocap_frame_rate"]))
        smpl_params = hd["smpl_params"]

        # Retarget the chosen person (spec.person_id), defaulting to the first present at frame 0.
        ids = [int(np.asarray(p["id"])) for p in smpl_params[0]]
        target_id = ids[0] if spec.person_id is None else spec.person_id
        if target_id not in ids:
            raise ValueError(f"person_id {target_id} not among frame-0 people {ids}")
        params, body = build_person_params(smpl_params, target_id, gender, Path(spec.smpl_model_dir),
                                           spec.smplh_dir, spec.smpl2smplx_pkl)
        joints = body.bone_positions(params)[:, :len(SMPLX_BODY_JOINTS)].astype(np.float32)   # demo joints, Z-up

        object_npz = human_npz.with_name(human_npz.name.replace("_human.npz", "_object.npz"))
        scanned_dir = human_npz.parent.parent / "scanned_object"
        cache_dir = Path(spec.cache_dir) / "hoim3_meshes" if spec.cache_dir else _MESH_CACHE
        object_poses, object_meshes = (((), ()) if not object_npz.exists() else _objects(
            object_npz, scanned_dir, cache_dir, spec.object_mesh_paths, spec.object_names))

        return RawMotion(
            joint_pos=joints, joint_names=SMPLX_BODY_JOINTS, fps=fps, source_format="hoim3",
            object_poses_raw=object_poses, object_mesh_paths=object_meshes, smpl_params=params)
