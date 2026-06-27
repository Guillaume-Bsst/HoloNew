"""prepare/calibration tests: the pure grounding/scale math (synthetic, torch-free), the
GroundedScene assembly, the builder determinism + cache round-trip, and an end-to-end grounding
check on real SFU data (skipped if absent)."""
from pathlib import Path

import numpy as np
import pytest

from holov2.contracts import (Calibration, CalibrationConfig, RawMotion, RobotSpec, SmplParams)
from holov2.prepare import scene
from holov2.prepare.calibration import (CalibrationBuilder, DEFAULT_HUMAN_HEIGHT, build_calibration,
                                        foot_floor_offset, human_stature, object_floor_offset)
from holov2.prepare.calibration.build import _foot_indices


def _write_cube_obj(path: Path, half: float = 0.5) -> None:
    """Write a minimal axis-aligned cube OBJ (local z in [-half, half]) for object-offset tests."""
    v = [(sx * half, sy * half, sz * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    f = [(1, 2, 4), (1, 4, 3), (5, 6, 8), (5, 8, 7), (1, 2, 6), (1, 6, 5),
         (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 4, 8), (2, 8, 6)]
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v) +
                    "".join(f"f {a} {b} {c}\n" for a, b, c in f))

_SMPLX_JOINTS = ("Pelvis", "L_Hip", "R_Hip", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot")


# --------------------------------------------------------------------------- fakes / fixtures
class FakeBody:
    """Minimal BodyModel stub: a fixed rest mesh (for the betas-FK stature). No torch — the floor
    offset is mocap-joint based, so the body is only ever used for ``rest_vertices``."""

    faces = np.zeros((1, 3), np.int64)
    n_bones = 55

    def __init__(self, rest_y_extent: float = 1.75):
        v = np.zeros((4, 3))
        v[:, 1] = np.array([-rest_y_extent / 2, rest_y_extent / 2, 0.0, 0.1])
        self._rest = v

    def rest_vertices(self, params):
        return self._rest


def _params(T: int) -> SmplParams:
    z = np.zeros
    return SmplParams(betas=z(16, np.float32), global_orient=z((T, 3), np.float32),
                      body_pose=z((T, 63), np.float32), left_hand_pose=z((T, 45), np.float32),
                      right_hand_pose=z((T, 45), np.float32), transl=z((T, 3), np.float32),
                      gender="neutral", model_type="smplx")


def _raw(T: int, parametric: bool = True, with_object: bool = False) -> RawMotion:
    joints = np.zeros((T, len(_SMPLX_JOINTS), 3), np.float32)
    joints[:, :, 2] = 0.3                                   # all joints 0.3 m up
    joints[:, 5:7, 2] = 0.05                                # feet (L_Foot, R_Foot) at 0.05 m
    objs = ((np.tile([1.0, 2.0, 0.4, 1, 0, 0, 0], (T, 1)).astype(np.float32),) if with_object else ())
    return RawMotion(joint_pos=joints, joint_names=_SMPLX_JOINTS, fps=30.0, source_format="fake",
                     object_poses_raw=objs, object_mesh_paths=((Path("o.obj"),) if with_object else ()),
                     smpl_params=_params(T) if parametric else None)


def _robot(h: float = 1.3) -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=h)


# --------------------------------------------------------------------------- pure functions
def test_human_stature():
    assert human_stature(FakeBody(rest_y_extent=1.75), _params(1)) == pytest.approx(1.75)


def test_foot_indices_finds_feet_then_ankles():
    assert _foot_indices(_SMPLX_JOINTS) == [5, 6]                      # L_Foot, R_Foot
    assert _foot_indices(("Pelvis", "L_Ankle", "R_Ankle")) == [1, 2]   # ankle fallback
    with pytest.raises(ValueError):
        _foot_indices(("Pelvis", "Head"))


def test_foot_floor_offset_percentile():
    # lower foot z (min over the two feet) ramps 0.0 -> 0.20 over 5 frames; the other foot is higher.
    jp = np.zeros((5, 7, 3))
    jp[:, 5, 2] = [0.00, 0.05, 0.10, 0.15, 0.20]          # L_Foot (the lower one)
    jp[:, 6, 2] = 0.30                                    # R_Foot, always higher
    assert foot_floor_offset(jp, [5, 6], 50.0) == pytest.approx(0.10)   # median
    assert foot_floor_offset(jp, [5, 6], 0.0) == pytest.approx(0.00)    # lowest
    assert foot_floor_offset(jp, [5, 6], 100.0) == pytest.approx(0.20)  # highest


def test_object_floor_offset_percentile():
    cube = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])  # z in [-1,1]
    poses = np.array([[0, 0, h, 1, 0, 0, 0] for h in (0.0, 1.0, 2.0)], float)   # identity rot, height h
    # lowest world z per frame = h + (-1) = [-1, 0, 1].
    assert object_floor_offset([cube], [poses], 0.0) == pytest.approx(-1.0)     # lowest reach
    assert object_floor_offset([cube], [poses], 50.0) == pytest.approx(0.0)
    assert object_floor_offset([], [], 5.0) == 0.0                              # no objects


# --------------------------------------------------------------------------- builder
def test_builder_human_offset_from_foot_percentile():
    calib = build_calibration(_raw(3), CalibrationConfig(), body=FakeBody())
    assert calib.human_stature == pytest.approx(1.75)                # real subject stature, robot-free
    assert calib.human_offset == pytest.approx(0.05)                 # foot-joint p50 (feet at 0.05)
    assert calib.object_offset == 0.0                                # no objects
    assert np.allclose(calib.root_frame, np.eye(4))


def test_builder_non_parametric_uses_default_stature():
    calib = build_calibration(_raw(3, parametric=False), CalibrationConfig())
    assert calib.human_offset == pytest.approx(0.05)                 # foot joints (same path)
    assert calib.object_offset == 0.0                                # no objects
    assert calib.human_stature == pytest.approx(DEFAULT_HUMAN_HEIGHT)  # default (no betas to FK)


def test_builder_object_offset_grounds_lowest_object(tmp_path):
    obj = tmp_path / "cube.obj"; _write_cube_obj(obj, half=0.5)      # local z in [-0.5, 0.5]
    T = 4
    joints = np.zeros((T, len(_SMPLX_JOINTS), 3), np.float32); joints[:, 5:7, 2] = 0.05
    pose = np.tile([0.0, 0.0, 0.2, 1, 0, 0, 0], (T, 1)).astype(np.float32)   # centroid at z=0.2
    raw = RawMotion(joint_pos=joints, joint_names=_SMPLX_JOINTS, fps=30.0, source_format="fake",
                    object_poses_raw=(pose,), object_mesh_paths=(obj,), smpl_params=None)
    calib = build_calibration(raw, CalibrationConfig())              # no body needed (non-parametric)
    assert calib.object_offset == pytest.approx(-0.3, abs=1e-6)      # lowest point = 0.2 - 0.5


def test_builder_parametric_requires_body_or_model_dir():
    with pytest.raises(ValueError):
        build_calibration(_raw(3), CalibrationConfig())              # no body, no model dir


def test_builder_deterministic_and_cache_roundtrip(tmp_path):
    obj = tmp_path / "cube.obj"; _write_cube_obj(obj)
    T = 3
    joints = np.zeros((T, len(_SMPLX_JOINTS), 3), np.float32); joints[:, 5:7, 2] = 0.05
    pose = np.tile([0.0, 0.0, 0.2, 1, 0, 0, 0], (T, 1)).astype(np.float32)
    raw = RawMotion(joint_pos=joints, joint_names=_SMPLX_JOINTS, fps=30.0, source_format="fake",
                    object_poses_raw=(pose,), object_mesh_paths=(obj,), smpl_params=None)
    b = CalibrationBuilder()
    cfg = CalibrationConfig()
    c1 = b.build(cfg, raw)
    c2 = b.build(cfg, raw)
    assert (c1.human_stature, c1.human_offset, c1.object_offset) == \
           (c2.human_stature, c2.human_offset, c2.object_offset)
    assert b.cache_key(cfg, raw) == b.cache_key(cfg, raw)            # stable, robot-free key
    p = tmp_path / "calib.npz"
    b.save(c1, p)
    loaded = b.load(p)
    assert loaded.human_stature == pytest.approx(c1.human_stature)
    assert loaded.human_offset == pytest.approx(c1.human_offset)
    assert loaded.object_offset == pytest.approx(c1.object_offset)
    assert np.allclose(loaded.root_frame, c1.root_frame)


# --------------------------------------------------------------------------- scene assembly
def test_assemble_grounds_human_and_objects_by_their_own_offsets():
    raw = _raw(4, with_object=True)
    # Distinct human vs object offsets -> the test fails if they are not applied independently.
    calib = Calibration(human_stature=1.75, human_offset=0.2, object_offset=0.15, root_frame=np.eye(4))
    g = scene.assemble(raw, calib)
    assert g.joint_pos.shape == raw.joint_pos.shape
    assert np.allclose(g.joint_pos[:, :, 2], raw.joint_pos[:, :, 2] - 0.2)        # human: human_offset
    assert np.allclose(g.smpl_params.transl[:, 1], raw.smpl_params.transl[:, 1] - 0.2)  # native y
    assert np.allclose(g.object_poses[0][:, 2], raw.object_poses_raw[0][:, 2] - 0.15)   # object: shared
    assert np.allclose(g.object_poses[0][:, :2], raw.object_poses_raw[0][:, :2])        # xy untouched
    assert g.calibration.human_stature == 1.75    # carried; scale composed downstream, not baked here
    assert g.is_parametric and g.n_objects == 1


def test_assemble_non_parametric_scene():
    g = scene.assemble(_raw(3, parametric=False), Calibration(1.7, 0.1, 0.0, np.eye(4)))
    assert g.smpl_params is None and not g.is_parametric and g.n_objects == 0


# --------------------------------------------------------------------------- real data (skip if absent)
_NPZ = Path("/home/vboxuser/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew/demo_data/"
            "SFU/0005_2FeetJump001.npz")
_SMPLX = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx")


@pytest.mark.skipif(not (_NPZ.exists() and _SMPLX.is_dir()), reason="SFU data / SMPL-X model absent")
def test_calibration_grounds_real_sfu_foot_to_zero():
    from holov2.contracts import SceneSpec
    from holov2.prepare.load import load
    from holov2.prepare.load.smpl import build_body_model

    spec = SceneSpec(dataset="sfu", motion_path=_NPZ, robot=_robot(1.3), smpl_model_dir=_SMPLX)
    raw = load(spec)
    body = build_body_model(raw.smpl_params, _SMPLX)

    calib = build_calibration(raw, CalibrationConfig(), body=body)   # robot-free, foot p50
    assert 1.4 < calib.human_stature < 2.1                  # a plausible human stature (m)
    assert abs(calib.human_offset) < 0.3                    # SFU is already near the floor
    assert calib.object_offset == 0.0                       # SFU is body-only

    # End-to-end: grounding zeroes the foot-joint p50 residual (the offset IS that percentile).
    g = scene.assemble(raw, calib)
    foot = _foot_indices(g.joint_names)
    residual = foot_floor_offset(g.joint_pos, foot, 50.0)
    assert abs(residual) < 1e-5, f"grounded foot still off by {residual:.5f} m"
