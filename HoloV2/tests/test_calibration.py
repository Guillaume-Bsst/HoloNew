"""prepare/calibration tests: the pure grounding/scale math (synthetic, torch-free), the
GroundedScene assembly, the builder determinism + cache round-trip, and an end-to-end grounding
check on real SFU data (skipped if absent)."""
from pathlib import Path

import numpy as np
import pytest

from holov2.contracts import (Calibration, CalibrationConfig, RawMotion, RobotSpec, SmplParams)
from holov2.prepare import scene
from holov2.prepare.calibration import (CalibrationBuilder, DEFAULT_HUMAN_HEIGHT, build_calibration,
                                        human_stature, sole_floor_offset, toe_ground_offset)
from holov2.prepare.calibration.build import _toe_indices

_SMPLX_JOINTS = ("Pelvis", "L_Hip", "R_Hip", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot")


# --------------------------------------------------------------------------- fakes / fixtures
class FakeBody:
    """Minimal BodyModel stub: a fixed rest mesh (for stature) and a per-frame posed cloud whose
    lowest vertex z is a prescribed value (for the floor offset). No torch."""

    faces = np.zeros((1, 3), np.int64)
    n_bones = 55

    def __init__(self, rest_y_extent: float, frame_min_z: np.ndarray):
        v = np.zeros((4, 3))
        v[:, 1] = np.array([-rest_y_extent / 2, rest_y_extent / 2, 0.0, 0.1])
        self._rest = v
        self._min_z = np.asarray(frame_min_z, float)

    def rest_vertices(self, params):
        return self._rest

    def posed_vertices(self, params, t):
        z = self._min_z[t]
        return np.array([[0.0, 0.0, z], [0.1, 0.0, z + 1.0]])


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
    body = FakeBody(rest_y_extent=1.75, frame_min_z=np.zeros(1))
    assert human_stature(body, _params(1)) == pytest.approx(1.75)


def test_toe_indices_finds_feet_then_ankles():
    assert _toe_indices(_SMPLX_JOINTS) == [5, 6]                       # L_Foot, R_Foot
    assert _toe_indices(("Pelvis", "L_Ankle", "R_Ankle")) == [1, 2]    # ankle fallback
    with pytest.raises(ValueError):
        _toe_indices(("Pelvis", "Head"))


def test_toe_ground_offset_normal_and_mat():
    j = _raw(3).joint_pos                                  # feet at 0.05
    assert toe_ground_offset(j, [5, 6], mat_height=0.0) == pytest.approx(0.05)
    # On a mat: lowest toe a full mat_height up -> keep it on the mat top (drop less).
    j2 = j.copy(); j2[:, 5:7, 2] = 0.15
    assert toe_ground_offset(j2, [5, 6], mat_height=0.1) == pytest.approx(0.05)


def test_sole_floor_offset_median_mat_and_margin():
    body = FakeBody(1.75, frame_min_z=np.array([0.07, 0.07, 0.07, 0.50]))   # one crouch outlier
    p = _params(4)
    assert sole_floor_offset(body, p, mat_height=0.0) == pytest.approx(0.07)       # robust median
    assert sole_floor_offset(body, p, mat_height=0.0, contact_margin=0.02) == pytest.approx(0.09)
    body2 = FakeBody(1.75, frame_min_z=np.full(4, 0.15))
    assert sole_floor_offset(body2, p, mat_height=0.1) == pytest.approx(0.05)      # mat path


# --------------------------------------------------------------------------- builder
def test_builder_uses_surface_when_parametric():
    body = FakeBody(rest_y_extent=1.75, frame_min_z=np.full(3, 0.08))
    calib = build_calibration(_raw(3), CalibrationConfig(mat_height=0.0), body=body)
    assert calib.human_stature == pytest.approx(1.75)                # real subject stature, robot-free
    assert calib.floor_offset == pytest.approx(0.08)                 # surface sole, not toe (0.05)
    assert np.allclose(calib.root_frame, np.eye(4))


def test_builder_falls_back_to_toe_when_non_parametric():
    calib = build_calibration(_raw(3, parametric=False), CalibrationConfig(mat_height=0.0))
    assert calib.floor_offset == pytest.approx(0.05)                 # toe joints
    assert calib.human_stature == pytest.approx(DEFAULT_HUMAN_HEIGHT)  # default (no betas to FK)


def test_builder_parametric_requires_body_or_model_dir():
    with pytest.raises(ValueError):
        build_calibration(_raw(3), CalibrationConfig())              # no body, no model dir


def test_builder_deterministic_and_cache_roundtrip(tmp_path):
    body = FakeBody(1.75, frame_min_z=np.full(3, 0.08))
    b = CalibrationBuilder()
    cfg, raw = CalibrationConfig(mat_height=0.0), _raw(3)
    c1 = b.build(cfg, raw, body=body)
    c2 = b.build(cfg, raw, body=body)
    assert (c1.human_stature, c1.floor_offset) == (c2.human_stature, c2.floor_offset)
    assert b.cache_key(cfg, raw) == b.cache_key(cfg, raw)            # stable, robot-free key
    p = tmp_path / "calib.npz"
    b.save(c1, p)
    loaded = b.load(p)
    assert loaded.human_stature == pytest.approx(c1.human_stature)
    assert loaded.floor_offset == pytest.approx(c1.floor_offset)
    assert np.allclose(loaded.root_frame, c1.root_frame)


# --------------------------------------------------------------------------- scene assembly
def test_assemble_grounds_scene_and_carries_calibration():
    raw = _raw(4, with_object=True)
    calib = Calibration(human_stature=1.75, floor_offset=0.2, root_frame=np.eye(4))
    g = scene.assemble(raw, calib)
    assert g.joint_pos.shape == raw.joint_pos.shape
    assert np.allclose(g.joint_pos[:, :, 2], raw.joint_pos[:, :, 2] - 0.2)       # joints dropped
    assert np.allclose(g.object_poses[0][:, 2], raw.object_poses_raw[0][:, 2] - 0.2)  # object dropped
    assert np.allclose(g.object_poses[0][:, :2], raw.object_poses_raw[0][:, :2])      # xy untouched
    assert np.allclose(g.smpl_params.transl[:, 1], raw.smpl_params.transl[:, 1] - 0.2)  # native y
    assert g.calibration.human_stature == 1.75    # carried; scale composed downstream, not baked here
    assert g.is_parametric and g.n_objects == 1


def test_assemble_non_parametric_scene():
    g = scene.assemble(_raw(3, parametric=False), Calibration(1.7, 0.1, np.eye(4)))
    assert g.smpl_params is None and not g.is_parametric and g.n_objects == 0


# --------------------------------------------------------------------------- real data (skip if absent)
_NPZ = Path("/home/vboxuser/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew/demo_data/"
            "SFU/0005_2FeetJump001.npz")
_SMPLX = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx")


@pytest.mark.skipif(not (_NPZ.exists() and _SMPLX.is_dir()), reason="SFU data / SMPL-X model absent")
def test_calibration_grounds_real_sfu_sole_to_zero():
    from holov2.contracts import SceneSpec
    from holov2.prepare.load import load
    from holov2.prepare.load.smpl import build_body_model

    spec = SceneSpec(dataset="sfu", motion_path=_NPZ, robot=_robot(1.3), smpl_model_dir=_SMPLX)
    raw = load(spec)
    body = build_body_model(raw.smpl_params, _SMPLX)
    cfg = CalibrationConfig(mat_height=0.0)                  # disable the mat branch -> sole hits z=0

    calib = build_calibration(raw, cfg, body=body)          # robot-free
    assert 1.4 < calib.human_stature < 2.1                  # a plausible human stature (m)
    assert abs(calib.floor_offset) < 0.3                    # SFU is already near the floor

    # End-to-end: grounding the params makes the residual sole offset vanish (reuse the same body).
    g = scene.assemble(raw, calib)
    residual = sole_floor_offset(body, g.smpl_params, mat_height=0.0)
    assert abs(residual) < 1e-3, f"grounded sole still off by {residual:.4f} m"
