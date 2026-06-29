"""style.build tests: SMPL bones -> ``StyleTargets`` (GMR posture tracking, G1 table).

Three layers:
- UNIT (synthetic, torch-free): a small ``FramePose`` with KNOWN bone transforms + a fake ``RobotSpec``
  -> shapes / link names / weights, and a hand-computed scale + offset check for two links.
- V1 PARITY (synthetic): feed the SAME bone-derived inputs through HoloNew's V1 ``scale``/``offset``
  pure functions (loaded torch-free) and compare per link — validates the ported math against the
  reference. (We compare against V1 ``scale``+``offset`` directly, NOT the full ``compute_stages``,
  which adds the clip-wide FLOOR drop we intentionally drop and needs the 52-joint MuJoCo .pt loader.)
- REAL-DATA structural (skips when HODome / SMPL-X / corr_neutral are absent): prepare -> frame_pose
  -> style.build; asserts 14 finite links + weights, the pelvis target near the SMPL pelvis, and
  PRINTS the min foot-target z (does dropping the floor stage float the feet?).
"""
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import RobotSpec
from src.targets.config import SMPL_BODY_INDEX, StyleConfig, style_table
from src.targets.contracts import FramePose
from src.targets import style
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX, V1_TEST_SOCP as _V1

_G1_LINKS = tuple(style_table("g1").keys())
_CFG = StyleConfig()   # defaults: the morphological SCALE values + reference heights the test recomputes


# --------------------------------------------------------------------------- helpers
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _robot() -> RobotSpec:
    # real URDF: the synthetic tests only read robot.name, but the real-data test runs prepare() which
    # now loads the URDF eagerly (pinocchio) — a fake path would fail there.
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


def _quat_wxyz_to_mat(q) -> np.ndarray:
    q = np.asarray(q, float)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _synthetic_pose() -> FramePose:
    """22 SMPL-X bones: identity rotations except a rotated pelvis (tests the orientation compose),
    distinct positions on the bodies the unit checks read (pelvis / left_knee / left_foot)."""
    J = 22
    bone_rot = np.tile(np.eye(3), (J, 1, 1)).astype(np.float64)
    bone_rot[SMPL_BODY_INDEX["pelvis"]] = R.from_euler("z", 90, degrees=True).as_matrix()
    bone_pos = np.zeros((J, 3), np.float64)
    bone_pos[SMPL_BODY_INDEX["pelvis"]] = [1.0, 2.0, 3.0]
    bone_pos[SMPL_BODY_INDEX["left_knee"]] = [1.5, 2.0, 1.0]
    bone_pos[SMPL_BODY_INDEX["left_foot"]] = [1.2, 2.0, 0.1]
    return FramePose(bone_rot=bone_rot, bone_pos=bone_pos,
                     object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))


# --------------------------------------------------------------------------- unit
def test_style_shapes_and_links():
    st = style.build(_synthetic_pose(), _robot(), stature=0.9)        # ratio = 0.5
    L = len(_G1_LINKS)
    assert st.link_names == _G1_LINKS and L == 14
    assert st.position.shape == (L, 3)
    assert st.orientation is not None and st.orientation.shape == (L, 4)
    assert st.position.dtype == np.float64 and st.orientation.dtype == np.float64
    # StyleTargets is GEOMETRY only — tracking weights are a solver concern, not produced here.
    assert not hasattr(st, "weight_pos") and not hasattr(st, "weight_rot")
    # frozen numpy output (read-only buffers).
    for a in (st.position, st.orientation):
        assert a.flags.writeable is False


def test_style_scale_and_offset_hand_computed():
    pose = _synthetic_pose()
    stature = 0.9
    ratio = stature / _CFG.human_height_assumption                    # 0.5
    base = _CFG.scale_torso_legs * ratio                             # 0.45 (pelvis is torso/legs)
    root = pose.bone_pos[SMPL_BODY_INDEX["pelvis"]]
    scaled_root = np.array([root[0], root[1], root[2] * base])        # sx=sy=1, sz=base
    st = style.build(pose, _robot(), stature=stature)
    idx = {n: i for i, n in enumerate(st.link_names)}

    # pelvis: pos_offset 0 -> world target == scaled_root (scale applied, x/y native, z*base).
    np.testing.assert_allclose(st.position[idx["pelvis"]], scaled_root, atol=1e-9)
    # pelvis orientation: src_rot (Rz90) composed with rot_offset (0.5,-0.5,-0.5,-0.5).
    exp_rot = pose.bone_rot[SMPL_BODY_INDEX["pelvis"]] @ _quat_wxyz_to_mat((0.5, -0.5, -0.5, -0.5))
    got_rot = R.from_quat(st.orientation[idx["pelvis"]][[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(got_rot, exp_rot, atol=1e-9)

    # left_knee: pos_offset 0, identity src_rot -> world target == pure SCALE in pelvis-local frame.
    s_knee = _CFG.scale_torso_legs * ratio                          # 0.45 (knee is torso/legs)
    exp_knee = (pose.bone_pos[SMPL_BODY_INDEX["left_knee"]] - root) * s_knee + scaled_root
    np.testing.assert_allclose(st.position[idx["left_knee_link"]], exp_knee, atol=1e-9)

    # left foot (ankle_roll link): nonzero pos_offset (0, 0.02, 0) rotated into the re-oriented
    # (rot_offset) body frame. The GMR "toe" frame is remapped to the G1 foot link ``ankle_roll``.
    s_foot = _CFG.scale_torso_legs * ratio                          # foot is torso/legs
    scaled_foot = (pose.bone_pos[SMPL_BODY_INDEX["left_foot"]] - root) * s_foot + scaled_root
    rot = _quat_wxyz_to_mat((0.5, -0.5, -0.5, -0.5))                 # identity src_rot on left_foot
    exp_foot = scaled_foot + rot @ np.array([0.0, 0.02, 0.0])
    np.testing.assert_allclose(st.position[idx["left_ankle_roll_link"]], exp_foot, atol=1e-9)


def test_style_unknown_robot_raises():
    with pytest.raises(ValueError):
        style.build(_synthetic_pose(), RobotSpec(name="zzz", urdf_path=Path("x"),
                                                 link_names=("p",), dof=1, height=1.0), stature=1.0)


# --------------------------------------------------------------------------- V1 parity


def _load_v1_preprocess():
    """Load HoloNew V1 ``test_socp/preprocess`` torch-free (rewrite its relative ``.tables`` import
    onto a tiny shim package), or None if the V1 tree is absent."""
    if not (_V1 / "preprocess.py").exists() or not (_V1 / "tables.py").exists():
        return None
    pkg = types.ModuleType("v1gmr"); pkg.__path__ = []; sys.modules["v1gmr"] = pkg
    st = importlib.util.spec_from_file_location("v1gmr.tables", _V1 / "tables.py")
    mt = importlib.util.module_from_spec(st); sys.modules["v1gmr.tables"] = mt; st.loader.exec_module(mt)
    src = (_V1 / "preprocess.py").read_text().replace("from .tables import", "from v1gmr.tables import")
    mp = types.ModuleType("v1gmr.preprocess")
    exec(compile(src, str(_V1 / "preprocess.py"), "exec"), mp.__dict__)
    return mp


_V1PP = _load_v1_preprocess()


@pytest.mark.skipif(_V1PP is None, reason="HoloNew V1 test_socp/preprocess absent")
def test_style_matches_v1_scale_offset():
    """Identical bone-derived inputs through V1 ``scale``+``offset`` (scale_xy=1.0, scale_z=None) must
    reproduce style.build per link (position to machine precision, orientation up to quaternion sign)."""
    pose = _synthetic_pose()
    stature = 1.62
    ratio = stature / _CFG.human_height_assumption
    # V1 HumanData: {smpl_body: (world pos, world quat wxyz)} from the SAME bones we feed style.build.
    human_data = {}
    for body, idx in SMPL_BODY_INDEX.items():
        q_xyzw = R.from_matrix(pose.bone_rot[idx]).as_quat()
        human_data[body] = (pose.bone_pos[idx].astype(float), q_xyzw[[3, 0, 1, 2]].astype(float))
    od = _V1PP.offset(_V1PP.scale(human_data, ratio, scale_xy=1.0, scale_z=None))

    st = style.build(pose, _robot(), stature=stature)
    table = style_table("g1")
    for i, link in enumerate(st.link_names):
        body = table[link][0]
        v1_pos, v1_quat = od[body]
        np.testing.assert_allclose(st.position[i], v1_pos, atol=1e-9, err_msg=f"pos {link}")
        d = min(np.linalg.norm(st.orientation[i] - v1_quat), np.linalg.norm(st.orientation[i] + v1_quat))
        assert d < 1e-7, f"quat {link}: {st.orientation[i]} vs {v1_quat}"


# --------------------------------------------------------------------------- real-data structural
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()


@pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model / corr_neutral.npz absent")
def test_style_on_real_data(tmp_path, capsys):
    from src.prepare.config import PrepareConfig
    from src.prepare.contracts import SceneSpec
    from src.prepare.runner import prepare
    from src.targets.pipeline import frame_pose

    (tmp_path / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ, robot=_robot(),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    grounded, _ = prepare(spec, PrepareConfig())

    pose = frame_pose(grounded, 0)
    st = style.build(pose, _robot(), grounded.body.stature)

    assert st.link_names == _G1_LINKS
    assert np.isfinite(st.position).all() and np.isfinite(st.orientation).all()
    assert len(st.position) == 14

    # pelvis target near the SMPL pelvis (x/y native by scale_xy=1.0; z is morphologically scaled).
    pelvis_i = st.link_names.index("pelvis")
    smpl_pelvis = pose.bone_pos[SMPL_BODY_INDEX["pelvis"]]
    np.testing.assert_allclose(st.position[pelvis_i][:2], smpl_pelvis[:2], atol=1e-6)

    # min foot-target z over the foot (ankle_roll) links: does dropping the clip-wide FLOOR stage float the feet?
    foot_z = [st.position[st.link_names.index(n)][2] for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    min_foot_z = float(min(foot_z))
    with capsys.disabled():
        print(f"\n[style real-data] min foot-target z = {min_foot_z:.4f} m "
              f"(left={foot_z[0]:.4f}, right={foot_z[1]:.4f}); subject stature={grounded.body.stature:.3f}")
    assert np.isfinite(min_foot_z)
