"""Validation tests for contracts.py: immutability (FIX 1), shape checks (FIX 2),
and Evaluator fail-fast guard (FIX 3).

Tests are RED before the fixes are applied; they go GREEN once the implementation lands.
"""
import numpy as np
import pytest

from src.prepare.contracts import (
    Channel, CorrespondenceTable, InteractionContext, PointCloud, SDF,
)
from src.targets.contracts import (
    ContactEval, ContactEnvEval, ContactField, FramePose, FrameTargets,
    MultiChannelField, StyleEval, StyleTargets,
)
from src.targets.evaluator import Evaluator


# ─────────────────────────────── helpers ────────────────────────────────────
def _mcf(c: int, p: int) -> MultiChannelField:
    return MultiChannelField(
        distance=np.zeros((c, p)), direction=np.zeros((c, p, 3)),
        witness=np.zeros((c, p, 3)), active=np.zeros((c, p), bool),
        channels=tuple(f"ch{i}" for i in range(c)))


def _frame_pose(j: int = 5, n: int = 2) -> FramePose:
    return FramePose(
        bone_rot=np.zeros((j, 3, 3)), bone_pos=np.zeros((j, 3)),
        object_rot=np.zeros((n, 3, 3)), object_pos=np.zeros((n, 3)))


# ════════════════════════════════════════════════════════════════════════════
# FIX 2 — shape validation
# ════════════════════════════════════════════════════════════════════════════

class TestMultiChannelFieldValidation:
    def test_valid_construction_ok(self):
        _mcf(2, 5)  # should not raise

    def test_p_mismatch_active(self):
        C, P = 2, 5
        with pytest.raises(ValueError, match="active"):
            MultiChannelField(
                distance=np.zeros((C, P)), direction=np.zeros((C, P, 3)),
                witness=np.zeros((C, P, 3)), active=np.zeros((C, P + 1), bool),
                channels=tuple(f"ch{i}" for i in range(C)))

    def test_p_mismatch_direction(self):
        C, P = 2, 5
        with pytest.raises(ValueError, match="direction"):
            MultiChannelField(
                distance=np.zeros((C, P)), direction=np.zeros((C, P + 1, 3)),
                witness=np.zeros((C, P, 3)), active=np.zeros((C, P), bool),
                channels=tuple(f"ch{i}" for i in range(C)))

    def test_p_mismatch_witness(self):
        C, P = 2, 5
        with pytest.raises(ValueError, match="witness"):
            MultiChannelField(
                distance=np.zeros((C, P)), direction=np.zeros((C, P, 3)),
                witness=np.zeros((C, P + 1, 3)), active=np.zeros((C, P), bool),
                channels=tuple(f"ch{i}" for i in range(C)))

    def test_direction_bad_trailing_dim(self):
        C, P = 2, 5
        with pytest.raises(ValueError, match="direction"):
            MultiChannelField(
                distance=np.zeros((C, P)), direction=np.zeros((C, P, 2)),
                witness=np.zeros((C, P, 3)), active=np.zeros((C, P), bool),
                channels=tuple(f"ch{i}" for i in range(C)))

    def test_witness_bad_trailing_dim(self):
        C, P = 2, 5
        with pytest.raises(ValueError, match="witness"):
            MultiChannelField(
                distance=np.zeros((C, P)), direction=np.zeros((C, P, 3)),
                witness=np.zeros((C, P, 2)), active=np.zeros((C, P), bool),
                channels=tuple(f"ch{i}" for i in range(C)))


class TestStyleTargetsValidation:
    def test_valid_construction_ok(self):
        L = 3
        StyleTargets(link_names=("a", "b", "c"),
                     position=np.zeros((L, 3)), orientation=np.zeros((L, 4)))

    def test_valid_no_orientation(self):
        L = 3
        StyleTargets(link_names=("a", "b", "c"), position=np.zeros((L, 3)))

    def test_valid_empty_links(self):
        StyleTargets(link_names=(), position=np.zeros((0, 3)))

    def test_position_l_mismatch_raises(self):
        L = 3
        with pytest.raises(ValueError, match="position"):
            StyleTargets(link_names=("a", "b", "c"), position=np.zeros((L + 1, 3)))

    def test_orientation_l_mismatch_raises(self):
        L = 3
        with pytest.raises(ValueError, match="orientation"):
            StyleTargets(link_names=("a", "b", "c"),
                         position=np.zeros((L, 3)),
                         orientation=np.zeros((L + 1, 4)))


class TestFramePoseValidation:
    def test_valid_construction_ok(self):
        _frame_pose(5, 2)

    def test_valid_zero_bones(self):
        _frame_pose(0, 0)  # body=None path uses (0, 3, 3) / (0, 3)

    def test_j_mismatch_raises(self):
        with pytest.raises(ValueError, match="bone"):
            FramePose(bone_rot=np.zeros((5, 3, 3)), bone_pos=np.zeros((6, 3)),
                      object_rot=np.zeros((2, 3, 3)), object_pos=np.zeros((2, 3)))

    def test_n_mismatch_raises(self):
        with pytest.raises(ValueError, match="object"):
            FramePose(bone_rot=np.zeros((5, 3, 3)), bone_pos=np.zeros((5, 3)),
                      object_rot=np.zeros((2, 3, 3)), object_pos=np.zeros((3, 3)))


# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — immutability at the type level
# ════════════════════════════════════════════════════════════════════════════

class TestImmutability:
    def test_contact_field_arrays_immutable(self):
        P = 4
        cf = ContactField(distance=np.zeros(P), direction=np.zeros((P, 3)),
                          witness=np.zeros((P, 3)), active=np.zeros(P, bool))
        assert not cf.distance.flags.writeable
        assert not cf.direction.flags.writeable
        assert not cf.witness.flags.writeable
        assert not cf.active.flags.writeable

    def test_mcf_arrays_immutable(self):
        f = _mcf(2, 5)
        assert not f.distance.flags.writeable
        assert not f.direction.flags.writeable
        assert not f.witness.flags.writeable
        assert not f.active.flags.writeable

    def test_style_targets_arrays_immutable(self):
        L = 3
        st = StyleTargets(link_names=("a", "b", "c"),
                          position=np.zeros((L, 3)), orientation=np.zeros((L, 4)))
        assert not st.position.flags.writeable
        assert st.orientation is not None and not st.orientation.flags.writeable

    def test_style_targets_no_orientation_position_immutable(self):
        L = 2
        st = StyleTargets(link_names=("a", "b"), position=np.zeros((L, 3)))
        assert not st.position.flags.writeable

    def test_frame_pose_arrays_immutable(self):
        fp = _frame_pose(3, 2)
        assert not fp.bone_rot.flags.writeable
        assert not fp.bone_pos.flags.writeable
        assert not fp.object_rot.flags.writeable
        assert not fp.object_pos.flags.writeable

    def test_style_eval_arrays_immutable(self):
        L, nv = 3, 7
        se = StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                       jac_pos=np.zeros((L, 3, nv)), jac_rot=np.zeros((L, 3, nv)),
                       link_names=("a", "b", "c"))
        assert not se.position.flags.writeable
        assert not se.rotation.flags.writeable
        assert not se.jac_pos.flags.writeable
        assert not se.jac_rot.flags.writeable

    def test_contact_env_eval_arrays_immutable(self):
        C, P = 2, 4
        env = ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P, 3, 6)),
                             probe_jac_obj=np.zeros((C, P, 3, 6)))
        assert not env.cloud_jac_self.flags.writeable
        assert not env.probe_jac_obj.flags.writeable

    def test_contact_eval_arrays_immutable(self):
        C, M, nv = 2, 5, 7
        env = (ContactEnvEval(field=_mcf(C, 3), cloud_jac_self=np.zeros((3, 3, 6)),
                              probe_jac_obj=np.zeros((C, 3, 3, 6))),)
        ce = ContactEval(field=_mcf(C, M), point_jac=np.zeros((M, 3, nv)),
                         probe_jac_obj=np.zeros((C, M, 3, 6)), env=env)
        assert not ce.point_jac.flags.writeable
        assert not ce.probe_jac_obj.flags.writeable


# ════════════════════════════════════════════════════════════════════════════
# FIX 3 — Evaluator fail-fast guard
# ════════════════════════════════════════════════════════════════════════════

class _StubRobot:
    """Minimal RobotModel-like stub with only 'pelvis' in link_names (missing 13 g1 style links)."""

    link_names = ("pelvis",)
    dof = 1
    nq = 8
    nv = 7

    def link_transforms(self, q):
        return np.zeros((1, 3, 3)), np.zeros((1, 3))

    def rest_transforms(self):
        return np.zeros((1, 3, 3)), np.zeros((1, 3))

    def neutral(self):
        return np.zeros(self.nq)

    def integrate(self, q, v):
        return np.zeros(self.nq)

    def link_jacobians(self, q):
        return (np.zeros((1, 3, 3)), np.zeros((1, 3)),
                np.zeros((1, 3, self.nv)), np.zeros((1, 3, self.nv)))


def _minimal_ctx() -> InteractionContext:
    """Minimal InteractionContext using a stub robot with only 'pelvis' in link_names."""
    sdf = SDF(grid=np.zeros((2, 2, 2)), witness=np.zeros((2, 2, 2, 3)),
              origin=np.zeros(3), spacing=0.1, name="ground")
    channel = Channel(name="ground", object_idx=None, sdf=sdf)
    cloud = PointCloud(parts=np.zeros((1, 1), int), weights=np.ones((1, 1)),
                       offsets=np.zeros((1, 1, 3)))
    corr = CorrespondenceTable(smpl_idx=np.zeros(1, int), link_idx=np.zeros(1, int),
                               offset_local=np.zeros((1, 3)), link_names=("a",))
    return InteractionContext(
        channels=(channel,), human_cloud=cloud, object_clouds=(),
        correspondence=corr, margin=0.05, robot_cloud=cloud,
        robot=_StubRobot())


def test_evaluator_raises_on_missing_robot_links():
    """Evaluator must fail at CONSTRUCTION when robot.link_names is missing style links for the
    given robot_name — not silently fail at the first call to .style()."""
    ctx = _minimal_ctx()
    with pytest.raises(ValueError, match="style links"):
        Evaluator(ctx, "g1")
