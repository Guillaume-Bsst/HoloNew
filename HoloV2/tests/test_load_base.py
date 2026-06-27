"""Unit tests for the dataset-loader registry (``src.prepare.load.base``)."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RawMotion, RobotSpec, SceneSpec
from src.prepare.load import base


def _spec(dataset: str) -> SceneSpec:
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=1, height=1.3)
    return SceneSpec(dataset=dataset, motion_path=Path("m"), robot=robot)


def _raw() -> RawMotion:
    return RawMotion(
        joint_pos=np.zeros((3, 2, 3)), joint_names=("a", "b"), fps=30.0,
        source_format="dummy", object_poses_raw=(), object_mesh_paths=(),
    )


def test_register_and_dispatch():
    @base.register_loader("dummy_ok")
    class _Loader:
        def load(self, spec):
            return _raw()

    assert isinstance(base.get_loader("dummy_ok"), base.MotionLoader)
    raw = base.load(_spec("dummy_ok"))
    assert raw.n_frames == 3
    assert raw.is_parametric is False


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        base.get_loader("does_not_exist")


def test_duplicate_registration_raises():
    @base.register_loader("dup")
    class _A:
        def load(self, spec):
            return _raw()

    with pytest.raises(ValueError):
        @base.register_loader("dup")
        class _B:
            def load(self, spec):
                return _raw()
