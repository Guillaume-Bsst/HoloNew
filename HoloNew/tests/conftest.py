# tests/conftest.py
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parent
PKG = HERE.parent

@pytest.fixture(scope="session")
def golden_qpos():
    return np.load(HERE / "golden" / "baseline_qpos.npz")["qpos"]

@pytest.fixture(scope="session")
def robot_urdf():
    return str(PKG / "models" / "g1" / "g1_29dof.urdf")
