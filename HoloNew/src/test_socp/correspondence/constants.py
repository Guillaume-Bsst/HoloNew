import os
from pathlib import Path

_PKG = Path(__file__).resolve().parents[3]       # .../HoloNew/HoloNew
G1_29DOF_URDF = str(_PKG / "models" / "g1" / "g1_29dof.urdf")
HUMAN_GRID_DENSITY = 2000.0
G1_DENSITY = 3000.0
OT_REG = 0.005
# SMPL-X model dir is NOT bundled (license + size) and machine-specific: read it from
# the WBT_SMPLX_DIR env var. No hardcoded default — callers guard with os.path.isdir
# (tests skip, builder degrades to the bundled correspondence) and real use fails
# clearly at load time if the env var is unset.
SMPLX_MODEL_DIR_DEFAULT = os.environ.get("WBT_SMPLX_DIR", "")
