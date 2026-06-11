from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent.parent       # .../HoloNew/HoloNew
G1_29DOF_URDF = str(_PKG / "models" / "g1" / "g1_29dof.urdf")
HUMAN_GRID_DENSITY = 2000.0
G1_DENSITY = 3000.0
OT_REG = 0.005
# SMPL-X model dir is NOT bundled (license + size); default to the local data dir.
SMPLX_MODEL_DIR_DEFAULT = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models"
