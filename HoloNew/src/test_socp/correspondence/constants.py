from pathlib import Path

_PKG = Path(__file__).resolve().parents[3]       # .../HoloNew/HoloNew
G1_29DOF_URDF = str(_PKG / "models" / "g1" / "g1_29dof.urdf")
HUMAN_GRID_DENSITY = 2000.0
G1_DENSITY = 3000.0
OT_REG = 0.005


def _smplx_model_dir() -> str:
    # SMPL-X model dir is NOT bundled (license + size) and machine-specific: read it from
    # path.yaml (smplx_models). No hardcoded default — callers guard with os.path.isdir
    # (tests skip, builder degrades to the bundled correspondence) and real use fails
    # clearly if path.yaml lacks it.
    try:
        from HoloNew.src.paths import get_path
        return str(get_path("smplx_models"))   # the models/ dir (smplx/ lives under it)
    except Exception:  # noqa: BLE001
        return ""


SMPLX_MODEL_DIR_DEFAULT = _smplx_model_dir()
