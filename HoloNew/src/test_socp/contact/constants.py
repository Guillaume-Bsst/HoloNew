CONTACT_MARGIN_M = 0.10
OBJECT_FIELD_RESOLUTION = 0.01
FLOOR_GRID_SIZE = 4.0
FLOOR_GRID_DENSITY = 500.0
OBJECT_GRID_DENSITY = 5000.0


def _omomo_dir() -> str:
    # OMOMO release root (for the object meshes / subject betas) — external, not bundled
    # and machine-specific: read it from path.yaml (omomo). No hardcoded default —
    # callers guard with os.path.isdir (tests skip, mesh degrades to the neutral shape).
    try:
        from HoloNew.src.paths import get_path
        return str(get_path("omomo"))
    except Exception:  # noqa: BLE001
        return ""


OMOMO_DIR_DEFAULT = _omomo_dir()
