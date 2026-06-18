import os

CONTACT_MARGIN_M = 0.10
OBJECT_FIELD_RESOLUTION = 0.01
FLOOR_GRID_SIZE = 4.0
FLOOR_GRID_DENSITY = 500.0
OBJECT_GRID_DENSITY = 5000.0
# OMOMO raw object data (for the largebox mesh) — external, not bundled and
# machine-specific: read it from the WBT_OMOMO_DIR env var. No hardcoded default —
# callers guard with os.path.isdir (tests skip, mesh degrades to the neutral shape)
# and real use fails clearly if the env var is unset.
OMOMO_DIR_DEFAULT = os.environ.get("WBT_OMOMO_DIR", "")
