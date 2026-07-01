"""Resolved external dataset/model paths for the test suite — the single import point, so no test
hardcodes a machine-specific absolute path.

External roots live in ``HoloV2/paths.toml`` (the schema is ``[models]`` + ``[datasets.<name>]``); edit
that file for your machine. Repo-INTERNAL paths (the V1 ``test_socp`` source, ``HoloNew/demo_data``)
resolve relative to the repo, not the toml — they travel with the checkout.

Usage in a test (module level, for the skip gate):
    from datapaths import SMPLX_MODELS, HODOME, DEMO_DATA, V1_TEST_SOCP
"""
from __future__ import annotations

import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve()
_HOLOV2 = _HERE.parents[1]            # .../HoloNew/HoloV2
_REPO = _HERE.parents[2]             # .../HoloNew  (outer git repo: holds HoloV2/ and the V1 HoloNew/)

# The real ``paths.toml`` is machine-specific (gitignored); fall back to the committed template so a
# fresh checkout imports cleanly (its example paths are absent -> data-gated tests skip, never crash).
_TOML = _HOLOV2 / "paths.toml"
if not _TOML.exists():
    _TOML = _HOLOV2 / "paths.toml.example"
with open(_TOML, "rb") as _f:
    _CFG = tomllib.load(_f)


def _p(*keys: str) -> Path:
    """Nested toml lookup -> expanded Path (raises KeyError if a required key is absent)."""
    node = _CFG
    for k in keys:
        node = node[k]
    return Path(node).expanduser()


def _opt(*keys: str) -> Path | None:
    try:
        return _p(*keys)
    except KeyError:
        return None


# --- models (the SMPL-X / SMPL-H model dirs) ---
SMPLX_MODELS = _p("models", "smplx")
SMPLH_MODELS = _p("models", "smplh")

# --- dataset roots ---
HODOME = _p("datasets", "hodome", "motion")          # HODome release root (holds smplx/, object/)
SFU = _p("datasets", "sfu", "motion")                # SFU release root (<subject>/<name>.npz)
OMOMO_NEW = _p("datasets", "omomo", "motion")        # InterMimic .pt sequences
OMOMO = _p("datasets", "omomo", "meta")              # OMOMO release root (betas pickle)
HOIM3 = _p("datasets", "hoim3", "motion")            # HOI-M3 mocap_ground
PAHOI = _opt("datasets", "pahoi", "motion")          # PA-HOI Mocap_data root (None si absent)

# HOI-M3 auxiliary (usually absent on disk -> the gated tests skip): take the toml value if present,
# else derive the conventional location under the top-level models dir so `.exists()` simply returns
# False without the test crashing.
_MODELS_ROOT = SMPLX_MODELS.parents[2]               # .../models  (smplx is models/models_smplx_v1_1/models/smplx)
MODEL_TRANSFER = _opt("models", "model_transfer") or (_MODELS_ROOT / "model_transfer")
SMPL2SMPLX = _opt("models", "smpl2smplx") or (MODEL_TRANSFER / "smpl2smplx_deftrafo_setup.pkl")

# --- repo-internal (travel with the checkout, NOT external) ---
V1_TEST_SOCP = _REPO / "HoloNew" / "src" / "test_socp"
DEMO_DATA = _REPO / "HoloNew" / "demo_data"

# Assets HoloV2-internes fixés dans le checkout (ni machine-spécifiques, ni dans paths.toml)
CORR_NEUTRAL = _HOLOV2 / "cache" / "correspondence" / "corr_neutral.npz"   # appariement OT figé
G1_URDF      = _HOLOV2 / "models" / "g1" / "g1_29dof.urdf"                 # URDF robot G1
