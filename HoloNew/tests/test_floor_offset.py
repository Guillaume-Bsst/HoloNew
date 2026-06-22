"""Robust floor correction: drop the human so the typical stance sole rests slightly
below z=0 (biased toward penetration for solid floor contact), not floating. The drop is
median(per-frame lowest sole) + a contact margin — robust to outlier penetrating/crouch
frames, constant per clip so it preserves vertical dynamics (jumps stay jumps)."""
import numpy as np

from HoloNew.src.test_socp.contact.smplx_field import robust_floor_offset


def test_median_plus_margin():
    soles = np.array([0.02, 0.02, 0.02, 0.04])     # median 0.02
    assert robust_floor_offset(soles, 0.01) == 0.03


def test_robust_to_penetrating_outlier():
    # One deeply-penetrating frame (imperfect SMPL) must NOT drag the drop down.
    soles = np.array([0.02, 0.02, 0.02, 0.04, -0.20])   # median still 0.02
    assert robust_floor_offset(soles, 0.01) == 0.03


def test_margin_biases_toward_penetration():
    # After dropping by the offset, the median stance sole sits at -margin (penetrating).
    soles = np.array([0.00, 0.02, 0.04])           # median 0.02
    off = robust_floor_offset(soles, 0.015)
    assert np.isclose(np.median(soles) - off, -0.015)


# --- integration: the builder applies the correction so soles rest on the floor ---
import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from HoloNew.src.paths import get_path  # noqa: E402

_HODOME = get_path("hodome") / "smplx" / "subject01_baseball.npz"
_SMPLX = get_path("smplx_models") / "smplx"
_HAVE = _HODOME.exists() and _SMPLX.is_dir() and (get_path("hodome") / "object" / "subject01_baseball.npz").exists()


@pytest.mark.skipif(not _HAVE, reason="HODome + SMPL-X assets not present")
def test_builder_grounds_soles_to_floor():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

    cfg = RetargetingConfig(dataset="hodome", motion_name="subject01_baseball",
                            task_type="object_interaction",
                            retargeter=TestSocpRetargeterConfig(floor_contact_margin=0.01))
    normalize_dataset_cfg(cfg)
    rt = TestSocpRetargeter.from_config(cfg)
    assert rt._floor_offset > 0.015          # a real downward correction was applied

    pr = rt.smplx_ground_probe
    q = rt._smplx_orientations
    samp = np.linspace(0, q.shape[0] - 1, 30).astype(int)
    mins = np.array([float(pr.human_body.placed_points(
        q[t], rt.gmr_grounded[t, 0], pr.cache, smpl_order=True)[:, 2].min()) for t in samp])
    med = float(np.median(mins))
    # Biased to slight penetration (~ -margin), and NOT floating a couple cm like before.
    assert -0.04 < med < 0.005, f"median sole z={med:.3f} not grounded to the floor"


@pytest.mark.skipif(not _HAVE, reason="HODome + SMPL-X assets not present")
def test_floor_correction_can_be_disabled():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

    cfg = RetargetingConfig(dataset="hodome", motion_name="subject01_baseball",
                            task_type="object_interaction",
                            retargeter=TestSocpRetargeterConfig(floor_contact_margin=None))
    normalize_dataset_cfg(cfg)
    rt = TestSocpRetargeter.from_config(cfg)
    assert rt._floor_offset == 0.0           # None -> legacy joint-only grounding
