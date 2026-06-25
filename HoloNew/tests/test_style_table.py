"""STYLE_WEIGHT_TABLE: independent intra-style distribution (Brick 2)."""
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_style_table_exists_and_has_pelvis_tilt():
    from HoloNew.src.test_socp.tables import STYLE_WEIGHT_TABLE
    assert "__pelvis_tilt__" in STYLE_WEIGHT_TABLE
    assert all(v >= 0 for v in STYLE_WEIGHT_TABLE.values())
    assert sum(STYLE_WEIGHT_TABLE.values()) > 0


def test_style_table_keys_match_bodies(rt):
    """Every rotation-tracked body that build_style_blocks iterates must have a
    table entry, so the table-driven path produces the same NON-ZERO set of
    blocks as the legacy path."""
    from HoloNew.src.test_socp.tables import STYLE_WEIGHT_TABLE
    from HoloNew.src.test_socp.style import build_style_blocks
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE_SINGLE
    from HoloNew.src.test_socp.targets import ground_frame_targets
    gpos, gquat = rt.gmr_floor["pos"], rt.gmr_floor["quat"]
    ft = ground_frame_targets(gpos[0], gquat[0], IK_MATCH_TABLE_SINGLE)
    q = rt.q_init_full[:36]
    legacy = build_style_blocks(rt, q, ft, lambda_ws=1.0, sigma_R=1.0, style_weights=None)
    table = build_style_blocks(rt, q, ft, lambda_ws=1.0, sigma_R=1.0, style_weights=STYLE_WEIGHT_TABLE)
    # Uniform table must produce the SAME number of blocks as legacy (every tracked body covered).
    assert len(table) == len(legacy) and len(table) > 0


def test_rt_uses_table_by_default(rt):
    from HoloNew.src.test_socp.tables import STYLE_WEIGHT_TABLE
    assert getattr(rt, "style_weights", None) == STYLE_WEIGHT_TABLE
