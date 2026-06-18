"""Smoke tests for ResidualBlock builders in the D/X/P interaction terms."""
from __future__ import annotations


def test_interaction_block_builders_exist():
    from HoloNew.src.test_socp.interaction import build_dx_blocks, build_p_blocks
    assert callable(build_dx_blocks) and callable(build_p_blocks)
