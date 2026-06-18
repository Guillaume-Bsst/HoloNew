"""Smoke test for the ResidualBlock movable-object term builders."""
import pytest


def test_movable_block_builders_exist():
    from HoloNew.src.test_socp.movable import (
        build_wo_block,
        build_wo_position_anchor_block,
        build_object_floor_blocks,
        build_object_floor_persistence_blocks,
    )
    for f in (
        build_wo_block,
        build_wo_position_anchor_block,
        build_object_floor_blocks,
        build_object_floor_persistence_blocks,
    ):
        assert callable(f)
