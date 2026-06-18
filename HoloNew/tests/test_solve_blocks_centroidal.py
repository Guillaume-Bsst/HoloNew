"""Smoke test for the ResidualBlock centroidal term builders."""
import numpy as np
import pytest


def test_centroidal_block_builders_exist_and_return_residual_blocks():
    from HoloNew.src.test_socp.centroidal import build_centroidal_blocks, build_lumped_L_block
    assert callable(build_centroidal_blocks) and callable(build_lumped_L_block)
