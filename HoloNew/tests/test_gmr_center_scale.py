"""compute_stages root_xy_scale: pull the GMR skeleton's root toward the world
origin by a global factor (to match holosoma), as a rigid XY translation."""
import importlib

import numpy as np
import pytest

MODS = ["HoloNew.src.gmr_socp.preprocess", "HoloNew.src.test_socp.preprocess"]


def _synthetic(mod):
    tables = importlib.import_module(mod.rsplit(".", 1)[0] + ".tables")
    n_joints = 52
    T = 4
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(T, n_joints, 3)).astype(np.float32)
    # Put the pelvis at a clearly off-origin XY so scaling toward center is visible.
    pelvis_idx = tables.HUMAN_BODY_TO_IDX[tables.HUMAN_ROOT_NAME]
    pos[:, pelvis_idx, :2] += np.array([3.0, -2.0], np.float32)
    quats = np.zeros((T, n_joints, 4), np.float32)
    quats[..., 0] = 1.0  # identity wxyz
    return tables, pos, quats, pelvis_idx


@pytest.mark.parametrize("mod", MODS)
def test_default_scale_preserves_raw_pelvis_xy(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    out = cs(pos, quats, anchor_root_xy=True)  # root_xy_scale defaults to 1.0
    pelvis_bi = tables.MAPPED_BODY_NAMES.index(tables.HUMAN_ROOT_NAME)
    raw_xy = pos[:, pelvis_idx, :2]
    for s in out:
        np.testing.assert_allclose(out[s]["pos"][:, pelvis_bi, :2], raw_xy, rtol=1e-5)


@pytest.mark.parametrize("mod", MODS)
def test_root_xy_scale_pulls_pelvis_toward_center(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    scale = 0.5
    out = cs(pos, quats, anchor_root_xy=True, root_xy_scale=scale)
    pelvis_bi = tables.MAPPED_BODY_NAMES.index(tables.HUMAN_ROOT_NAME)
    expected_xy = pos[:, pelvis_idx, :2] * scale
    for s in out:
        np.testing.assert_allclose(out[s]["pos"][:, pelvis_bi, :2], expected_xy, rtol=1e-5)


@pytest.mark.parametrize("mod", MODS)
def test_scaling_is_rigid_xy_translation_only(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    scale = 0.5
    base = cs(pos, quats, anchor_root_xy=True, root_xy_scale=1.0)
    pulled = cs(pos, quats, anchor_root_xy=True, root_xy_scale=scale)
    raw_xy = pos[:, pelvis_idx, :2]
    expected_shift = (raw_xy * scale - raw_xy)[:, None, :]  # (T, 1, 2)
    for s in base:
        delta = pulled[s]["pos"][:, :, :2] - base[s]["pos"][:, :, :2]
        # Same XY shift for every body (rigid), and Z untouched.
        np.testing.assert_allclose(delta, np.broadcast_to(expected_shift, delta.shape), atol=1e-5)
        np.testing.assert_allclose(pulled[s]["pos"][:, :, 2], base[s]["pos"][:, :, 2], atol=1e-6)
