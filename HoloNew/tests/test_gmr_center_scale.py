"""compute_stages scale_xy / scale_z: place the GMR skeleton's root in the world inside
the scale stage (raw_root_axis * value, per axis group). The scaled / offset / ground
stages carry the placement; the 'mapped' stage is the raw pre-scale bodies, untouched."""
import importlib

import numpy as np
import pytest

MODS = ["HoloNew.src.gmr_socp.preprocess", "HoloNew.src.test_socp.preprocess"]

# Stages produced by scale() (carry the placement); 'mapped' is the raw pre-scale bodies.
PLACED_STAGES = ("scaled", "offset", "floor")


def _synthetic(mod):
    tables = importlib.import_module(mod.rsplit(".", 1)[0] + ".tables")
    n_joints = 52
    T = 4
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(T, n_joints, 3)).astype(np.float32)
    # Put the pelvis at a clearly off-origin XY/Z so placement is visible.
    pelvis_idx = tables.HUMAN_BODY_TO_IDX[tables.HUMAN_ROOT_NAME]
    pos[:, pelvis_idx, :] += np.array([3.0, -2.0, 1.5], np.float32)
    quats = np.zeros((T, n_joints, 4), np.float32)
    quats[..., 0] = 1.0  # identity wxyz
    return tables, pos, quats, pelvis_idx


@pytest.mark.parametrize("mod", MODS)
def test_default_scale_preserves_raw_pelvis_xy(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    out = cs(pos, quats)  # scale_xy defaults to 1.0
    pelvis_bi = tables.MAPPED_BODY_NAMES.index(tables.HUMAN_ROOT_NAME)
    raw_xy = pos[:, pelvis_idx, :2]
    # At 1.0 every stage (mapped and placed) keeps the raw pelvis XY.
    for s in out:
        np.testing.assert_allclose(out[s]["pos"][:, pelvis_bi, :2], raw_xy, rtol=1e-5)


@pytest.mark.parametrize("mod", MODS)
def test_xy_scale_pulls_pelvis_toward_center(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    scale_xy = 0.5
    out = cs(pos, quats, scale_xy=scale_xy)
    pelvis_bi = tables.MAPPED_BODY_NAMES.index(tables.HUMAN_ROOT_NAME)
    raw_xy = pos[:, pelvis_idx, :2]
    # The placed stages sit at raw_xy * scale_xy ...
    for s in PLACED_STAGES:
        np.testing.assert_allclose(out[s]["pos"][:, pelvis_bi, :2], raw_xy * scale_xy, rtol=1e-5)
    # ... while 'mapped' (pre-scale raw bodies) keeps the raw pelvis XY.
    np.testing.assert_allclose(out["mapped"]["pos"][:, pelvis_bi, :2], raw_xy, rtol=1e-5)


@pytest.mark.parametrize("mod", MODS)
def test_xy_scaling_is_rigid_translation_only(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    scale_xy = 0.5
    base = cs(pos, quats, scale_xy=1.0)
    pulled = cs(pos, quats, scale_xy=scale_xy)
    raw_xy = pos[:, pelvis_idx, :2]
    expected_shift = (raw_xy * scale_xy - raw_xy)[:, None, :]  # (T, 1, 2)
    # Placed stages: the same rigid XY shift for every body, Z untouched.
    for s in PLACED_STAGES:
        delta = pulled[s]["pos"][:, :, :2] - base[s]["pos"][:, :, :2]
        np.testing.assert_allclose(delta, np.broadcast_to(expected_shift, delta.shape), atol=1e-5)
        np.testing.assert_allclose(pulled[s]["pos"][:, :, 2], base[s]["pos"][:, :, 2], atol=1e-6)
    # 'mapped' is pre-scale: identical regardless of scale_xy.
    np.testing.assert_allclose(pulled["mapped"]["pos"], base["mapped"]["pos"], atol=1e-6)


@pytest.mark.parametrize("mod", MODS)
def test_z_scale_places_pelvis_height(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    scale_z = 0.5
    out = cs(pos, quats, scale_z=scale_z)  # scale_xy stays default 1.0
    pelvis_bi = tables.MAPPED_BODY_NAMES.index(tables.HUMAN_ROOT_NAME)
    raw_z = pos[:, pelvis_idx, 2]
    # The 'scaled' stage (pre offset / ground-drop) places the root Z at raw_z * scale_z.
    np.testing.assert_allclose(out["scaled"]["pos"][:, pelvis_bi, 2], raw_z * scale_z, rtol=1e-5)
    # 'mapped' keeps the raw pelvis Z.
    np.testing.assert_allclose(out["mapped"]["pos"][:, pelvis_bi, 2], raw_z, rtol=1e-5)


@pytest.mark.parametrize("mod", MODS)
def test_z_scaling_is_rigid_translation_only(mod):
    cs = importlib.import_module(mod).compute_stages
    tables, pos, quats, pelvis_idx = _synthetic(mod)
    # Compare two explicit Z placements so the rigid Z shift is unambiguous (the
    # ground stage re-drops the min z, so check the pre-ground 'scaled' / 'offset').
    a = cs(pos, quats, scale_z=1.0)
    b = cs(pos, quats, scale_z=0.5)
    raw_z = pos[:, pelvis_idx, 2]
    expected_shift = (raw_z * 0.5 - raw_z * 1.0)[:, None]  # (T, 1)
    for s in ("scaled", "offset"):
        delta = b[s]["pos"][:, :, 2] - a[s]["pos"][:, :, 2]
        np.testing.assert_allclose(delta, np.broadcast_to(expected_shift, delta.shape), atol=1e-5)
        # XY untouched by scale_z.
        np.testing.assert_allclose(b[s]["pos"][:, :, :2], a[s]["pos"][:, :, :2], atol=1e-6)
