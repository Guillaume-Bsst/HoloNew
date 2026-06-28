"""Unit tests for ``eval_fields`` — the per-frame trilinear evaluation of a posed cloud against all
``Channel``s. Synthetic + analytic (no trimesh, no torch): the flat-ground SDF (``build_plane_sdf``)
is an exact affine field, so distance/witness/direction have closed forms to check against, and a
hand-placed object pose verifies the object-local frame mapping and that witness/direction come back
IN that local frame (no world round-trip).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import Channel
from src.prepare.sdf.build import build_plane_sdf
from src.targets.contracts import MultiChannelField
from src.targets.interaction import eval_fields


def _ground_channel(margin: float = 0.1) -> Channel:
    """Flat z=0 ground over a generous xy span (analytic plane SDF)."""
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=margin, name="ground")
    return Channel("ground", None, sdf)


# object_rot/object_pos are unused by a ground (object_idx is None) channel; pass empty (0, ...) arrays.
_NO_OBJ_ROT = np.zeros((0, 3, 3))
_NO_OBJ_POS = np.zeros((0, 3))


def test_ground_plane_distance_witness_direction():
    margin = 0.1
    ch = _ground_channel(margin)
    # probes at known heights (all within margin of z=0): three above, one penetrating below.
    points = np.array([
        [0.10, 0.20, 0.03],     # above the floor
        [-0.20, 0.10, 0.05],    # above the floor
        [0.30, -0.30, 0.00],    # exactly on the surface
        [0.15, 0.15, -0.02],    # penetrating (below the floor)
    ])
    field = eval_fields(points, (ch,), _NO_OBJ_ROT, _NO_OBJ_POS, margin)

    # distance == z (signed), exactly (the plane field is affine).
    assert np.allclose(field.distance[0], points[:, 2], atol=1e-5)
    # witness == (x, y, 0).
    assert np.allclose(field.witness[0, :, :2], points[:, :2], atol=1e-5)
    assert np.allclose(field.witness[0, :, 2], 0.0, atol=1e-5)
    # direction = (0, 0, sign(z)) — surface -> point; +z above, -z penetrating; 0 on-surface.
    assert np.allclose(field.direction[0, 0], [0, 0, 1], atol=1e-5)
    assert np.allclose(field.direction[0, 1], [0, 0, 1], atol=1e-5)
    assert np.allclose(field.direction[0, 2], [0, 0, 0], atol=1e-5)   # on-surface -> zero
    assert np.allclose(field.direction[0, 3], [0, 0, -1], atol=1e-5)  # penetrating -> -z
    # all four are within the band -> active.
    assert field.active[0].all()


def test_active_mask_respects_margin():
    margin = 0.1
    ch = _ground_channel(margin)
    points = np.array([
        [0.0, 0.0, 0.05],   # inside band  -> active
        [0.0, 0.0, 0.20],   # in grid (z<0.3) but z >= margin -> inactive
        [0.0, 0.0, 5.00],   # out of grid -> inactive
    ])
    field = eval_fields(points, (ch,), _NO_OBJ_ROT, _NO_OBJ_POS, margin)
    assert field.active[0].tolist() == [True, False, False]


def test_object_channel_local_frame_mapping():
    """A NON-identity object pose (rotation AND translation): the probe must be mapped to local and
    the witness/direction returned IN the object-local frame, NOT remapped back to world."""
    margin = 0.1
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=margin, name="obj0")
    ch = Channel("obj0", 0, sdf)

    rot = R.from_rotvec([0.3, -0.7, 1.1]).as_matrix()   # arbitrary rotation
    pos = np.array([1.2, -0.4, 2.5])                    # arbitrary translation
    object_rot = rot[None]                              # (1, 3, 3)
    object_pos = pos[None]                              # (1, 3)

    # Choose a point in object-LOCAL coords, then push it to world via the object pose.
    p_local_true = np.array([0.10, 0.20, 0.07])         # z=0.07 < margin -> active
    p_world = rot @ p_local_true + pos
    field = eval_fields(p_world[None], (ch,), object_rot, object_pos, margin)

    # distance is frame-invariant: equals the local z over the z=0 plane.
    assert np.isclose(field.distance[0, 0], 0.07, atol=1e-5)
    # witness is the LOCAL nearest point (x, y, 0), NOT world and NOT R@witness+t.
    assert np.allclose(field.witness[0, 0], [0.10, 0.20, 0.0], atol=1e-5)
    # direction is the LOCAL contact normal (0, 0, 1), distinct from the world normal R@(0,0,1).
    assert np.allclose(field.direction[0, 0], [0, 0, 1], atol=1e-5)
    world_normal = rot @ np.array([0.0, 0.0, 1.0])
    assert not np.allclose(field.direction[0, 0], world_normal, atol=1e-3)
    assert field.active[0, 0]


def test_inactive_probe_is_zeroed():
    margin = 0.1
    ch = _ground_channel(margin)
    points = np.array([
        [0.0, 0.0, 0.04],   # active reference
        [0.1, 0.1, 5.0],    # far -> out of grid -> inactive
    ])
    field = eval_fields(points, (ch,), _NO_OBJ_ROT, _NO_OBJ_POS, margin)
    assert not field.active[0, 1]
    assert field.distance[0, 1] == margin             # inactive distance clamps to +margin
    assert np.allclose(field.direction[0, 1], 0.0)    # zeroed
    assert np.allclose(field.witness[0, 1], 0.0)      # zeroed


def test_output_is_channel_first():
    margin = 0.1
    ground = _ground_channel(margin)
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=margin, name="obj0")
    obj = Channel("obj0", 0, sdf)
    object_rot = np.eye(3)[None]
    object_pos = np.zeros((1, 3))

    p = 7
    points = np.random.default_rng(0).uniform(-0.2, 0.2, size=(p, 3))
    field = eval_fields(points, (ground, obj), object_rot, object_pos, margin)

    assert isinstance(field, MultiChannelField)
    c = 2
    assert field.distance.shape == (c, p)
    assert field.direction.shape == (c, p, 3)
    assert field.witness.shape == (c, p, 3)
    assert field.active.shape == (c, p)
    assert field.channels == ("ground", "obj0")
    assert field.n_channels == c
    assert field.n_points == p
