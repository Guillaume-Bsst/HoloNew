"""Tests for the per-frame contact-field driver (combined.py)."""
import numpy as np
import trimesh
from HoloNew.src.test_socp.contact.backends.sdf import build_object_field
from HoloNew.src.test_socp.contact.combined import compute_contact_fields


def test_compute_contact_fields_sdf_mode():
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    sdf = build_object_field(box, margin=0.05, resolution=0.02)
    T, N = 2, 6
    hverts = np.zeros((T, N, 3), float)
    hprobes = np.random.default_rng(0).standard_normal((T, N, 3)) * 0.1
    quats = np.tile(np.array([1.0, 0, 0, 0]), (T, 52, 1))
    pelvises = np.zeros((T, 3))
    obj_poses = np.tile(np.array([1.0, 0, 0, 0, 0, 0, 0]), (T, 1))
    out = compute_contact_fields(
        T=T, quats=quats, pelvises=pelvises, human_faces=np.zeros((0, 3), int),
        human_body_params=None, human_pc_cache=None, object_mesh=box,
        object_grid_local=None, obj_poses=obj_poses,
        floor_grid=np.zeros((4, 3), float), margin=0.05,
        hverts=hverts, hprobes=hprobes, object_sdf=sdf,
    )
    assert set(out) == {"human_floor", "human_object"}
    assert out["human_floor"].distance.shape == (T, N)
    assert out["human_object"].distance.shape == (T, N)
