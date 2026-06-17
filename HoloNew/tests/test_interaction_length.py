"""Interaction length L is a well-integrated config variable.

L_interaction is the master length: it drives the object-SDF bake band, the field query
margin (so distances are not clamped before L), and the per-channel activation range.
L_floor / L_object are optional per-channel overrides; sdf_resolution sets the grid voxel.
Runs with different L rebake + cache the object SDF on demand, keyed by (L, resolution)."""
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.contact.backends.sdf import sdf_surface_field


def _make(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003",
        data_format="smplh", retargeter=TestSocpRetargeterConfig(**kw)))


def test_L_interaction_master_drives_both_channels():
    rt = _make(L_interaction=0.30, sdf_resolution=0.03)
    if rt.smplx_ground_probe is None:
        pytest.skip("assets not present")
    assert rt.L_floor == pytest.approx(0.30)
    assert rt.L_object == pytest.approx(0.30)
    # Query margin tracks L so the field is not clamped before L.
    assert rt.smplx_ground_probe.margin >= 0.30 - 1e-9


def test_L_overrides_take_precedence_over_master():
    rt = _make(L_interaction=0.30, L_object=0.15, sdf_resolution=0.03)
    if rt.smplx_ground_probe is None:
        pytest.skip("assets not present")
    assert rt.L_object == pytest.approx(0.15)
    assert rt.L_floor == pytest.approx(0.30)


def test_larger_L_rebakes_a_wider_object_band():
    # Default L=0.10 uses the bundled band (~0.10); L=0.30 rebakes a wider band so a point
    # ~0.20 m off the surface goes from INACTIVE (out of band) to ACTIVE.
    rt_lo = _make(L_interaction=0.10)
    rt_hi = _make(L_interaction=0.30, sdf_resolution=0.03)
    if rt_lo.object_sdf is None or rt_hi.object_sdf is None:
        pytest.skip("object assets not present")
    assert float(rt_lo.object_sdf.data.max()) == pytest.approx(0.10, abs=1e-2)
    assert float(rt_hi.object_sdf.data.max()) == pytest.approx(0.30, abs=2e-2)

    # A point ~0.20 m beyond the +x face (object-local): active only under the wide band.
    p = np.array([[0.43, 0.0, 0.0]], float)
    f_lo = sdf_surface_field(p, rt_lo.object_sdf, margin=rt_lo.smplx_ground_probe.margin)
    f_hi = sdf_surface_field(p, rt_hi.object_sdf, margin=rt_hi.smplx_ground_probe.margin)
    assert not bool(f_lo.active[0]), "point should be outside the narrow (L=0.10) band"
    assert bool(f_hi.active[0]), "point should be inside the wide (L=0.30) band"
    assert 0.10 < float(f_hi.distance[0]) < 0.30
