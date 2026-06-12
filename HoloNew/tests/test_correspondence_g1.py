import numpy as np
import yourdfpy
from HoloNew.src.gmr_socp_v2.correspondence.constants import G1_29DOF_URDF
from HoloNew.src.gmr_socp_v2.correspondence.g1_surface import sample_g1_surface, build_rest_cfg

def test_g1_surface_samples_valid_links():
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    surf = sample_g1_surface(urdf, density=300.0)   # low density = fast
    M = surf.points_world.shape[0]
    assert M > 0
    assert surf.points_world.shape == (M, 3) and surf.offset_local.shape == (M, 3)
    assert surf.link_idx.min() >= 0 and surf.link_idx.max() < len(surf.link_names)
    assert surf.seg.min() >= 0 and surf.seg.max() <= 14

def test_build_rest_cfg_length():
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    cfg = build_rest_cfg(urdf)
    assert cfg.shape[0] == len(urdf.actuated_joint_names)
