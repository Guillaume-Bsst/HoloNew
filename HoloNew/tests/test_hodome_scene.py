from pathlib import Path
import trimesh
import mujoco
from HoloNew.src.data_loaders.hodome_scene import build_hodome_scene_xml

_G1 = Path("models/g1/g1_29dof.xml")


def test_scene_content(tmp_path):
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh)
    out = tmp_path / "scene.xml"
    p = build_hodome_scene_xml(_G1, "baseball", mesh, output_path=out)
    txt = Path(p).read_text()
    assert '<mesh name="baseball_mesh"' in txt
    assert str(mesh.resolve()) in txt              # absolute object-mesh path
    assert '<body name="baseball_link">' in txt
    assert "<freejoint/>" in txt
    assert 'diaginertia="0.002 0.002 0.002"' in txt


def test_scene_parses_and_adds_free_joint(tmp_path):
    # Written next to the robot meshes so meshdir="assets/" resolves; cleaned up after.
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(mesh)
    base_nq = mujoco.MjModel.from_xml_path(str(_G1)).nq
    out = _G1.with_name("g1_29dof_w_pytesttoken.xml")
    try:
        build_hodome_scene_xml(_G1, "pytesttoken", mesh, output_path=out)
        m = mujoco.MjModel.from_xml_path(str(out))
        assert m.nq == base_nq + 7                 # object free joint adds 7 qpos
    finally:
        out.unlink(missing_ok=True)
