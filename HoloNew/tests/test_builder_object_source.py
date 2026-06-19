import numpy as np
from types import SimpleNamespace

from HoloNew.src.test_socp.builder import resolve_object_inputs


def test_resolve_object_inputs_legacy_no_dataset(monkeypatch):
    # No dataset -> uses constants.OBJECT_MESH_FILE + .pt poses.
    cfg = SimpleNamespace(dataset=None, motion_path=None, obj_path=None, model_path=None,
                          task_type="object_interaction", motion_data_config=None)
    constants = SimpleNamespace(OBJECT_MESH_FILE="models/largebox/largebox.obj")
    called = {}

    def fake_pt(path):
        called["pt"] = path
        return None, np.zeros((5, 7))

    monkeypatch.setattr("HoloNew.src.test_socp.builder.load_intermimic_data", fake_pt)
    mesh, poses = resolve_object_inputs(cfg, constants, pt_path="seq.pt")
    assert str(mesh) == "models/largebox/largebox.obj"
    assert poses.shape == (5, 7) and called["pt"] == "seq.pt"


def test_resolve_object_inputs_dataset_empty():
    cfg = SimpleNamespace(dataset="sfu", motion_path="m.npz", obj_path=None, model_path=None,
                          task_type="robot_only", motion_data_config=None)
    mesh, poses = resolve_object_inputs(cfg, SimpleNamespace(OBJECT_MESH_FILE=None),
                                        pt_path=None)
    assert mesh is None and poses is None


def test_resolve_object_inputs_legacy_no_mesh():
    cfg = SimpleNamespace(dataset=None, motion_path=None, obj_path=None, model_path=None,
                          task_type="robot_only", motion_data_config=None)
    mesh, poses = resolve_object_inputs(cfg, SimpleNamespace(OBJECT_MESH_FILE=None),
                                        pt_path=None)
    assert mesh is None and poses is None
