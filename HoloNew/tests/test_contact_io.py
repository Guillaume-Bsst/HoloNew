import numpy as np
from HoloNew.src.test_socp.contact.contact_field import ContactField
from HoloNew.src.test_socp.contact.contact_io import save_contact_fields, load_contact_fields


def _cf(T, N):
    return ContactField(distance=np.zeros((T, N), np.float32), direction=np.zeros((T, N, 3), np.float32),
                        witness=np.zeros((T, N, 3), np.float32), active=np.zeros((T, N), bool))


def test_contact_fields_roundtrip(tmp_path):
    fields = {"human_floor": _cf(3, 5), "human_object": _cf(3, 5)}
    p = tmp_path / "c.npz"
    save_contact_fields(p, fields)
    r = load_contact_fields(p)
    assert set(r) == {"human_floor", "human_object"}
    assert r["human_floor"].distance.shape == (3, 5)
    assert isinstance(r["human_object"], ContactField)


def test_motion_loads_demo_shapes():
    from HoloNew.src.test_socp.contact.motion import load_pt_motion
    joints, obj_poses, quats = load_pt_motion("demo_data/OMOMO_new/sub3_largebox_003.pt")
    assert joints.shape[1:] == (52, 3)
    assert obj_poses.shape[1] == 7
    assert quats.shape[1:] == (52, 4)
    assert joints.shape[0] == obj_poses.shape[0] == quats.shape[0]
