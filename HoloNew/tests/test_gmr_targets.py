import numpy as np
from HoloNew.src.gmr_socp_v1.targets import build_frame_targets
from HoloNew.src.gmr_socp_v1.tables import IK_MATCH_TABLE1


def test_build_frame_targets_applies_offsets_and_maps_frames():
    J = 52
    pos = np.zeros((J, 3)); pos[0] = [1.0, 2.0, 3.0]           # pelvis position
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (J, 1))    # identity wxyz
    targets = build_frame_targets(pos, quat, IK_MATCH_TABLE1)
    assert set(targets) == set(IK_MATCH_TABLE1)
    p, R, w_p, w_r = targets["pelvis"]
    np.testing.assert_allclose(p, [1.0, 2.0, 3.0], atol=1e-9)
    assert R.shape == (3, 3)
    assert (w_p, w_r) == (IK_MATCH_TABLE1["pelvis"][1], IK_MATCH_TABLE1["pelvis"][2])


def test_load_pt_quaternions_demo_shape():
    from HoloNew.src.gmr_socp_v1.targets import load_pt_quaternions
    q = load_pt_quaternions("demo_data/OMOMO_new/sub3_largebox_003.pt")
    assert q.ndim == 3 and q.shape[1] == 52 and q.shape[2] == 4
    norms = np.linalg.norm(q.reshape(-1, 4), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
