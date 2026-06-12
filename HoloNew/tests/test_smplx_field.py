import numpy as np

from HoloNew.src.test_socp.contact.smplx_field import SmplxGroundProbe


class _FakeBody:
    """placed_points returns two fixed points translated onto the pelvis target."""
    def __init__(self):
        self.calls = []

    def placed_points(self, quats_wxyz, pelvis_target, cache, frame_idx=None):
        self.calls.append((frame_idx, np.asarray(pelvis_target, float).copy()))
        return (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + np.asarray(pelvis_target, float))


class _FakeSDF:
    def __init__(self):
        self.last = None

    def query(self, pts_local, margin):
        self.last = (np.asarray(pts_local, float).copy(), margin)
        return "FIELD"


def _probe(T, body, sdf, margin=0.1):
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))          # identity wxyz
    trans = np.arange(T * 3, dtype=float).reshape(T, 3) + 5.0
    return SmplxGroundProbe(human_body=body, cache=None, object_sdf=sdf,
                            obj_quat=quat, obj_trans=trans, margin=margin), trans


def test_probe_places_then_queries_in_object_local_frame():
    body, sdf = _FakeBody(), _FakeSDF()
    probe, trans = _probe(3, body, sdf, margin=0.1)
    pelvis = np.array([2.0, 3.0, 1.0])
    out = probe(1, quats_wxyz=np.zeros((52, 4)), pelvis_grounded=pelvis)

    world = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + pelvis
    expected_local = world - trans[1]            # identity rot -> world_to_local subtracts trans[t]
    np.testing.assert_allclose(sdf.last[0], expected_local, atol=1e-6)
    assert sdf.last[1] == 0.1                     # margin forwarded
    assert out == "FIELD"                         # ContactField returned as-is
    assert body.calls[-1][0] == 1                 # frame_idx forwarded to placed_points


def test_probe_is_causal_reads_only_frame_t():
    body, sdf = _FakeBody(), _FakeSDF()
    probe, trans = _probe(4, body, sdf, margin=0.2)
    probe(2, np.zeros((52, 4)), np.zeros(3))       # frame 2, pelvis at origin
    world = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(sdf.last[0], world - trans[2], atol=1e-6)   # used obj pose t=2 only
