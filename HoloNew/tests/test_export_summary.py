import json
import numpy as np
from HoloNew.evaluation.export.summary import reduce_channel, write_summary


def test_reduce_channel_scalars():
    r = reduce_channel(np.array([0.0, 3.0, 4.0]))
    assert r["mean"] == 7.0 / 3.0
    assert r["min"] == 0.0
    assert r["max"] == 4.0
    np.testing.assert_allclose(r["rms"], np.sqrt((0 + 9 + 16) / 3))


def test_reduce_channel_empty_is_zeros():
    r = reduce_channel(np.array([]))
    assert r == {"mean": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}


def test_write_summary_json(tmp_path):
    path = tmp_path / "run_summary.json"
    write_summary(path, {"diag/foot_slip": np.array([0.0, 1.0])})
    data = json.loads(path.read_text())
    assert data["diag/foot_slip"]["max"] == 1.0
