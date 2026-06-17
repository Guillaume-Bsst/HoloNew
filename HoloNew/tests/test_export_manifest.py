import json
import numpy as np
from HoloNew.evaluation.export.manifest import build_manifest, write_manifest


def _channels():
    return {"tracking/mpjpe/LeftFoot": np.zeros(3),
            "smoothness/joint_accel/j0": np.zeros(3),
            "solver/cost": np.zeros(3)}


def test_build_manifest_counts_and_missing():
    m = build_manifest(_channels())
    assert m["families"]["tracking"]["present"] is True
    assert m["families"]["tracking"]["n_channels"] == 1
    assert m["families"]["contacts"]["present"] is False
    assert "contacts" in m["missing"]
    assert m["n_channels_total"] == 3


def test_manifest_records_errors():
    m = build_manifest(_channels(), errors={"contacts": "no object sdf"})
    assert m["families"]["contacts"]["error"] == "no object sdf"


def test_write_manifest_roundtrip(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_manifest(path, _channels(), errors={"contacts": "boom"})
    data = json.loads(path.read_text())
    assert data["families"]["smoothness"]["n_channels"] == 1
    assert data["families"]["contacts"]["error"] == "boom"
