# tests/test_intermimic_quats.py
from pathlib import Path

import numpy as np
from HoloNew.src.utils import load_intermimic_quats

_PT = Path("demo_data/OMOMO_new/sub3_largebox_003.pt")


def test_quats_shape_and_unit_norm():
    quats = load_intermimic_quats(str(_PT))
    assert quats.ndim == 3 and quats.shape[1:] == (52, 4)
    norms = np.linalg.norm(quats.reshape(-1, 4), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
