import numpy as np
from HoloNew.src.holosoma.preprocess import compute_holosoma_stages, ground_to_floor


def test_ground_to_floor_drops_lowest_toe_to_zero():
    raw = np.zeros((3, 52, 3), float)
    raw[:, :, 2] = 2.0
    raw[:, 3, 2] = 1.4; raw[:, 7, 2] = 1.5   # toes lowest
    before = raw.copy()
    out = ground_to_floor(raw, toe_indices=[3, 7], mat_height=0)
    np.testing.assert_allclose(out[:, 3, 2].min(), 0.0, atol=1e-6)   # lowest toe on floor
    np.testing.assert_allclose(out[:, :, 2], raw[:, :, 2] - 1.4)     # uniform z drop
    np.testing.assert_array_equal(raw, before)                       # input untouched


def test_holosoma_stages_shapes_and_steps():
    T, J = 4, 52
    raw = np.zeros((T, J, 3), float)
    raw[:, :, 2] = 1.0                 # all joints at z=1
    raw[:, 3, 2] = 0.5; raw[:, 7, 2] = 0.6   # toes (indices 3,7) lowest
    out = compute_holosoma_stages(raw, scale=0.5, toe_indices=[3, 7],
                                  mapped_indices=[0, 1, 2, 3, 4], mat_height=0)
    assert set(out) == {"Original", "Grounded", "Scaled", "Mapped"}
    assert out["Original"].shape == (T, J, 3)
    # Grounded: lowest toe (z=0.5) moved to 0
    np.testing.assert_allclose(out["Grounded"][:, 3, 2], 0.0, atol=1e-6)
    # Scaled = grounded * 0.5
    np.testing.assert_allclose(out["Scaled"], out["Grounded"] * 0.5)
    # Mapped = scaled at the 5 mapped indices
    assert out["Mapped"].shape == (T, 5, 3)
    np.testing.assert_allclose(out["Mapped"], out["Scaled"][:, [0, 1, 2, 3, 4]])
