"""The viewer's Grounded smplx skeleton must be grounded over the FULL sequence (like
the solve's gmr_grounded), not over the displayed [:max_frames] window. Otherwise the
displayed human is Z-shifted from the contact cloud whenever the sequence's lowest foot
falls outside the window (e.g. --max_frames 20 on a long HODome clip)."""
import numpy as np

from HoloNew.examples.view_stages import grounded_smplx_skeleton


def test_grounding_uses_full_sequence_not_window():
    # 5 frames, 2 joints (pelvis=0, foot=1). The foot dips lowest at frame 4, AFTER the
    # 2-frame display window. All z < mat_height so no mat adjustment muddies the check.
    raw = np.zeros((5, 2, 3), np.float32)
    raw[:, 0, 2] = 1.0
    raw[:, 1, 2] = [0.05, 0.05, 0.05, 0.05, -0.05]
    toe = [1]

    g = grounded_smplx_skeleton(raw, toe, 2)
    assert g.shape == (2, 2, 3)
    # Full-sequence z_min = -0.05, so every joint is shifted UP by 0.05.
    np.testing.assert_allclose(g[0, 0, 2], 1.0 - (-0.05), atol=1e-6)     # pelvis
    np.testing.assert_allclose(g[:, 1, 2], np.array([0.05, 0.05]) - (-0.05), atol=1e-6)  # foot
    # Sanity: the buggy window-only grounding would have used z_min=0.05 -> pelvis 0.95.
    assert abs(float(g[0, 0, 2]) - 0.95) > 1e-3
