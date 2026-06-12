"""Parity: HoloNew's native retargeter vs vanilla holosoma_retargeting.

HoloNew's `interaction_mesh_retargeter` is holosoma's retargeter with the
increment-1 modular-viz changes only (Viewer extraction, RetargetResult return);
the SOCP solver is byte-for-byte unchanged. This test pins that the *output* is
identical to upstream holosoma.

References (both for sub3_largebox_003, robot_only, smplh):
- tests/golden/baseline_qpos.npz       -> HoloNew native qpos (the golden test
  `test_retarget_golden.py` guarantees the live HoloNew native equals this).
- tests/golden/holosoma_vanilla_qpos.npz -> qpos produced by the UPSTREAM
  holosoma_retargeting example, run in the `hsretargeting` conda env.

Regenerate the vanilla reference (only if upstream holosoma changes):
    cd modules/third_party/holosoma/src/holosoma_retargeting/holosoma_retargeting
    <hsretargeting-python> examples/robot_retarget.py --data_path demo_data/OMOMO_new \
        --task-type robot_only --task-name sub3_largebox_003 --data_format smplh \
        --save_dir /tmp/vanilla && cp /tmp/vanilla/sub3_largebox_003.npz \
        <HoloNew>/tests/golden/holosoma_vanilla_qpos.npz
"""
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_HOLONEW = _HERE / "golden" / "baseline_qpos.npz"
_VANILLA = _HERE / "golden" / "holosoma_vanilla_qpos.npz"


@pytest.mark.skipif(not _VANILLA.exists(), reason="vanilla holosoma reference not present")
def test_native_qpos_matches_vanilla_holosoma():
    holonew = np.load(_HOLONEW)["qpos"]
    vanilla = np.load(_VANILLA)["qpos"]

    assert holonew.shape == vanilla.shape, (holonew.shape, vanilla.shape)
    max_diff = float(np.max(np.abs(holonew - vanilla)))
    print(f"\nnative vs vanilla holosoma: max|diff| = {max_diff:.3e}  shape={holonew.shape}")

    # The modular-viz refactor must not perturb the solver output at all.
    np.testing.assert_allclose(holonew, vanilla, atol=1e-6)
