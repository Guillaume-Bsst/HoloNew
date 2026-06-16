"""SMPL true-CoM reference: the per-part rigid-FK CoM must match the posed-mesh
centroid (the LBS ground truth) to within the pose-blendshape residual (~1 cm),
and must sit far from the pelvis joint (proving the pelvis proxy was wrong)."""
import numpy as np
import torch
import pytest
from scipy.spatial.transform import Rotation as R

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.smpl_com import calibrate_smpl_com, smpl_com_from_pose


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig()))


def test_smpl_com_matches_posed_mesh_centroid(rt):
    hb = rt.smplx_ground_probe.human_body
    calib = calibrate_smpl_com(hb)
    model = hb.model
    parents = calib.parents
    for t in [0, 10, 25]:
        q = rt.human_quat[t]
        # Pose the model exactly like placed_verts_smpl to get the ground-truth mesh + root.
        qx = np.zeros((parents.shape[0], 4)); qx[:, 3] = 1.0
        qx[:22] = np.asarray(q[:22])[:, [1, 2, 3, 0]]
        qx /= (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
        Rg = R.from_quat(qx).as_matrix()
        rel = np.matmul(np.transpose(Rg[parents], (0, 2, 1)), Rg); rel[parents == -1] = Rg[parents == -1]
        go = torch.from_numpy(R.from_matrix(rel[0]).as_rotvec()).float().view(1, 3)
        bp = torch.from_numpy(R.from_matrix(rel[1:22]).as_rotvec()).float().view(1, -1)
        with torch.no_grad():
            o = model(global_orient=go, body_pose=bp, betas=hb._betas,
                      return_verts=True, return_joints=True)
        verts = o.vertices[0].numpy()
        root = o.joints[0].numpy()[0]                      # posed pelvis = FK root
        com_truth = verts.mean(0)                          # uniform-density mesh centroid
        com = smpl_com_from_pose(calib, q, root)
        err = np.linalg.norm(com - com_truth)
        pelvis_gap = np.linalg.norm(com_truth - root)
        assert err < 0.012, f"t={t}: CoM off by {err*1000:.1f} mm vs mesh centroid"
        assert pelvis_gap > 0.2, f"t={t}: CoM should be far from pelvis, got {pelvis_gap*1000:.0f} mm"
