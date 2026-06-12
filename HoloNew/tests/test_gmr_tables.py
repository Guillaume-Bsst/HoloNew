from HoloNew.src.gmr_socp_v1.tables import (
    IK_MATCH_TABLE1, IK_MATCH_TABLE2, HUMAN_BODY_TO_IDX,
)
from HoloNew.config_types.data_type import SMPLH_DEMO_JOINTS

_EXPECTED = {
    "pelvis": "Pelvis", "left_hip": "L_Hip", "left_knee": "L_Knee",
    "left_foot": "L_Ankle", "right_hip": "R_Hip", "right_knee": "R_Knee",
    "right_foot": "R_Ankle", "spine3": "Chest",
    "left_shoulder": "L_Shoulder", "left_elbow": "L_Elbow", "left_wrist": "L_Wrist",
    "right_shoulder": "R_Shoulder", "right_elbow": "R_Elbow", "right_wrist": "R_Wrist",
}

def test_human_body_indices_match_smplh_demo_joints():
    for body, idx in HUMAN_BODY_TO_IDX.items():
        assert SMPLH_DEMO_JOINTS[idx] == _EXPECTED[body], (body, idx, SMPLH_DEMO_JOINTS[idx])

def test_tables_have_same_robot_frames():
    assert set(IK_MATCH_TABLE1) == set(IK_MATCH_TABLE2)

def test_table_row_shape():
    for table in (IK_MATCH_TABLE1, IK_MATCH_TABLE2):
        for frame, (human, pos_w, rot_w, pos_off, rot_off) in table.items():
            assert human in HUMAN_BODY_TO_IDX
            assert len(pos_off) == 3 and len(rot_off) == 4
