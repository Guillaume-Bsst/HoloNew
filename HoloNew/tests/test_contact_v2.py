"""Task 8: verify that TestSocpRetargeter.from_config loads the bundled
object SDF and demo contact field into the retargeter.

No coal / SMPL-X required — uses only the precomputed .npz artefacts
under assets/contact/.
"""
from __future__ import annotations


def test_v2_loads_bundled_contact():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.contact.contact_field import ContactField

    rt = TestSocpRetargeter.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh")
    )
    assert rt.object_sdf is not None
    assert isinstance(rt.contact_fields, dict)
    assert "human_object" in rt.contact_fields
    assert isinstance(rt.contact_fields["human_object"], ContactField)
