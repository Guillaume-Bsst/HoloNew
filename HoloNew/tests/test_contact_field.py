import numpy as np
from HoloNew.src.test_socp.contact.contact_field import ContactField
from HoloNew.src.test_socp.contact.backends.floor import floor_field

def test_floor_field_signs_and_active():
    pts = np.array([[0, 0, -0.01], [0, 0, 0.02], [0, 0, 1.0]], float)
    f = floor_field(pts, margin=0.05)
    assert f.active.tolist() == [True, True, False]
    assert f.distance[2] == np.float32(0.05)
    assert f.direction[0, 2] == -1.0 and f.direction[1, 2] == 1.0

def test_contact_field_is_frozen_dataclass():
    f = ContactField(distance=np.zeros(2), direction=np.zeros((2, 3)),
                     witness=np.zeros((2, 3)), active=np.zeros(2, bool))
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.distance = np.ones(2)
