def test_pinocchio_imports_and_has_freeflyer():
    import pinocchio as pin
    assert hasattr(pin, "JointModelFreeFlyer")
    assert hasattr(pin, "integrate")
    assert hasattr(pin, "jacobianCenterOfMass")
