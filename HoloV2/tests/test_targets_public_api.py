"""The targets package re-exports the pure interaction kernel reused by solve, so downstream imports
the PACKAGE surface (from src.targets import ...), never the internal submodule."""


def test_kernel_is_importable_from_package():
    from src.targets import pose_cloud, eval_fields, MultiChannelField
    from src.targets.interaction import pose_cloud as _pc
    from src.targets.interaction.fields import eval_fields as _ef
    from src.targets.contracts import MultiChannelField as _mcf
    assert pose_cloud is _pc           # same object, just re-exported at the package surface
    assert eval_fields is _ef
    assert MultiChannelField is _mcf


def test_public_surface_exports_evaluator_and_eval_types():
    from src.targets import Evaluator, StyleEval, ContactEval, ContactEnvEval
    from src.targets.evaluator import Evaluator as _Ev
    from src.targets.contracts import (StyleEval as _SE, ContactEval as _CE,
                                       ContactEnvEval as _CEE)
    assert Evaluator is _Ev                # same object, re-exported at the package surface
    assert StyleEval is _SE and ContactEval is _CE and ContactEnvEval is _CEE
