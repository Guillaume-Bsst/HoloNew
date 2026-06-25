"""The ground stage is the single source of truth for the object pose across the whole
pipeline (SDF, interaction, movable). The viewer must reflect this: all stages show the
grounded object (from _obj_poses_raw) so there is no ambiguity about which object pose
is 'real'. Only 'Original' stays raw as a baseline reference of the unprocessed capture.

Regressions covered:
- Chair sinks into floor with 'Solved object pose' OFF on 'Grounded' stage.
- Chair sinks into floor on 'Mapped' stage (raw object, not ground-stage object)."""
import types

import numpy as np

from HoloNew.examples.view_stages import OBJECT_GROUNDED_STAGES
from HoloNew.src.viewer import Viewer


def _fake_viewer(stage_set, grounded, raw):
    # Minimal stand-in exposing exactly what Viewer._object_pose reads.
    method = types.SimpleNamespace(object_pose_scaled=grounded)
    return types.SimpleNamespace(
        object_scaled_stages=frozenset(stage_set),
        _methods={"TEST-SOCP": method},
        _method_dd=types.SimpleNamespace(value="TEST-SOCP"),
        object_pose_scaled=None,
        object_pose_raw=raw)


def test_grounded_stage_shows_grounded_object():
    grounded = np.full((4, 7), 1.0)
    raw = np.zeros((4, 7))
    fv = _fake_viewer(OBJECT_GROUNDED_STAGES, grounded, raw)
    out = Viewer._object_pose(fv, "Grounded")
    np.testing.assert_array_equal(out, grounded)


def test_original_stage_stays_raw():
    grounded = np.full((4, 7), 1.0)
    raw = np.zeros((4, 7))
    fv = _fake_viewer(OBJECT_GROUNDED_STAGES, grounded, raw)
    out = Viewer._object_pose(fv, "Original")
    np.testing.assert_array_equal(out, raw)


def test_mapped_stage_shows_grounded_object():
    grounded = np.full((4, 7), 1.0)
    raw = np.zeros((4, 7))
    fv = _fake_viewer(OBJECT_GROUNDED_STAGES, grounded, raw)
    out = Viewer._object_pose(fv, "Mapped")
    np.testing.assert_array_equal(out, grounded)


def test_scaled_stages_still_grounded():
    grounded = np.full((4, 7), 1.0)
    raw = np.zeros((4, 7))
    fv = _fake_viewer(OBJECT_GROUNDED_STAGES, grounded, raw)
    for stage in ("Scaled", "Offset", "Floor"):
        out = Viewer._object_pose(fv, stage)
        np.testing.assert_array_equal(out, grounded)
