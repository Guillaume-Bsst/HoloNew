"""Contrats vue-modèle — validation des formes + l'ensemble complet des champs SolvedFrame (phase B
les remplit; phase A laisse VizFrame.solved=None). Construits à partir des contrats minimaux du
pipeline réel (numpy-only, pas de viser)."""
import numpy as np
import pytest

from src.prepare.contracts import Channel, CorrespondenceTable, SDF
from src.targets.contracts import (FramePose, FrameTargets, MultiChannelField,
                                    RobotInteractionTargets, EnvironmentInteractionTargets, StyleTargets)
from src.viz.model import SolvedFrame, VizContext, VizFrame


def _sdf() -> SDF:
    return SDF(grid=np.zeros((2, 2, 2)), witness=np.zeros((2, 2, 2, 3)),
               origin=np.zeros(3), spacing=0.1, name="ground")


def _channels(n_objects: int = 0) -> tuple:
    """Channels minimaux : ground + N objets factices."""
    ground = Channel(name="ground", object_idx=None, sdf=_sdf(), geodesic=None)
    objs = tuple(
        Channel(name=f"obj{i}", object_idx=i, sdf=_sdf(), geodesic=None)
        for i in range(n_objects)
    )
    return (ground,) + objs


def _correspondence() -> CorrespondenceTable:
    """CorrespondenceTable minimale (1 point) pour les tests unitaires."""
    return CorrespondenceTable(
        smpl_idx=np.array([0], dtype=np.int64),
        link_idx=np.array([0], dtype=np.int64),
        offset_local=np.zeros((1, 3), dtype=np.float64),
        link_names=("link0",),
    )


def _ctx(n_objects: int = 0) -> VizContext:
    return VizContext(
        channel_names=tuple(["ground"] + [f"obj{i}" for i in range(n_objects)]),
        margin=0.05, style_link_names=("a", "b"),
        smpl_faces=np.zeros((4, 3), np.int64), smpl_parents=np.array([-1, 0, 1]),
        n_objects=n_objects,
        channels=_channels(n_objects),
        correspondence=_correspondence(),
        robot_urdf_path=__import__("pathlib").Path("/tmp/g1.urdf"),
        has_solve=False, ground_sdf=_sdf())


def _field(C: int = 1, P: int = 3) -> MultiChannelField:
    return MultiChannelField(
        distance=np.zeros((C, P)), direction=np.zeros((C, P, 3)),
        witness=np.zeros((C, P, 3)), active=np.zeros((C, P), bool),
        channels=tuple(f"c{i}" for i in range(C)))


def _targets() -> FrameTargets:
    style = StyleTargets(link_names=("a", "b"), position=np.zeros((2, 3)),
                         orientation=np.tile([1.0, 0, 0, 0], (2, 1)))
    return FrameTargets(style=style,
                        robot_interaction=RobotInteractionTargets(field=_field()),
                        env_interaction=EnvironmentInteractionTargets(per_object=()),
                        object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))


def _pose() -> FramePose:
    return FramePose(bone_rot=np.zeros((3, 3, 3)), bone_pos=np.zeros((3, 3)),
                     object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))


def test_context_ok_and_channel_count():
    ctx = _ctx(n_objects=2)
    assert ctx.channel_names == ("ground", "obj0", "obj1")
    assert ctx.has_solve is False


def test_context_bad_channel_count_raises():
    with pytest.raises(ValueError):
        VizContext(channel_names=("ground",), margin=0.05, style_link_names=(),
                   smpl_faces=np.zeros((4, 3), np.int64), smpl_parents=np.array([-1]),
                   n_objects=2,  # 2 objets mais seulement 1 nom de canal -> désaccord
                   channels=_channels(2), correspondence=_correspondence(),
                   robot_urdf_path=__import__("pathlib").Path("/x"), has_solve=False, ground_sdf=_sdf())


def test_context_bad_faces_raises():
    with pytest.raises(ValueError):
        VizContext(channel_names=("ground",), margin=0.05, style_link_names=(),
                   smpl_faces=np.zeros((4, 4), np.int64), smpl_parents=np.array([-1]), n_objects=0,
                   channels=_channels(0), correspondence=_correspondence(),
                   robot_urdf_path=__import__("pathlib").Path("/x"), has_solve=False, ground_sdf=_sdf())


def test_vizframe_solved_none_ok():
    fr = VizFrame(pose=_pose(), smpl_verts_world=np.zeros((5, 3), np.float32),
                  human_cloud_world=np.zeros((3, 3), np.float32), object_clouds_world=(),
                  human_field=_field(), targets=_targets(), solved=None)
    assert fr.solved is None and fr.smpl_verts_world.shape == (5, 3)


def test_vizframe_bad_cloud_raises():
    with pytest.raises(ValueError):
        VizFrame(pose=_pose(), smpl_verts_world=None,
                 human_cloud_world=np.zeros((3, 2), np.float32),  # pas (N, 3)
                 object_clouds_world=(), human_field=_field(), targets=_targets(), solved=None)


def test_solvedframe_full_field_set():
    sf = SolvedFrame(q=np.zeros(35), object_poses=np.zeros((1, 7)),
                     robot_points_world=np.zeros((10, 3)), link_transforms=np.zeros((4, 4, 4)),
                     style_achieved=None, contact_achieved=None,
                     cost=1.0, cost_by_term={"S-pos": 0.5}, n_iters=3, status="optimal")
    assert sf.q.shape == (35,) and sf.object_poses.shape == (1, 7)
    assert sf.cost_by_term["S-pos"] == 0.5 and sf.status == "optimal"
