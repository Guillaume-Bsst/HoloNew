"""Generate a robot+object MuJoCo scene xml for an arbitrary (HODome) object,
mirroring the bundled models/g1/g1_29dof_w_largebox.xml: a single convex-hull mesh
geom + free joint, fixed largebox-style inertia. Written next to the robot meshes so
the base MJCF meshdir resolves; the object mesh is referenced by absolute path."""
from __future__ import annotations

from pathlib import Path

from HoloNew.src.data_loaders.base import resolve_loader

# Object body block, copied verbatim from g1_29dof_w_largebox.xml (convex-hull mesh
# geom, fixed mass/inertia). {token} fills the object name.
_OBJECT_BLOCK = """    <body name="{token}_link">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
        <geom name="{token}" type="mesh" mesh="{token}_mesh"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>
    </body>
"""


def build_hodome_scene_xml(robot_xml_path, token, mesh_obj_path, output_path=None) -> Path:
    """Robot+object scene xml for `token`, mesh = `mesh_obj_path` (convex hull).

    Injects the object mesh asset (absolute path) before the first </asset> and the
    object body before </worldbody>. Default output is next to the robot xml as
    <robot_stem>_w_<token>.xml (the path the solver scene swap expects)."""
    robot_xml_path = Path(robot_xml_path)
    mesh_abs = str(Path(mesh_obj_path).resolve())
    content = robot_xml_path.read_text()

    asset = f'    <mesh name="{token}_mesh" file="{mesh_abs}" scale="1 1 1"/>\n'
    try:
        i = content.index("</asset>")               # first </asset> = the mesh-asset block
    except ValueError:
        raise ValueError(f"No </asset> in robot MJCF {robot_xml_path}") from None
    content = content[:i] + asset + content[i:]

    try:
        j = content.index("</worldbody>")
    except ValueError:
        raise ValueError(f"No </worldbody> in robot MJCF {robot_xml_path}") from None
    content = content[:j] + _OBJECT_BLOCK.format(token=token) + content[j:]

    if output_path is None:
        output_path = robot_xml_path.with_name(f"{robot_xml_path.stem}_w_{token}.xml")
    output_path = Path(output_path)
    output_path.write_text(content)
    return output_path


def ensure_object_scene_xml(cfg, constants) -> "Path | None":
    """Build and write the robot+object scene xml for a HODome object_interaction run.

    Calls the HODome loader's object_source to retrieve the mesh path, then delegates
    to build_hodome_scene_xml to write the scene xml next to the robot xml (the path
    TestSocpRetargeter.__init__ expects: ROBOT_URDF_FILE with .urdf -> _w_<token>.xml).

    Returns the written Path, or None when the scene xml is not needed:
    - cfg.dataset is not "hodome"
    - task_type is "robot_only"
    - the loader returns no object sources
    """
    if cfg.dataset != "hodome":
        return None
    if cfg.task_type == "robot_only":
        return None

    loader = resolve_loader("hodome")
    sources = loader.object_source(
        motion_path=cfg.motion_path,
        obj_path=cfg.obj_path,
        model_path=cfg.model_path,
        task_type=cfg.task_type,
        constants=constants,
        motion_data_config=cfg.motion_data_config,
        smpl_model_dir=getattr(cfg, "smpl_model_dir", None),
    )
    if not sources:
        return None

    token = constants.OBJECT_NAME
    robot_xml_path = Path(constants.ROBOT_URDF_FILE).with_suffix(".xml")
    return build_hodome_scene_xml(robot_xml_path, token, sources[0].mesh_path)
