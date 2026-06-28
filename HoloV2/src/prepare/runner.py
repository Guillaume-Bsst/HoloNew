"""prepare/ orchestrator — builds (or loads from cache) the build-once assets and assembles
the prepare outputs for a scene.

Drives the deliverable builders (calibration, sdf, point_cloud) through their AssetBuilder
``cache_key``/``build``/``load``. This is the ONE place with side effects (the disk cache): the
builders stay pure, effects live at this edge. Instrumentation lives HERE via ``prof`` spans +
``event`` (cache hit/miss), NEVER inside the builders. See docs/PREPARE.md, CACHE.md, OBS.md.

Assembly DAG (one-way, dependency order):
    load -> body -> calibration -> grounded scene -> channels (ground + objects)
    -> correspondence (carries the sampling) -> human cloud (bound to that sampling)
    -> object clouds -> InteractionContext.

The scene stays at NATIVE (human) scale: the human->robot placement scale is a (human, robot)
quantity owned by the downstream transport seam, NOT applied here (the interaction is scale-free).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..obs import NULL
from .config import PrepareConfig
from .contracts import (Channel, GroundedScene, InteractionContext, SceneSpec)
from . import scene
from .calibration import CalibrationBuilder
from .load import load
from .load.mesh import load_mesh
from .load.smpl import build_body_model, rest_body_model
from .point_cloud import HumanCloudBuilder, ObjectCloudBuilder
from .point_cloud.correspondence import CorrespondenceBuilder
from .sdf import SdfBuilder, build_plane_sdf

# The committed neutral correspondence ships under this fixed name (built by
# correspondence/build.regenerate); the runner reuses it as the (robot, template) default instead of
# a hashed key, so every scene shares the one bundled OT table. A cache miss rebuilds it in place.
_CORR_NAME = "corr_neutral"


def _cache_dir(spec: SceneSpec) -> Path:
    """Resolve the cache root: ``spec.cache_dir`` or the package default ``HoloV2/cache/``."""
    if spec.cache_dir is not None:
        return Path(spec.cache_dir)
    return Path(__file__).resolve().parents[2] / "cache"


def _load_or_build(builder: Any, subdir: str, key: str, build: Callable[[], Any],
                   cache_dir: Path, prof, *, force: bool = False) -> Any:
    """Generic load-or-build for one asset, keyed under ``cache_dir/<subdir>/<key>.npz``.

    The AssetBuilder signatures DIFFER (each takes its own sub-config + inputs), so the caller closes
    over the specific ``builder.build(...)`` call in the no-arg ``build`` thunk and computes ``key``
    via the matching ``builder.cache_key(...)``. This keeps the helper free of any per-asset shape.
    ``force`` skips the load short-circuit (used by ``build_all`` to warm the cache)."""
    path = cache_dir / subdir / f"{key}.npz"
    if not force and path.exists():
        prof.event("cache hit", asset=subdir, key=key[:8])
        return builder.load(path)
    prof.event("cache miss", asset=subdir, key=key[:8])
    asset = build()
    builder.save(asset, path)
    return asset


def _scene_xy_bounds(grounded: GroundedScene) -> tuple[np.ndarray, np.ndarray]:
    """``(xy_min (2,), xy_max (2,))`` horizontal extent the flat-ground plane SDF must span: the
    grounded demo joints (where the human cloud lives) unioned with the object positions over the
    clip. ``build_plane_sdf`` pads this further, so the contact band around the feet/objects is in
    grid (probes beyond it are out-of-grid -> inactive, like any object SDF)."""
    xy = [np.asarray(grounded.joint_pos, np.float64)[:, :, :2].reshape(-1, 2)]
    for poses in grounded.object_poses:
        xy.append(np.asarray(poses, np.float64)[:, :2])
    pts = np.concatenate(xy, axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def _build_channels(grounded: GroundedScene, spec: SceneSpec, config: PrepareConfig,
                    cache_dir: Path, prof, *, force: bool) -> tuple[Channel, ...]:
    """The evaluation channels, ground FIRST (``channels[0]``): a flat plane SDF over the scene
    extent by default (analytic, NOT cached), or a cached terrain SDF when ``spec.ground_mesh_path``
    is set; then one cached object SDF per object, ``object_idx`` aligned 0..N-1 with the scene's
    object order."""
    sdf_builder = SdfBuilder()
    if spec.ground_mesh_path is None:
        xy_min, xy_max = _scene_xy_bounds(grounded)
        prof.event("ground plane (analytic)")
        ground_sdf = build_plane_sdf(xy_min, xy_max, config.sdf.spacing, config.sdf.margin,
                                     name="ground")
    else:
        gv, gf = load_mesh(spec.ground_mesh_path)
        ground_sdf = _load_or_build(
            sdf_builder, "sdf", sdf_builder.cache_key(config.sdf, gv, gf),
            lambda: sdf_builder.build(config.sdf, gv, gf, name="ground"),
            cache_dir, prof, force=force)
    channels = [Channel("ground", None, ground_sdf)]

    for i, mesh_path in enumerate(grounded.object_mesh_paths):
        v, f = load_mesh(mesh_path)
        sdf = _load_or_build(
            sdf_builder, "sdf", sdf_builder.cache_key(config.sdf, v, f),
            lambda v=v, f=f, i=i: sdf_builder.build(config.sdf, v, f, name=f"obj{i}"),
            cache_dir, prof, force=force)
        channels.append(Channel(f"obj{i}", i, sdf))
    return tuple(channels)


def _correspondence(spec: SceneSpec, config: PrepareConfig, cache_dir: Path, prof, *, force: bool):
    """Load-or-build the ``(CorrespondenceTable, SurfaceSampling)`` pair. Reuses the committed
    ``corr_neutral.npz`` (robot-agnostic on the human side); a miss rebuilds it from a NEUTRAL
    template body (zero betas) + the robot URDF."""
    builder = CorrespondenceBuilder()
    return _load_or_build(
        builder, "correspondence", _CORR_NAME,
        lambda: builder.build(
            config, rest_body_model(np.zeros(10, np.float32), "neutral", spec.smpl_model_dir),
            spec.robot),
        cache_dir, prof, force=force)


def _validate(grounded: GroundedScene, channels: tuple[Channel, ...], human_cloud,
              object_clouds: tuple, correspondence) -> None:
    """Contract invariants of the assembled context — raise (not assert) on violation (golden rule).

    Ground first, object channels/clouds aligned with the scene's object order, and the human cloud's
    sampling bound to the correspondence (else the transport gather silently points at another point
    order)."""
    if channels[0].object_idx is not None:
        raise ValueError("channels[0] must be the static ground (object_idx is None)")
    obj_channels = channels[1:]
    if not (len(obj_channels) == len(object_clouds) == grounded.n_objects):
        raise ValueError(
            f"object channel/cloud/scene counts disagree: {len(obj_channels)} channels, "
            f"{len(object_clouds)} clouds, {grounded.n_objects} scene objects")
    for i, ch in enumerate(obj_channels):
        if ch.object_idx != i:
            raise ValueError(f"object channel {i} has object_idx={ch.object_idx}, expected {i}")
    if human_cloud.sampling_id != correspondence.smpl_sampling_id:
        raise ValueError(
            f"human cloud sampling_id {human_cloud.sampling_id!r} != correspondence "
            f"smpl_sampling_id {correspondence.smpl_sampling_id!r} — transport would be wrong")


def _run(spec: SceneSpec, config: PrepareConfig, prof, *, force: bool):
    """Shared assembly core for ``prepare`` (load-or-build) and ``build_all`` (force-build)."""
    with prof.span("prepare"):
        raw = load(spec)
        if not raw.is_parametric:
            raise ValueError(
                "prepare needs a parametric body (SMPL params) for the interaction context; "
                "this source is positions-only (style-only path not assembled here)")
        body = build_body_model(raw.smpl_params, spec.smpl_model_dir)
        cache_dir = _cache_dir(spec)

        calib_builder = CalibrationBuilder()
        with prof.span("calibration"):
            calib = _load_or_build(
                calib_builder, "calibration", calib_builder.cache_key(config.calibration, raw),
                lambda: calib_builder.build(config.calibration, raw), cache_dir, prof, force=force)

        with prof.span("scene"):
            grounded = scene.assemble(raw, calib, body)

        with prof.span("sdf", n=grounded.n_objects + 1):
            channels = _build_channels(grounded, spec, config, cache_dir, prof, force=force)

        with prof.span("correspondence"):
            corr_table, sampling = _correspondence(spec, config, cache_dir, prof, force=force)

        with prof.span("point_cloud"):
            human_builder = HumanCloudBuilder()
            human_cloud = _load_or_build(
                human_builder, "cloud/human",
                human_builder.cache_key(config.cloud, grounded.smpl_params, sampling),
                lambda: human_builder.build(config.cloud, body, sampling),
                cache_dir, prof, force=force)
            obj_builder = ObjectCloudBuilder()
            object_clouds = []
            for mesh_path in grounded.object_mesh_paths:
                v, f = load_mesh(mesh_path)
                object_clouds.append(_load_or_build(
                    obj_builder, "cloud/object", obj_builder.cache_key(config.cloud, v, f),
                    lambda v=v, f=f: obj_builder.build(config.cloud, v, f),
                    cache_dir, prof, force=force))
            object_clouds = tuple(object_clouds)

        _validate(grounded, channels, human_cloud, object_clouds, corr_table)

        # margin = the SDF stored band (config.sdf.margin): the band the eval activates within IS the
        # band the grids store, a single source of truth (no separate PrepareConfig.margin knob).
        ctx = InteractionContext(channels=channels, human_cloud=human_cloud,
                                 object_clouds=object_clouds, correspondence=corr_table,
                                 margin=config.sdf.margin)
        return grounded, ctx


def prepare(scene_spec: SceneSpec, config: PrepareConfig, prof=NULL):
    """scene_spec (subject, objects, robot, take) + ``PrepareConfig`` -> ``(GroundedScene,
    InteractionContext)``. The grounding ``Calibration`` rides inside ``grounded.calibration`` (so the
    return is a 2-tuple); the subject ``body`` is built ONCE here (the only place with the SMPL model
    dir) and threaded into the scene. Every build-once asset is load-or-built from the disk cache."""
    return _run(scene_spec, config, prof, force=False)


def build_all(scene_spec: SceneSpec, config: PrepareConfig, prof=NULL):
    """Force-build every asset for a scene (no online use) — for offline batch / warm cache: runs
    each builder's ``build``+``save``, skipping the load short-circuit. Returns the assembled
    ``(GroundedScene, InteractionContext)`` like ``prepare``."""
    return _run(scene_spec, config, prof, force=True)
