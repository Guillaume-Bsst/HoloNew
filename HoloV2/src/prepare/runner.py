"""prepare/ orchestrator — builds (or loads from cache) the build-once assets and assembles
the prepare outputs for a scene.

Drives the deliverable builders (calibration, sdf, point_cloud) through their AssetBuilder
``cache_key``/``build``/``load``. Instrumentation lives HERE via ``prof`` spans + ``event``
(cache hit/miss), NEVER inside the builders. See docs/PREPARE.md, CACHE.md, OBS.md.
"""
from __future__ import annotations

from ..obs import NULL


def prepare(scene_spec, config, prof=NULL):
    """scene_spec (sujet, objets, robot, prise) + PrepareConfig -> (GroundedScene, InteractionContext).
    The grounding ``Calibration`` rides inside ``grounded.calibration`` (provenance/viz), so the return
    is a 2-tuple. The subject ``body`` is built ONCE here (the only place with the SMPL model dir) and
    threaded into the scene; calibration is body-free, so it does not rebuild one. Span plan (filled
    when prepare/ is implemented):

        with prof.span("prepare"):
            raw  = load(scene_spec)
            body = build_body_model(raw.smpl_params, scene_spec.smpl_model_dir) if raw.is_parametric else None
            with prof.span("calibration"):    calib = CalibrationBuilder().build_or_load(...)  # body-free
            with prof.span("scene"):          grounded = scene.assemble(raw, calib, body)
            with prof.span("sdf", n=N):       sdfs = [sdf.build_or_load(o, config) ...]
            with prof.span("point_cloud"):    clouds = pointcloud.build_or_load(body, ...)
            with prof.span("correspondence"): corr = correspondence.build_or_load(...)
        return grounded, InteractionContext(...)   # scale = robot.height / body.stature added with transport
    """
    raise NotImplementedError


def build_all(scene_spec, config, prof=NULL):
    """Force-build every asset for a scene (no online use) — for offline batch / warm cache."""
    raise NotImplementedError
