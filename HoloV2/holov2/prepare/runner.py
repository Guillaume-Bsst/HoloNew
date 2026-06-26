"""prepare/ orchestrator — builds (or loads from cache) the build-once assets and assembles
the prepare outputs for a scene.

Drives the deliverable builders (calibration, sdf, point_cloud) through their AssetBuilder
``cache_key``/``build``/``load``. Instrumentation lives HERE via ``prof`` spans + ``event``
(cache hit/miss), NEVER inside the builders. See docs/PREPARE.md, CACHE.md, OBS.md.
"""
from __future__ import annotations

from ..obs import NULL


def prepare(scene_spec, config, prof=NULL):
    """scene_spec (sujet, objets, robot, prise) + Config -> (GroundedScene, InteractionContext,
    Calibration).  Span plan (filled when prepare/ is implemented):

        with prof.span("prepare"):
            with prof.span("calibration"):    calib = calibration.build_or_load(...)   # event hit/miss
            with prof.span("scene"):          grounded = scene.assemble(..., calib)
            with prof.span("sdf", n=N):       sdfs = [sdf.build_or_load(o, config) ...]
            with prof.span("point_cloud"):    clouds = pointcloud.build_or_load(...)
            with prof.span("correspondence"): corr = correspondence.build_or_load(...)
    """
    raise NotImplementedError


def build_all(scene_spec, config, prof=NULL):
    """Force-build every asset for a scene (no online use) — for offline batch / warm cache."""
    raise NotImplementedError
