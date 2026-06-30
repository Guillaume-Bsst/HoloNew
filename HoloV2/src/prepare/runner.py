"""Orchestrateur prepare/ — construit (ou charge du cache) les assets build-once et assemble
les sorties prepare pour une scène.

Pilote les builders livrables (calibration, sdf, point_cloud) via leurs ``cache_key``/``build``/``load``
AssetBuilder. C'est l'UNIQUE endroit avec effets de bord (le cache disque) : les builders restent purs,
les effets vivent à cette frontière. L'instrumentation vit ICI via les spans ``prof`` + ``event``
(cache hit/miss), JAMAIS dans les builders. Voir docs/PREPARE.md, CACHE.md, OBS.md.

DAG d'assemblage (one-way, ordre dépendances) :
    load → body → calibration → grounded scene → channels (ground + objects)
    → correspondence (porte le sampling) → human cloud (lié au sampling)
    → object clouds → InteractionContext.

La scène reste à l'échelle NATIVE (humaine) : l'échelle de scène (placement human→robot) est appliquée
en aval, sur les RÉFÉRENCES de ``targets`` (``targets.config.SceneScaleConfig``), après l'éval —
PAS ici (prepare reste à l'échelle réelle pour que la détection des contacts soit correcte).
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
from .load.robot import build_robot_model
from .point_cloud.correspondence import CorrespondenceBuilder, robot_point_cloud
from .geodesic import GeodesicBuilder
from .sdf import SdfBuilder, build_plane_sdf

# La correspondance neutre commitée voyage sous ce nom fixe (construite par
# correspondence/build.regenerate) ; le runner la réutilise comme défaut (robot, template) plutôt qu'une
# clé hashée, donc chaque scène partage la table OT bundée unique. Un cache miss la reconstruit sur place.
_CORR_NAME = "corr_neutral"


def _cache_dir(spec: SceneSpec) -> Path:
    """Résout la racine cache : ``spec.cache_dir`` ou le défaut package ``HoloV2/cache/``."""
    if spec.cache_dir is not None:
        return Path(spec.cache_dir)
    return Path(__file__).resolve().parents[2] / "cache"


def _load_or_build(builder: Any, subdir: str, key: str, build: Callable[[], Any],
                   cache_dir: Path, prof, *, force: bool = False) -> Any:
    """Load-or-build générique pour un asset, keyé sous ``cache_dir/<subdir>/<key>.npz``.

    Les signatures AssetBuilder DIFFERENT (chacun prend sa propre sous-config + inputs), donc l'appelant
    ferme sur l'appel spécifique ``builder.build(...)`` dans le thunk ``build`` sans-arg et calcule
    ``key`` via le ``builder.cache_key(...)`` correspondant. Ceci garde l'helper libre de toute forme
    per-asset. ``force`` ignore le short-circuit de load (utilisé par ``build_all`` pour réchauffer le cache)."""
    path = cache_dir / subdir / f"{key}.npz"
    if not force and path.exists():
        prof.event("cache hit", asset=subdir, key=key[:8])
        return builder.load(path)
    prof.event("cache miss", asset=subdir, key=key[:8])
    asset = build()
    builder.save(asset, path)
    return asset


def _scene_xy_bounds(grounded: GroundedScene) -> tuple[np.ndarray, np.ndarray]:
    """``(xy_min (2,), xy_max (2,))`` étendue horizontale que le SDF plan sol-plat doit couvrir : les
    joints démo ancrés (où vit le nuage humain) uniés avec les positions objets sur le clip.
    ``build_plane_sdf`` rembourrre ceci davantage, donc la bande contact autour pieds/objets est dans
    la grille (les sondes au-delà sont out-of-grid → inactives, comme tout SDF objet)."""
    xy = [np.asarray(grounded.joint_pos, np.float64)[:, :, :2].reshape(-1, 2)]
    for poses in grounded.object_poses:
        xy.append(np.asarray(poses, np.float64)[:, :2])
    pts = np.concatenate(xy, axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def _build_channels(grounded: GroundedScene, spec: SceneSpec, config: PrepareConfig,
                    cache_dir: Path, prof, *, force: bool,
                    object_meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Channel, ...]:
    """Les canaux d'évaluation, sol FIRST (``channels[0]``) : un SDF plan sol-plat sur l'étendue scène
    par défaut (analytique, NON caché), ou un SDF terrain caché quand ``spec.ground_mesh_path`` est défini ;
    puis un SDF objet caché par objet, ``object_idx`` aligné 0..N-1 avec l'ordre objets de la scène.
    ``object_meshes`` (``(verts, faces)`` par objet, chargés UNE FOIS par l'appelant et alignés avec
    ``grounded.object_mesh_paths``) alimente à la fois ce build et le build object-cloud, donc chaque mesh
    est lu une fois par ``prepare`` (le SDF et la géodésique d'un objet partagent sa géométrie)."""
    sdf_builder = SdfBuilder()
    geo_builder = GeodesicBuilder()
    if spec.ground_mesh_path is None:
        xy_min, xy_max = _scene_xy_bounds(grounded)
        prof.event("ground plane (analytic)")
        ground_sdf = build_plane_sdf(xy_min, xy_max, config.sdf.spacing, config.sdf.margin,
                                     name="ground")
        ground_geo = None                                       # sol PLAN → euclidien analytique
    else:
        gv, gf = load_mesh(spec.ground_mesh_path)
        ground_sdf = _load_or_build(
            sdf_builder, "sdf", sdf_builder.cache_key(config.sdf, gv, gf),
            lambda: sdf_builder.build(config.sdf, gv, gf, name="ground"),
            cache_dir, prof, force=force)
        ground_geo = _load_or_build(
            geo_builder, "geodesic", geo_builder.cache_key(config.cloud, config.geodesic, gv, gf),
            lambda: geo_builder.build(config.cloud, config.geodesic, gv, gf, name="ground"),
            cache_dir, prof, force=force)
    channels = [Channel("ground", None, ground_sdf, geodesic=ground_geo)]

    for i, (v, f) in enumerate(object_meshes):
        sdf = _load_or_build(
            sdf_builder, "sdf", sdf_builder.cache_key(config.sdf, v, f),
            lambda v=v, f=f, i=i: sdf_builder.build(config.sdf, v, f, name=f"obj{i}"),
            cache_dir, prof, force=force)
        geo = _load_or_build(
            geo_builder, "geodesic", geo_builder.cache_key(config.cloud, config.geodesic, v, f),
            lambda v=v, f=f, i=i: geo_builder.build(config.cloud, config.geodesic, v, f, name=f"obj{i}"),
            cache_dir, prof, force=force)
        channels.append(Channel(f"obj{i}", i, sdf, geodesic=geo))
    return tuple(channels)


def _correspondence(spec: SceneSpec, config: PrepareConfig, cache_dir: Path, prof, *, force: bool):
    """Load-or-build la paire ``(CorrespondenceTable, SurfaceSampling)``. Réutilise la ``corr_neutral.npz``
    commitée (robot-agnostic côté humain) ; un miss la reconstruit à partir d'un body template NEUTRE
    (zéro betas) + l'URDF robot."""
    builder = CorrespondenceBuilder()
    return _load_or_build(
        builder, "correspondence", _CORR_NAME,
        lambda: builder.build(
            config, rest_body_model(np.zeros(10, np.float32), "neutral", spec.smpl_model_dir),
            spec.robot),
        cache_dir, prof, force=force)


def _validate(grounded: GroundedScene, channels: tuple[Channel, ...], human_cloud,
              object_clouds: tuple, correspondence, robot_cloud) -> None:
    """Invariants contrats du contexte assemblé — lève (pas assert) en violation (règle d'or).

    Sol first, canaux/nuages objets alignés avec l'ordre objets de la scène, et le sampling du nuage
    humain lié à la correspondance (sinon le gather transport pointe silencieusement sur un autre ordre
    de points). Le robot cloud doit porter les MÊMES M points que la correspondance (sinon la re-eval
    online adresse un ensemble de points différent)."""
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
    if robot_cloud.n_points != correspondence.n_points:
        raise ValueError(
            f"robot_cloud has {robot_cloud.n_points} points, correspondence has "
            f"{correspondence.n_points} — they must be the same M points")


def _run(spec: SceneSpec, config: PrepareConfig, prof, *, force: bool):
    """Cœur d'assemblage partagé pour ``prepare`` (load-or-build) et ``build_all`` (force-build)."""
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

        # Meshes objets chargés UNE FOIS ici, puis partagés par le build canal SDF/géodésique ET le
        # build object-cloud en dessous — donc chaque fichier mesh est lu une fois par prepare (pas deux fois).
        object_meshes = [load_mesh(p) for p in grounded.object_mesh_paths]

        with prof.span("sdf+geodesic", n=grounded.n_objects + 1):
            channels = _build_channels(grounded, spec, config, cache_dir, prof, force=force,
                                       object_meshes=object_meshes)

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
            for v, f in object_meshes:
                object_clouds.append(_load_or_build(
                    obj_builder, "cloud/object", obj_builder.cache_key(config.cloud, v, f),
                    lambda v=v, f=f: obj_builder.build(config.cloud, v, f),
                    cache_dir, prof, force=force))
            object_clouds = tuple(object_clouds)

        robot = build_robot_model(spec.robot)
        robot_cloud = robot_point_cloud(corr_table, robot.link_names)

        _validate(grounded, channels, human_cloud, object_clouds, corr_table, robot_cloud)

        # margin = la bande stockée SDF (config.sdf.margin) : la bande que l'eval active est LA bande
        # que les grilles stockent, une unique source de vérité (pas de knob PrepareConfig.margin séparé).
        ctx = InteractionContext(channels=channels, human_cloud=human_cloud,
                                 object_clouds=object_clouds, correspondence=corr_table,
                                 margin=config.sdf.margin, robot_cloud=robot_cloud, robot=robot)
        return grounded, ctx


def prepare(scene_spec: SceneSpec, config: PrepareConfig, prof=NULL):
    """scene_spec (sujet, objets, robot, prise) + ``PrepareConfig`` → ``(GroundedScene,
    InteractionContext)``. L'ancrage ``Calibration`` voyage dans ``grounded.calibration`` (donc le
    retour est un 2-tuple) ; le ``body`` du sujet est construit UNE FOIS ici (l'unique place avec le
    répertoire modèle SMPL) et enfilé dans la scène. Chaque asset build-once est load-or-built du
    cache disque."""
    return _run(scene_spec, config, prof, force=False)


def build_all(scene_spec: SceneSpec, config: PrepareConfig, prof=NULL):
    """Force-build chaque asset pour une scène (pas d'usage online) — pour batch offline / réchauffer
    cache : exécute le ``build``+``save`` de chaque builder, ignorant le short-circuit de load. Retourne
    la ``(GroundedScene, InteractionContext)`` assemblée comme ``prepare``."""
    return _run(scene_spec, config, prof, force=True)
