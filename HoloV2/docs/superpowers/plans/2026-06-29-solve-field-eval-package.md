# Réévaluation des champs de contact pour `solve` — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fournir à `solve` (1) les quantités de référence `distance`+`witness` par canal/point déjà dans `FrameTargets`, et (2) le package de calcul online (`pose_cloud`/`eval_fields`/`MultiChannelField` en API publique de `targets`) pour recalculer ces quantités sur le cloud robot (posé FK@q) et les clouds objets.

**Architecture:** `targets` **possède** le noyau d'évaluation et l'expose en surface publique ; `solve` (aval) le réutilise — zéro nouveau module partagé, dépendances vers l'aval uniquement. `InteractionContext` gagne `robot_cloud` (les M points de correspondance en nuage K=1) + `robot` (moteur cinématique, symétrique à `GroundedScene.body`) ; `FrameTargets` gagne les poses objets/frame.

**Tech Stack:** Python 3.11, numpy (cœur pur, torch-free), scipy (tests), yourdfpy (FK robot, caché derrière `UrdfRobot`), pytest. Spec : `docs/superpowers/specs/2026-06-29-solve-field-eval-package-design.md`.

## Global Constraints

- **Python de l'env** : `~/.holonew_deps/miniconda3/envs/holonew/bin/python` — toutes les commandes pytest le préfixent (alias `PY` ci-dessous), lancées **depuis `HoloV2/`**.
  `PY="$HOME/.holonew_deps/miniconda3/envs/holonew/bin/python"`
- **Imports** : DANS `src/` relatifs et n'importer que la **surface publique** de l'amont (jamais un sous-module interne) ; tests en absolu (`from src.… import …`).
- **Contrats** : dataclasses `frozen`, **numpy-only** importables partout (les deps lourdes restent derrière les protocols `RobotModel`/`BodyModel`). Valider les invariants par `raise ValueError` explicite (jamais `assert`).
- **dtype** : compute `float64` ; arrays stockés `float32`. **Quaternions wxyz**. `J_demo` ≠ `J_bones`.
- **Commits** : conventionnels, auteur `Guillaume-Bsst <guibesset@free.fr>` ; **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic).
- **Hors scope (à l'utilisateur)** : les fonctions de coût (résidus D/X `wd`/`wx`, activation, Jacobiennes) et `RobotModel.point_jacobians`.

## File Structure

- **Create** `src/prepare/point_cloud/correspondence/robot_cloud.py` — helper pur `robot_point_cloud` (CorrespondenceTable → PointCloud K=1, remap liens par nom).
- **Modify** `src/prepare/point_cloud/correspondence/__init__.py` — export `robot_point_cloud`.
- **Modify** `src/prepare/contracts.py` — `InteractionContext` += `robot_cloud`, `robot`.
- **Modify** `src/prepare/runner.py` — construit le `RobotModel` + `robot_cloud`, les passe au contexte ; `_validate` += invariant robot_cloud.
- **Modify** `src/targets/contracts.py` — `FrameTargets` += `object_rot`, `object_pos`.
- **Modify** `src/targets/pipeline.py` — `_build_frame` remplit `object_rot`/`object_pos` depuis `pose`.
- **Modify** `src/targets/__init__.py` — ré-export `pose_cloud`, `eval_fields`, `MultiChannelField`.
- **Create tests** `test_robot_cloud.py`, `test_runner_validate.py`, `test_pipeline_targets.py`, `test_targets_public_api.py`, `test_solve_field_eval.py`.
- **Modify test** `tests/test_runner_prepare.py` — URDF réel + assertions des nouveaux champs.
- **Docs** : notes courtes dans `docs/TARGETS.md` (ré-export) ; docstrings de contrat mis à jour inline.

---

### Task 1: Helper pur `robot_point_cloud`

Le cloud robot = les M points de correspondance (`link_idx` + `offset_local`) vus comme `PointCloud` K=1, **avec remap des indices de lien depuis l'ordre `correspondence.link_names` vers l'ordre FK `robot.link_names`** (par nom), pour que `pose_cloud(cloud, *robot.link_transforms(q))` aille chercher le bon lien par point.

**Files:**
- Create: `src/prepare/point_cloud/correspondence/robot_cloud.py`
- Modify: `src/prepare/point_cloud/correspondence/__init__.py`
- Test: `tests/test_robot_cloud.py`

**Interfaces:**
- Consumes: `CorrespondenceTable` (`link_idx (M,)`, `offset_local (M,3)`, `link_names`, `n_points`), `PointCloud` (`parts (P,K)`, `weights (P,K)`, `offsets (P,K,3)`), `pose_cloud` (from `src.targets.interaction`).
- Produces: `robot_point_cloud(corr: CorrespondenceTable, robot_link_names: tuple[str, ...]) -> PointCloud` (K=1 ; `parts` indexent dans `robot_link_names`).

- [ ] **Step 1: Write the failing test**

`tests/test_robot_cloud.py` :
```python
"""Unit test for robot_point_cloud: the correspondence robot side as a K=1 PointCloud, with link
indices REMAPPED from the correspondence link order to the robot FK link order (by name), so that
pose_cloud under the robot's link_transforms reproduces link_pos + R @ offset_local."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import CorrespondenceTable
from src.prepare.point_cloud.correspondence import robot_point_cloud
from src.targets.interaction import pose_cloud


def _corr(link_idx, link_names, offsets):
    m = len(link_idx)
    return CorrespondenceTable(
        smpl_idx=np.arange(m), link_idx=np.asarray(link_idx, np.int64),
        offset_local=np.asarray(offsets, np.float64), link_names=tuple(link_names),
        smpl_sampling_id="test")


def test_remaps_link_order_by_name():
    # correspondence link order != robot FK order; the remap must follow NAMES, not raw indices.
    corr = _corr(link_idx=[0, 1, 0], link_names=("elbow", "wrist"),
                 offsets=[[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    robot_link_names = ("pelvis", "wrist", "knee", "elbow")          # elbow->3, wrist->1
    cloud = robot_point_cloud(corr, robot_link_names)
    assert cloud.n_points == 3 and cloud.n_influences == 1
    assert cloud.parts[:, 0].tolist() == [3, 1, 3]                   # elbow, wrist, elbow in FK order
    assert np.allclose(cloud.weights, 1.0)
    assert np.allclose(cloud.offsets[:, 0, :], corr.offset_local)


def test_posing_reproduces_link_placement():
    corr = _corr(link_idx=[1, 0], link_names=("a", "b"),
                 offsets=[[0.1, 0.2, 0.3], [-0.1, 0.0, 0.05]])
    robot_link_names = ("b", "a")                                    # a->FK 1, b->FK 0
    cloud = robot_point_cloud(corr, robot_link_names)
    rb, tb = R.from_rotvec([0, 0, 0.5]).as_matrix(), np.array([1.0, 0.0, 0.0])
    ra, ta = R.from_rotvec([0.3, 0, 0]).as_matrix(), np.array([0.0, 2.0, 0.0])
    part_rot, part_pos = np.stack([rb, ra]), np.stack([tb, ta])      # FK order [b, a]
    out = pose_cloud(cloud, part_rot, part_pos)
    assert np.allclose(out[0], ra @ corr.offset_local[0] + ta, atol=1e-6)   # point 0 on "a"
    assert np.allclose(out[1], rb @ corr.offset_local[1] + tb, atol=1e-6)   # point 1 on "b"


def test_missing_link_raises():
    corr = _corr(link_idx=[0], link_names=("ghost",), offsets=[[0, 0, 0]])
    with pytest.raises(ValueError, match="ghost"):
        robot_point_cloud(corr, ("pelvis", "knee"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_robot_cloud.py -q`
Expected: FAIL — `ImportError: cannot import name 'robot_point_cloud'`.

- [ ] **Step 3: Write the helper**

`src/prepare/point_cloud/correspondence/robot_cloud.py` :
```python
"""The robot side of the correspondence as a posable PointCloud — bridges the static
``CorrespondenceTable`` (robot point = link + local offset) to the shared ``pose_cloud`` op so
``solve`` can pose the robot control points by FK, exactly like the human/object clouds (homogeneity).
"""
from __future__ import annotations

import numpy as np

from ...contracts import CorrespondenceTable, PointCloud


def robot_point_cloud(corr: CorrespondenceTable,
                      robot_link_names: tuple[str, ...]) -> PointCloud:
    """The M correspondence robot points as a K=1 ``PointCloud``, ``parts`` indexing into
    ``robot_link_names`` (the FK order of ``RobotModel.link_transforms``).

    ``corr.link_idx`` indexes into ``corr.link_names``; we remap by NAME to ``robot_link_names`` so
    ``pose_cloud(cloud, *robot.link_transforms(q))`` gathers the right link per point. Raises if a
    correspondence link is absent from the robot's link set.
    """
    name_to_fk = {n: i for i, n in enumerate(robot_link_names)}
    try:
        corr_to_fk = np.array([name_to_fk[n] for n in corr.link_names], np.int64)   # (L_corr,)
    except KeyError as e:
        raise ValueError(
            f"correspondence link {e.args[0]!r} absent from robot link_names") from None
    parts = corr_to_fk[np.asarray(corr.link_idx)][:, None]                # (M, 1) into FK order
    weights = np.ones((corr.n_points, 1), np.float32)                     # K=1 rigid
    offsets = np.asarray(corr.offset_local, np.float32)[:, None, :]       # (M, 1, 3) link-local
    return PointCloud(parts=parts, weights=weights, offsets=offsets)
```

Then add the export to `src/prepare/point_cloud/correspondence/__init__.py` — extend the `from`-import and `__all__`:
```python
from .build import CorrespondenceBuilder, build_correspondence, regenerate
from .cache import load_correspondence, save_correspondence
from .robot_cloud import robot_point_cloud

__all__ = ["load_correspondence", "save_correspondence", "build_correspondence",
           "CorrespondenceBuilder", "regenerate", "robot_point_cloud"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_robot_cloud.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/prepare/point_cloud/correspondence/robot_cloud.py \
        src/prepare/point_cloud/correspondence/__init__.py \
        tests/test_robot_cloud.py
git commit -m "feat(holov2): robot_point_cloud — correspondence robot side as a poseable K=1 cloud"
```

---

### Task 2: `InteractionContext` porte `robot_cloud` + `robot` ; le runner les construit

Ajoute les deux champs au contexte (symétriques à `human_cloud`/`object_clouds` et à `GroundedScene.body`), les construit dans `prepare.runner` (le `RobotModel` via `build_robot_model`, le cloud via `robot_point_cloud`), et valide l'invariant `robot_cloud.n_points == correspondence.n_points`. **Le runner construit désormais le `RobotModel` → l'URDF devient requis pour `prepare()`** : le test d'intégration doit pointer sur l'URDF réel.

**Files:**
- Modify: `src/prepare/contracts.py` (`InteractionContext`)
- Modify: `src/prepare/runner.py` (`_validate`, imports, assemblage `_run`)
- Test: `tests/test_runner_validate.py` (nouveau, sans données)
- Modify test: `tests/test_runner_prepare.py` (URDF réel + assertions)
- Doc: `docs/PREPARE.md` (ligne sur les nouveaux champs)

**Interfaces:**
- Consumes: `robot_point_cloud` (Task 1), `build_robot_model(spec: RobotSpec) -> UrdfRobot` (from `src.prepare.load.robot` ; `UrdfRobot.link_names`, `.dof`, `.link_transforms(q)`).
- Produces: `InteractionContext(channels, human_cloud, object_clouds, correspondence, margin, robot_cloud: PointCloud, robot: RobotModel)` ; `_validate(grounded, channels, human_cloud, object_clouds, correspondence, robot_cloud)`.

- [ ] **Step 1: Write the failing unit test for the invariant**

`tests/test_runner_validate.py` :
```python
"""Unit test for runner._validate's robot_cloud invariant (synthetic, no data): the robot cloud must
carry the SAME M points as the correspondence, else the transport gather and the online re-eval would
disagree on the point set."""
import numpy as np
import pytest

from src.prepare.contracts import (Calibration, Channel, CorrespondenceTable, GroundedScene,
                                    PointCloud)
from src.prepare.sdf.build import build_plane_sdf
from src.prepare.runner import _validate


def _ground_channel():
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=0.1, name="ground")
    return Channel("ground", None, sdf)


def _grounded0():
    return GroundedScene(joint_pos=np.zeros((1, 1, 3), np.float32), joint_names=("a",),
                         object_poses=(), object_mesh_paths=(),
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=None)


def _cloud(n, sid=""):
    return PointCloud(parts=np.zeros((n, 1), np.int64), weights=np.ones((n, 1), np.float32),
                      offsets=np.zeros((n, 1, 3), np.float32), sampling_id=sid)


def _corr(m):
    return CorrespondenceTable(smpl_idx=np.arange(m), link_idx=np.zeros(m, np.int64),
                               offset_local=np.zeros((m, 3)), link_names=("root",),
                               smpl_sampling_id="s")


def test_validate_accepts_matching_robot_cloud():
    _validate(_grounded0(), (_ground_channel(),), _cloud(20, "s"), (), _corr(7), _cloud(7))  # no raise


def test_validate_rejects_robot_cloud_point_count_mismatch():
    with pytest.raises(ValueError, match="robot_cloud"):
        _validate(_grounded0(), (_ground_channel(),), _cloud(20, "s"), (), _corr(7), _cloud(6))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_runner_validate.py -q`
Expected: FAIL — `_validate()` takes 5 positional args (the 6th `robot_cloud` not accepted yet) → `TypeError`.

- [ ] **Step 3: Add the contract fields**

In `src/prepare/contracts.py`, `InteractionContext` — add the two fields after `margin` and extend the docstring invariants:
```python
    channels: tuple[Channel, ...]          # ground (static) + one per object
    human_cloud: PointCloud                # on the SMPL surface
    object_clouds: tuple[PointCloud, ...]  # one per object (object_clouds[i] <-> channel object_idx=i)
    correspondence: CorrespondenceTable    # SMPL -> robot (STATIC binding)
    margin: float                          # field activation margin (m)
    robot_cloud: PointCloud                # the M correspondence robot points as a K=1 cloud, parts
                                           # in robot FK link order — solve poses it at q (online re-eval)
    robot: RobotModel                      # q-dependent kinematics engine (FK to pose robot_cloud);
                                           # mirrors GroundedScene.body, heavy deps hidden in the instance
```
Add to the `Invariants` docstring block: `- ``robot_cloud.n_points == correspondence.n_points`` (same M points).`

- [ ] **Step 4: Wire the runner (build robot + cloud, validate, pass to context)**

In `src/prepare/runner.py`:

Imports — add `build_robot_model` and `robot_point_cloud`:
```python
from .load.robot import build_robot_model
from .point_cloud.correspondence import CorrespondenceBuilder, robot_point_cloud
```
(the second line replaces the existing `from .point_cloud.correspondence import CorrespondenceBuilder`).

`_validate` — add the `robot_cloud` parameter and the check (append at the end of the function body):
```python
def _validate(grounded: GroundedScene, channels: tuple[Channel, ...], human_cloud,
              object_clouds: tuple, correspondence, robot_cloud) -> None:
```
```python
    if robot_cloud.n_points != correspondence.n_points:
        raise ValueError(
            f"robot_cloud has {robot_cloud.n_points} points, correspondence has "
            f"{correspondence.n_points} — they must be the same M points")
```

`_run` — after `object_clouds = tuple(object_clouds)`, build the robot engine + cloud, then validate and assemble (replace the existing `_validate(...)` call and `ctx = InteractionContext(...)`):
```python
        robot = build_robot_model(spec.robot)
        robot_cloud = robot_point_cloud(corr_table, robot.link_names)

        _validate(grounded, channels, human_cloud, object_clouds, corr_table, robot_cloud)

        # margin = the SDF stored band (config.sdf.margin): the band the eval activates within IS the
        # band the grids store, a single source of truth (no separate PrepareConfig.margin knob).
        ctx = InteractionContext(channels=channels, human_cloud=human_cloud,
                                 object_clouds=object_clouds, correspondence=corr_table,
                                 margin=config.sdf.margin, robot_cloud=robot_cloud, robot=robot)
        return grounded, ctx
```

- [ ] **Step 5: Run the unit test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_runner_validate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Fix the integration test fixture (URDF now required) and assert the new fields**

In `tests/test_runner_prepare.py`:

Add the URDF path constant near the other path constants (after `_CORR = ...`):
```python
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
```
Gate on it inside `_pick()` (extend the guard condition):
```python
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
```
Point `_robot()` at the real URDF:
```python
def _robot() -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)
```
Add assertions at the end of `test_prepare_returns_grounded_and_context_with_invariants`:
```python
    # the robot side carried for solve: same M points as the correspondence + a usable FK engine.
    assert ctx.robot_cloud.n_points == ctx.correspondence.n_points
    assert ctx.robot_cloud.n_influences == 1
    assert "pelvis" in ctx.robot.link_names and ctx.robot.dof == 29
```

- [ ] **Step 7: Run the integration test (passes or skips cleanly)**

Run: `cd HoloV2 && $PY -m pytest tests/test_runner_prepare.py -q`
Expected: PASS if HODome/SMPL-X/corr/URDF present, else `s` (skipped) — never an error.

- [ ] **Step 8: Update the doc + commit**

Add one line to `docs/PREPARE.md` where `InteractionContext` is described: « `InteractionContext` porte aussi `robot_cloud` (M points de correspondance en nuage K=1, ordre FK) + `robot` (moteur FK, symétrique à `GroundedScene.body`) pour la réévaluation online dans `solve`. »

```bash
git add src/prepare/contracts.py src/prepare/runner.py \
        tests/test_runner_validate.py tests/test_runner_prepare.py docs/PREPARE.md
git commit -m "feat(holov2): InteractionContext carries robot_cloud + robot (online re-eval seam for solve)"
```

---

### Task 3: `FrameTargets` porte les poses objets de la frame

`solve` a besoin des poses objets par frame comme repère de canal (réévaluation) et comme réf/init des variables objet. Elles sont déjà calculées dans `FramePose` ; on les recopie dans `FrameTargets` (part objet seulement, pas les bones).

**Files:**
- Modify: `src/targets/contracts.py` (`FrameTargets`)
- Modify: `src/targets/pipeline.py` (`_build_frame`)
- Test: `tests/test_pipeline_targets.py` (nouveau, synthétique)
- Doc: `docs/TARGETS.md` (ligne sur les nouveaux champs)

**Interfaces:**
- Consumes: `FramePose` (`object_rot (N,3,3)`, `object_pos (N,3)`), `process_frame(grounded, ctx, robot, f) -> FrameTargets`, `frame_pose(grounded, f) -> FramePose`.
- Produces: `FrameTargets(style, robot_interaction, env_interaction, object_rot: np.ndarray, object_pos: np.ndarray)`.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline_targets.py` :
```python
"""process_frame integration (synthetic, torch-free): the assembled FrameTargets carries the per-frame
object world transforms (from frame_pose) — the seam solve needs to re-pose the object channels and
seed the object-variable terms. Exercises the full pose->style->eval->transport->assemble flow on a
fake 22-bone body + analytic plane SDFs (no SMPL, no trimesh, no robot URDF)."""
from pathlib import Path

import numpy as np

from src.prepare.contracts import (Calibration, Channel, CorrespondenceTable, GroundedScene,
                                    InteractionContext, PointCloud, RobotSpec)
from src.prepare.sdf.build import build_plane_sdf
from src.targets.pipeline import frame_pose, process_frame


class _Body22:
    """Minimal BodyModel with 22 SMPL-X bones (the G1 style table reads bone indices up to 21)."""
    faces = np.zeros((1, 3), np.int64)
    n_bones = 22
    stature = 1.7

    def bone_transforms(self, params, t):
        rot = np.tile(np.eye(3), (22, 1, 1))                 # (22, 3, 3)
        pos = np.zeros((22, 3)); pos[:, 2] = 0.9             # all bones at z=0.9
        pos[0, 2] = 0.9 + 0.01 * t                            # frame index leaks into pelvis z
        return rot, pos


def _ctx(n_obj):
    margin = 0.1
    channels = [Channel("ground", None,
                        build_plane_sdf([-1.0, -1.0], [1.0, 1.0], spacing=0.1, margin=margin,
                                        name="ground"))]
    for i in range(n_obj):
        channels.append(Channel(f"obj{i}", i,
                                build_plane_sdf([-1.0, -1.0], [1.0, 1.0], spacing=0.1, margin=margin,
                                                name=f"obj{i}")))
    human = PointCloud(parts=np.zeros((5, 1), np.int64), weights=np.ones((5, 1), np.float32),
                       offsets=np.zeros((5, 1, 3), np.float32), sampling_id="s")
    obj_clouds = tuple(PointCloud(parts=np.zeros((3, 1), np.int64), weights=np.ones((3, 1), np.float32),
                                  offsets=np.zeros((3, 1, 3), np.float32)) for _ in range(n_obj))
    corr = CorrespondenceTable(smpl_idx=np.array([0, 1, 2, 3]), link_idx=np.zeros(4, np.int64),
                               offset_local=np.zeros((4, 3)), link_names=("root",),
                               smpl_sampling_id="s")
    robot_cloud = PointCloud(parts=np.zeros((4, 1), np.int64), weights=np.ones((4, 1), np.float32),
                             offsets=np.zeros((4, 1, 3), np.float32))
    # process_frame never touches ctx.robot / ctx.robot_cloud (those are solve-only) -> robot=None ok.
    return InteractionContext(channels=tuple(channels), human_cloud=human, object_clouds=obj_clouds,
                              correspondence=corr, margin=margin, robot_cloud=robot_cloud, robot=None)


def _grounded(n_obj, T=3):
    obj = np.tile([0.2, 0.3, 0.5, 1, 0, 0, 0], (T, 1)).astype(np.float32)   # identity-quat pose
    return GroundedScene(joint_pos=np.zeros((T, 1, 3), np.float32), joint_names=("a",),
                         object_poses=(obj,) * n_obj, object_mesh_paths=(Path("o.obj"),) * n_obj,
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=_Body22())


def test_frame_targets_carry_object_poses_from_frame_pose():
    g, ctx = _grounded(n_obj=2), _ctx(n_obj=2)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ft = process_frame(g, ctx, robot, f=1)
    pose = frame_pose(g, f=1)
    assert ft.object_rot.shape == (2, 3, 3) and ft.object_pos.shape == (2, 3)
    assert np.allclose(ft.object_rot, pose.object_rot)
    assert np.allclose(ft.object_pos, pose.object_pos)
    assert np.allclose(ft.object_pos[0], [0.2, 0.3, 0.5])           # the grounded object position
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_pipeline_targets.py -q`
Expected: FAIL — `AttributeError: 'FrameTargets' object has no attribute 'object_rot'` (and `process_frame` builds `FrameTargets` without the new fields).

- [ ] **Step 3: Add the contract fields**

In `src/targets/contracts.py`, `FrameTargets` — add after `env_interaction` and note the use in the docstring:
```python
    style: StyleTargets
    robot_interaction: RobotInteractionTargets
    env_interaction: EnvironmentInteractionTargets
    object_rot: np.ndarray                 # (N, 3, 3) per-frame object world rotations — solve's
                                           # object-channel frame + the object-variable init/reference
    object_pos: np.ndarray                 # (N, 3)    per-frame object world positions
```

- [ ] **Step 4: Fill them in the pipeline**

In `src/targets/pipeline.py`, `_build_frame` — extend the `FrameTargets(...)` construction:
```python
        targets = FrameTargets(
            style=style_t,
            robot_interaction=robot_interaction_targets(robot_field),
            env_interaction=environment_interaction_targets(object_fields),
            object_rot=pose.object_rot,
            object_pos=pose.object_pos,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_pipeline_targets.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Update the doc + commit**

In `docs/TARGETS.md`, on the `FrameTargets` description, add: « + `object_rot`/`object_pos` (poses objets de la frame) pour que `solve` repose les canaux objets et initialise les variables objet. »

```bash
git add src/targets/contracts.py src/targets/pipeline.py tests/test_pipeline_targets.py docs/TARGETS.md
git commit -m "feat(holov2): FrameTargets carries per-frame object poses (solve channel frame + object-var init)"
```

---

### Task 4: `targets` ré-exporte le noyau d'évaluation public

`solve` doit importer `pose_cloud`/`eval_fields`/`MultiChannelField` depuis le **package** `targets`, jamais depuis le sous-module interne `targets.interaction`.

**Files:**
- Modify: `src/targets/__init__.py`
- Test: `tests/test_targets_public_api.py` (nouveau)

**Interfaces:**
- Produces: `from src.targets import pose_cloud, eval_fields, MultiChannelField` (mêmes objets que les internes, juste ré-exportés).

- [ ] **Step 1: Write the failing test**

`tests/test_targets_public_api.py` :
```python
"""The targets package re-exports the pure interaction kernel reused by solve, so downstream imports
the PACKAGE surface (from src.targets import ...), never the internal submodule."""


def test_kernel_is_importable_from_package():
    from src.targets import pose_cloud, eval_fields, MultiChannelField
    from src.targets.interaction import pose_cloud as _pc
    from src.targets.interaction.eval import eval_fields as _ef
    from src.targets.contracts import MultiChannelField as _mcf
    assert pose_cloud is _pc           # same object, just re-exported at the package surface
    assert eval_fields is _ef
    assert MultiChannelField is _mcf
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_targets_public_api.py -q`
Expected: FAIL — `ImportError: cannot import name 'pose_cloud' from 'src.targets'`.

- [ ] **Step 3: Re-export from the package init**

Replace `src/targets/__init__.py` with:
```python
"""``targets`` stage — online, q-independent: per-frame style + interaction targets.

Public surface (what downstream stages import): ``targets.contracts`` (the data types it produces,
e.g. ``FrameTargets``/``FrameTrace``), ``targets.config`` (its knobs, when added), AND the pure
interaction kernel reused by ``solve`` — ``pose_cloud`` / ``eval_fields`` (-> ``MultiChannelField``).
Import these from the PACKAGE (``from ..targets import ...``), never from the internal
``targets.interaction`` submodule. It consumes the upstream ``prepare`` contracts; ``solve`` and
``viz`` import their inputs from ``targets.contracts``.
"""
from .interaction import eval_fields, pose_cloud
from .contracts import MultiChannelField

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_targets_public_api.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/targets/__init__.py tests/test_targets_public_api.py
git commit -m "feat(holov2): expose pose_cloud/eval_fields/MultiChannelField as targets public surface"
```

---

### Task 5: Test d'intégration — le package online sur des assets réels (data-gated)

Valide le chemin complet du livrable 2 sur de vraies données : poser `ctx.robot_cloud` à une config via `ctx.robot` et réévaluer contre `ctx.channels` avec les **mêmes** `pose_cloud`/`eval_fields`, en retombant sur le layout `(C, M)` du champ de référence. Sanity du seam (pas une parité : le robot au repos n'est pas la démo).

**Files:**
- Test: `tests/test_solve_field_eval.py` (nouveau, data-gated)

**Interfaces:**
- Consumes: `prepare(spec, cfg) -> (GroundedScene, InteractionContext)`, `process_frame(...).robot_interaction.field`, `ctx.robot.link_transforms(q)`, `ctx.robot.dof`, `ctx.robot_cloud`, et le package public `pose_cloud`/`eval_fields`/`MultiChannelField`.

- [ ] **Step 1: Write the data-gated integration test**

`tests/test_solve_field_eval.py` :
```python
"""Online field package on REAL prepared assets (data-gated): pose the robot_cloud at a config via
ctx.robot and re-evaluate against ctx.channels with the SAME pose_cloud/eval_fields that targets uses,
so solve gets (distance, witness) per channel per robot point — matching the reference field's (C, M)
layout. Skips when the HODome demo / SMPL-X / committed correspondence / G1 URDF are absent."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets import MultiChannelField, eval_fields, pose_cloud
from src.targets.pipeline import process_frame

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick():
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X / corr / G1 URDF absent")


@_SKIP
def test_online_package_evaluates_robot_cloud_against_channels(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    M = ctx.correspondence.n_points
    C = len(ctx.channels)

    ft = process_frame(g, ctx, spec.robot, f=0)
    ref = ft.robot_interaction.field                       # reference (C, M) on the robot points
    assert ref.distance.shape == (C, M)

    # ONLINE: pose the robot_cloud at the rest config via ctx.robot, eval against the same channels.
    q = np.zeros(ctx.robot.dof)
    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))    # (M, 3) world
    assert pts.shape == (M, 3)
    cur = eval_fields(pts, ctx.channels, ft.object_rot, ft.object_pos, ctx.margin)
    assert isinstance(cur, MultiChannelField)
    assert cur.distance.shape == (C, M)                    # same (channel, point) layout as reference
    assert cur.witness.shape == (C, M, 3)
    assert np.isfinite(cur.distance).all()                 # finite everywhere (active or clamped)
    assert np.isfinite(cur.witness).all()
```

- [ ] **Step 2: Run it (passes on a data machine, else skips)**

Run: `cd HoloV2 && $PY -m pytest tests/test_solve_field_eval.py -q`
Expected: PASS if the demo data + G1 URDF are present, else `s` (skipped) — never an error.

- [ ] **Step 3: Commit**

```bash
git add tests/test_solve_field_eval.py
git commit -m "test(holov2): online field package re-evaluates robot_cloud on real prepared assets"
```

---

### Task 6: Vérification globale

- [ ] **Step 1: Compile-check the changed package modules**

Run: `cd HoloV2 && $PY -m py_compile src/prepare/contracts.py src/prepare/runner.py src/prepare/point_cloud/correspondence/robot_cloud.py src/targets/contracts.py src/targets/pipeline.py src/targets/__init__.py`
Expected: no output (exit 0).

- [ ] **Step 2: Run the new + adjacent unit tests (fast, no full suite)**

Run: `cd HoloV2 && $PY -m pytest tests/test_robot_cloud.py tests/test_runner_validate.py tests/test_pipeline_targets.py tests/test_targets_public_api.py tests/test_pose_cloud.py tests/test_eval_fields.py tests/test_transport.py tests/test_pipeline_frame_pose.py -q`
Expected: all PASS (the new tests + the unchanged kernel/pipeline tests still green).

- [ ] **Step 3: Confirm the contracts stay numpy-only-importable**

Run: `cd HoloV2 && $PY -c "import numpy; from src.prepare.contracts import InteractionContext; from src.targets.contracts import FrameTargets; print('ok')"`
Expected: prints `ok` (no torch/yourdfpy pulled at contract import).

## Self-Review

**1. Spec coverage:**
- Livrable 1 (références déjà dans `FrameTargets`) — confirmé, aucun changement requis (vérifié par Task 5 qui lit `robot_interaction.field`). ✓
- Livrable 2 (package online) — `targets/__init__` ré-export (Task 4) ; `InteractionContext` += `robot_cloud`+`robot` (Task 2) ; `FrameTargets` += poses objets (Task 3). ✓
- `robot_cloud` dérivé de `correspondence` avec remap par nom — Task 1. ✓
- Invariants d'assemblage (`robot_cloud.n_points == correspondence.n_points`, lien absent → raise) — Tasks 1 & 2. ✓
- Garantie de repère (réf/online même `eval_fields`) — structurellement assurée par la réutilisation littérale (Tasks 4–5) ; layout `(C,M)` vérifié en Task 5. ✓
- **Divergence assumée vs spec** : la « parité V1 » (`query_entities`/`contact_field`) n'est PAS codée comme tâche (elle exige le harnais V1 `rt`/pinocchio non cartographié ici ; l'écrire en demi-teinte serait un placeholder). Remplacée par le test de cohérence structurelle Task 5. Voir « Suites » — à confirmer avec l'utilisateur. `RobotModel.point_jacobians` reste hors scope (spec). ✓

**2. Placeholder scan:** aucun « TBD/TODO », chaque step de code montre le code réel, chaque commande a sa sortie attendue. ✓

**3. Type consistency:** `robot_point_cloud(corr, robot_link_names) -> PointCloud` (Task 1) consommé identiquement par le runner (Task 2) ; `_validate(..., robot_cloud)` aligné entre test (Task 2 Step 1) et impl (Step 4) ; `InteractionContext(..., robot_cloud, robot)` aligné entre contrat (Task 2), fixtures (Task 3), runner (Task 2) ; `FrameTargets(..., object_rot, object_pos)` aligné entre contrat, pipeline et test (Task 3). `process_frame(grounded, ctx, robot, f)` — `robot` est un `RobotSpec` (style table), distinct de `ctx.robot` (`RobotModel`) : cohérent partout. ✓

## Suites possibles (hors plan)
- Test de **parité V1** complet (`build_dx_blocks` côté V1 vs `(distance, witness)` V2 aux mêmes points monde) une fois le harnais V1 mobilisé.
- `RobotModel.point_jacobians` + le `ConstraintProvider` D/X (`wd`/`wx`) — la couche de coût utilisateur.
- Batch vectorisé sur T du package (déjà array-oriented).
