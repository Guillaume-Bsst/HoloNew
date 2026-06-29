# Scaling de scène configurable (style + interaction) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une échelle de scène configurable (`scale_xy`, `scale_z`, `None→ratio`, ancre statique origine/sol), partagée par les canaux `style` et `interaction`, appliquée **en étape finale sur les références** (jamais avant l'évaluation des contacts), sans toucher les packages d'évaluation.

**Architecture:** On évalue toujours sur la scène RÉELLE (assignation des contacts correcte), puis on scale les refs : positions style, trajectoire objet (`FrameTargets.object_pos`), witness du canal **sol**. Les witness objet sont en frame local et suivent la pose objet scalée (rien à coder). Le morphologique `0.9/0.8` reste intégralement dans `style/build.py` ; la `SceneScaleConfig` ne remplace que le facteur de placement du root et pilote l'interaction.

**Tech Stack:** Python 3.11, numpy float64, dataclasses frozen (stdlib). Tests : pytest, lancés avec `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q` depuis `HoloV2/`.

## Global Constraints

- `targets/config.py` = **stdlib-only** (dataclasses frozen, AUCUN import numpy/torch). La math numpy va dans `targets/scale.py`.
- `ratio = stature / human_height_assumption` (= le ratio du style, `StyleConfig.human_height_assumption` défaut `1.8`). **PAS** `robot_height/stature` (notes périmées du code).
- Ancre : xy autour de l'**origine monde** (×facteur), z autour du **sol** `ground_height` (défaut 0.0, `StyleConfig.ground_height`).
- `None → ratio` ; un float = facteur fixe. Défaut `SceneScaleConfig()` = `(None, None)` ⇒ `ratio` partout (le xy du root devient scalé).
- **Parité native** (invariant testé) : `SceneScaleConfig(scale_xy=1.0, scale_z=None)` reproduit EXACTEMENT la sortie style actuelle.
- Numpy : compute float64, sorties `frozen`/read-only comme les contrats existants.
- Commits conventionnels, **aucun tag/trailer Claude** (cf. `HoloV2/CLAUDE.md`). Auteur `Guillaume-Bsst <guibesset@free.fr>`.
- **Aucune** modif de `evaluator.py`, `interaction/eval.py`, `style/eval.py` (packages d'éval).

---

### Task 1 : `SceneScaleConfig` + helpers purs `resolve_scale` / `apply_scene_scale`

**Files:**
- Modify: `src/targets/config.py` (ajouter `SceneScaleConfig` + champ `scene_scale` dans `TargetsConfig`)
- Create: `src/targets/scale.py`
- Test: `tests/test_scene_scale.py`

**Interfaces:**
- Produces:
  - `SceneScaleConfig(scale_xy: float | None = None, scale_z: float | None = None)` (frozen).
  - `TargetsConfig(style: StyleConfig, scene_scale: SceneScaleConfig)`.
  - `resolve_scale(cfg: SceneScaleConfig, ratio: float) -> tuple[float, float]` (retourne `(s_xy, s_z)`, `None→ratio`).
  - `apply_scene_scale(points: np.ndarray, s_xy: float, s_z: float, ground_height: float = 0.0) -> np.ndarray` : similarité diagonale, xy autour de 0, z autour de `ground_height`. Retourne une COPIE float64.

- [ ] **Step 1 : Test d'échec — config + helpers**

Créer `tests/test_scene_scale.py` :

```python
"""SceneScaleConfig + resolve_scale/apply_scene_scale : la similarité de scène partagée."""
import numpy as np
import pytest

from src.targets.config import SceneScaleConfig, TargetsConfig, StyleConfig
from src.targets.scale import resolve_scale, apply_scene_scale


def test_config_defaults_and_validation():
    c = SceneScaleConfig()
    assert c.scale_xy is None and c.scale_z is None          # défaut = None -> ratio
    assert TargetsConfig().scene_scale == SceneScaleConfig()
    assert TargetsConfig().style == StyleConfig()
    with pytest.raises(ValueError):
        SceneScaleConfig(scale_xy=0.0)                        # facteur explicite doit être > 0
    with pytest.raises(ValueError):
        SceneScaleConfig(scale_z=-1.0)


def test_resolve_scale_none_is_ratio():
    assert resolve_scale(SceneScaleConfig(), ratio=0.5) == (0.5, 0.5)          # None,None -> ratio
    assert resolve_scale(SceneScaleConfig(scale_xy=1.0), ratio=0.5) == (1.0, 0.5)  # xy fixe, z=ratio
    assert resolve_scale(SceneScaleConfig(scale_xy=1.0, scale_z=2.0), 0.5) == (1.0, 2.0)


def test_apply_scene_scale_anchor_and_axes():
    pts = np.array([[2.0, 4.0, 6.0], [1.0, 0.0, 0.0]], np.float64)
    out = apply_scene_scale(pts, s_xy=0.5, s_z=0.25, ground_height=0.0)
    np.testing.assert_allclose(out, [[1.0, 2.0, 1.5], [0.5, 0.0, 0.0]])
    assert out.dtype == np.float64
    np.testing.assert_allclose(pts[0], [2.0, 4.0, 6.0])      # input non muté (copie)


def test_apply_scene_scale_z_anchored_on_ground():
    # un point SUR le sol reste sur le sol (z invariant) ; xy scalé autour de l'origine.
    pts = np.array([[3.0, 3.0, 0.2]], np.float64)
    out = apply_scene_scale(pts, s_xy=1.0, s_z=0.5, ground_height=0.2)
    np.testing.assert_allclose(out, [[3.0, 3.0, 0.2]])       # z == ground_height -> inchangé
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_scale.py -q`
Expected: FAIL (`ImportError: cannot import name 'SceneScaleConfig'` / `resolve_scale`).

- [ ] **Step 3 : Ajouter `SceneScaleConfig` + `scene_scale` dans `config.py`**

Dans `src/targets/config.py`, après la classe `StyleConfig` et AVANT `TargetsConfig`, ajouter :

```python
@dataclass(frozen=True)
class SceneScaleConfig:
    """Échelle de scène (placement), partagée par ``style`` et ``interaction`` et appliquée en étape
    finale sur les RÉFÉRENCES (jamais avant l'éval des contacts). Similarité diagonale ancrée
    statiquement : xy autour de l'origine monde, z autour du sol (``StyleConfig.ground_height``).

    ``None`` => le facteur de cet axe est ``ratio = stature / StyleConfig.human_height_assumption``
    (le MÊME ratio que le style, pour la cohérence style↔interaction). Un float = facteur fixe.
    Défaut ``(None, None)`` => ``ratio`` partout. ``scale_xy=1.0, scale_z=None`` reproduit le
    comportement style natif (xy non scalé, z par ``ratio``)."""

    scale_xy: float | None = None   # facteur xy autour de l'origine ; None -> ratio
    scale_z: float | None = None    # facteur z autour du sol ; None -> ratio

    def __post_init__(self) -> None:
        if self.scale_xy is not None and self.scale_xy <= 0.0:
            raise ValueError(f"scale_xy must be > 0 when set, got {self.scale_xy}")
        if self.scale_z is not None and self.scale_z <= 0.0:
            raise ValueError(f"scale_z must be > 0 when set, got {self.scale_z}")
```

Puis modifier `TargetsConfig` pour composer le nouveau sous-config :

```python
@dataclass(frozen=True)
class TargetsConfig:
    """All knobs of the ``targets`` step, composed — the single object ``pipeline`` receives; each op
    reads only its sub-config. ``style`` = recette + scalaires morpho ; ``scene_scale`` = la similarité
    de scène partagée par style + interaction (placement). L'``InteractionContext.margin`` reste un
    knob ``prepare``."""

    style: StyleConfig = field(default_factory=StyleConfig)
    scene_scale: SceneScaleConfig = field(default_factory=SceneScaleConfig)
```

- [ ] **Step 4 : Créer `src/targets/scale.py`**

```python
"""scale — la similarité de scène PARTAGÉE par ``style`` et ``interaction`` (placement des refs).

``resolve_scale`` résout ``None -> ratio`` ; ``apply_scene_scale`` applique la similarité diagonale
(xy autour de l'origine monde, z autour du sol). Pur, float64, torch-free ; aucune mutation de
l'entrée. ``style/build.py`` n'appelle PAS ``apply_scene_scale`` directement (le morphologique du
pelvis est entrelacé à son placement) : il partage ``resolve_scale`` + la convention d'ancre, et
applique les facteurs dans sa propre formule de root. ``interaction`` (trajectoire objet + witness
sol) utilise ``apply_scene_scale``.
"""
from __future__ import annotations

import numpy as np

from .config import SceneScaleConfig


def resolve_scale(cfg: SceneScaleConfig, ratio: float) -> tuple[float, float]:
    """``(s_xy, s_z)`` : chaque axe ``None`` -> ``ratio`` (= stature / human_height_assumption)."""
    s_xy = ratio if cfg.scale_xy is None else cfg.scale_xy
    s_z = ratio if cfg.scale_z is None else cfg.scale_z
    return s_xy, s_z


def apply_scene_scale(points: np.ndarray, s_xy: float, s_z: float,
                      ground_height: float = 0.0) -> np.ndarray:
    """``(..., 3)`` -> copie scalée : ``x,y *= s_xy`` (autour de 0), ``z`` autour de ``ground_height``
    (un point sur le sol reste sur le sol). Pur, float64, entrée non mutée."""
    out = np.asarray(points, np.float64).copy()
    out[..., 0] *= s_xy
    out[..., 1] *= s_xy
    out[..., 2] = ground_height + (out[..., 2] - ground_height) * s_z
    return out
```

- [ ] **Step 5 : Lancer le test, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_scale.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6 : Commit**

```bash
git add src/targets/config.py src/targets/scale.py tests/test_scene_scale.py
git commit -m "feat(holov2): SceneScaleConfig + resolve_scale/apply_scene_scale (similarité de scène partagée)"
```

---

### Task 2 : Câbler `scene_scale` dans le pipeline (threading, sans changement de comportement)

**Files:**
- Modify: `src/targets/pipeline.py` (passer `cfg.scene_scale` au style ; préparer l'usage interaction)
- Modify: `src/targets/style/build.py` (signature : accepter la `SceneScaleConfig`, encore non utilisée)
- Test: `tests/test_pipeline_targets.py` (déjà vert ; on vérifie qu'il le reste)

**Interfaces:**
- Consumes (Task 1) : `SceneScaleConfig`, `TargetsConfig.scene_scale`.
- Produces : `style.build(pose, robot, stature, cfg=StyleConfig(), scene=SceneScaleConfig())` (param `scene` ajouté, défaut neutre).

- [ ] **Step 1 : Ajouter le param `scene` à `style.build` (non utilisé encore)**

Dans `src/targets/style/build.py`, modifier l'import et la signature :

```python
from ..config import ARM_BODIES, ROOT_BODY, SMPL_BODY_INDEX, SceneScaleConfig, StyleConfig, style_table
```

```python
def build(pose: FramePose, robot: RobotSpec, stature: float,
          cfg: StyleConfig = StyleConfig(), scene: SceneScaleConfig = SceneScaleConfig()) -> StyleTargets:
```

(le corps reste inchangé à cette étape — `scene` n'est pas encore lu ; Task 3 l'utilise.)

- [ ] **Step 2 : Passer `cfg.scene_scale` depuis le pipeline**

Dans `src/targets/pipeline.py`, dans `_build_frame`, modifier l'appel style :

```python
        with prof.span("style"):
            style_t = style.build(pose, robot, grounded.body.stature, cfg.style, cfg.scene_scale)
```

- [ ] **Step 3 : Lancer les tests pipeline + style, vérifier qu'ils restent verts**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_pipeline_targets.py tests/test_style.py -q`
Expected: PASS (le param `scene` au défaut neutre ne change rien encore — `build` ne le lit pas).

- [ ] **Step 4 : Commit**

```bash
git add src/targets/pipeline.py src/targets/style/build.py
git commit -m "chore(holov2): câble TargetsConfig.scene_scale jusqu'à style.build (no-op)"
```

---

### Task 3 : Style — placement du root piloté par `SceneScaleConfig` (+ parité native)

**Files:**
- Modify: `src/targets/style/build.py` (root xy/z via la scène ; membres inchangés `morph·ratio`)
- Test: `tests/test_style.py` (épingler le natif ; ajouter le nouveau défaut)

**Interfaces:**
- Consumes : `resolve_scale`, `SceneScaleConfig`, `StyleConfig`.
- Produces : `style.build(..., scene=...)` applique le placement : root xy `× s_xy`, root z `× scale_torso_legs × s_z`, membres `× morph × ratio` (isotrope, inchangé), ancrés sur le root scalé.

- [ ] **Step 1 : Mettre à jour les tests style existants pour épingler le NATIF**

Dans `tests/test_style.py` : les tests `test_style_scale_and_offset_hand_computed`, `test_style_matches_v1_scale_offset` supposent le comportement actuel (xy natif). Sous le NOUVEAU défaut (`None→ratio`), le xy du root devient scalé → ces tests doivent passer la config native. Importer `SceneScaleConfig` et l'utiliser :

```python
from src.targets.config import SMPL_BODY_INDEX, SceneScaleConfig, StyleConfig, style_table
```

`_NATIVE = SceneScaleConfig(scale_xy=1.0, scale_z=None)` (ajouter en haut, après `_CFG = StyleConfig()`).

Dans `test_style_scale_and_offset_hand_computed`, remplacer l'appel :

```python
    st = style.build(pose, _robot(), stature=stature, scene=_NATIVE)
```

Dans `test_style_matches_v1_scale_offset`, idem :

```python
    st = style.build(pose, _robot(), stature=stature, scene=_NATIVE)
```

- [ ] **Step 2 : Ajouter le test du NOUVEAU défaut (xy scalé par ratio)**

Ajouter dans `tests/test_style.py` :

```python
def test_style_default_scales_root_xy_by_ratio():
    """Défaut SceneScaleConfig() = None,None -> ratio partout : le xy du root (pelvis) est scalé par
    ratio (alors que le natif le garde brut)."""
    pose = _synthetic_pose()
    stature = 0.9
    ratio = stature / _CFG.human_height_assumption                    # 0.5
    st_default = style.build(pose, _robot(), stature=stature)         # SceneScaleConfig() défaut
    st_native = style.build(pose, _robot(), stature=stature, scene=SceneScaleConfig(scale_xy=1.0))
    i = st_default.link_names.index("pelvis")
    root = pose.bone_pos[SMPL_BODY_INDEX["pelvis"]]
    # défaut : pelvis xy = ratio * root_xy ; natif : pelvis xy = root_xy
    np.testing.assert_allclose(st_default.position[i][:2], root[:2] * ratio, atol=1e-9)
    np.testing.assert_allclose(st_native.position[i][:2], root[:2], atol=1e-9)
    # z identique dans les deux (z = scale_torso_legs * ratio * root_z, scale_z=None dans les deux)
    np.testing.assert_allclose(st_default.position[i][2], st_native.position[i][2], atol=1e-9)
```

- [ ] **Step 3 : Lancer, vérifier l'échec du nouveau test (et la parité natif si déjà cassée)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_style.py -q`
Expected: `test_style_default_scales_root_xy_by_ratio` FAIL (le défaut ne scale pas encore le xy — `build` ignore `scene`).

- [ ] **Step 4 : Implémenter le placement piloté par la scène dans `build.py`**

Dans `src/targets/style/build.py`, ajouter l'import du resolver et remplacer le calcul du root :

```python
from ..scale import resolve_scale
```

Remplacer le bloc :

```python
    root_pos = bone_pos[SMPL_BODY_INDEX[ROOT_BODY]]                    # (3,)
    base = cfg.scale_torso_legs * ratio                               # pelvis is a torso/legs body
    scaled_root = np.array([root_pos[0], root_pos[1], root_pos[2] * base])   # sx = sy = 1.0, sz = base
    ground = np.array([0.0, 0.0, cfg.ground_height])
```

par :

```python
    root_pos = bone_pos[SMPL_BODY_INDEX[ROOT_BODY]]                    # (3,)
    # PLACEMENT du root via l'échelle de scène (None -> ratio) ; xy autour de l'origine, z autour du
    # sol. Le morphologique du pelvis (scale_torso_legs, le pelvis est torse/jambes) reste sur z.
    # scale_xy=1.0, scale_z=None reproduit le natif : xy brut, z = scale_torso_legs * ratio.
    s_xy, s_z = resolve_scale(scene, ratio)
    base_z = cfg.scale_torso_legs * s_z
    scaled_root = np.array([root_pos[0] * s_xy, root_pos[1] * s_xy,
                            cfg.ground_height + (root_pos[2] - cfg.ground_height) * base_z])
    ground = np.array([0.0, 0.0, cfg.ground_height])
```

(les membres restent inchangés : `s = morph * ratio` isotrope, `scaled_pos = (src - root) * s + scaled_root` ; ils suivent le ratio, pas la scène, ce qui préserve le natif et donne une mise à l'échelle uniforme au défaut.)

- [ ] **Step 5 : Lancer les tests style, vérifier le succès (natif + nouveau défaut)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_style.py -q`
Expected: PASS (parité native épinglée + nouveau défaut). Note : `test_style_matches_v1_scale_offset` est SKIP si la source V1 est absente — acceptable.

- [ ] **Step 6 : Commit**

```bash
git add src/targets/style/build.py tests/test_style.py
git commit -m "feat(holov2): style — placement du root via SceneScaleConfig (défaut None->ratio, natif épinglé)"
```

---

### Task 4 : Interaction — scaler la trajectoire objet (`FrameTargets.object_pos`)

**Files:**
- Modify: `src/targets/pipeline.py` (`_build_frame` : scaler `object_pos` remis au solveur)
- Test: `tests/test_pipeline_targets.py`

**Interfaces:**
- Consumes : `resolve_scale`, `apply_scene_scale`, `cfg.scene_scale`.
- Produces : `FrameTargets.object_pos` scalé (centre objet, autour origine/sol) ; `object_rot` inchangé ; les witness objet (frame local) suivent automatiquement.

- [ ] **Step 1 : Épingler le test existant qui suppose `object_pos` non scalé**

`test_frame_targets_carry_object_poses_from_frame_pose` (≈ l.63) compare `ft.object_pos` à `frame_pose` (non scalé) — il cassera sous le défaut scalant. Le pin en IDENTITÉ (`scale_xy=1.0, scale_z=1.0`, vrai no-op). Remplacer son appel `process_frame` :

```python
def test_frame_targets_carry_object_poses_from_frame_pose():
    from src.targets.config import TargetsConfig, SceneScaleConfig
    g, ctx = _grounded(n_obj=2), _ctx(n_obj=2)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    identity = TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0))   # no-op
    ft = process_frame(g, ctx, robot, f=1, cfg=identity)
    pose = frame_pose(g, f=1)
    assert ft.object_rot.shape == (2, 3, 3) and ft.object_pos.shape == (2, 3)
    assert np.allclose(ft.object_rot, pose.object_rot)
    assert np.allclose(ft.object_pos, pose.object_pos)
    assert np.allclose(ft.object_pos[0], [0.2, 0.3, 0.5])           # the grounded object position
```

- [ ] **Step 2 : Test d'échec — `object_pos` scalé par défaut**

Ajouter dans `tests/test_pipeline_targets.py` :

```python
def test_object_pos_scaled_by_scene():
    from src.targets.config import StyleConfig, TargetsConfig, SceneScaleConfig
    g, ctx = _grounded(n_obj=1), _ctx(n_obj=1)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ratio = g.body.stature / StyleConfig().human_height_assumption          # 1.7 / 1.8
    identity = TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0))
    ft_native = process_frame(g, ctx, robot, f=0, cfg=identity)
    ft_scaled = process_frame(g, ctx, robot, f=0)                            # défaut None,None -> ratio
    np.testing.assert_allclose(ft_native.object_pos[0], [0.2, 0.3, 0.5], atol=1e-9)
    np.testing.assert_allclose(ft_scaled.object_pos[0], np.array([0.2, 0.3, 0.5]) * ratio, atol=1e-9)
    np.testing.assert_array_equal(ft_scaled.object_rot, ft_native.object_rot)   # rotation inchangée
```

- [ ] **Step 3 : Lancer, vérifier l'échec du nouveau test (et l'épinglage de l'existant)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_pipeline_targets.py -q`
Expected: `test_object_pos_scaled_by_scene` FAIL (`object_pos` pas encore scalé) ; `test_frame_targets_carry_object_poses_from_frame_pose` PASS (identité = no-op, déjà vrai même avant Task 4).

- [ ] **Step 4 : Scaler `object_pos` dans `_build_frame`**

Dans `src/targets/pipeline.py`, ajouter les imports :

```python
from .scale import apply_scene_scale, resolve_scale
```

Dans `_build_frame`, remplacer l'assemblage des `FrameTargets` :

```python
        targets = FrameTargets(
            style=style_t,
            robot_interaction=robot_interaction_targets(robot_field),
            env_interaction=environment_interaction_targets(object_fields),
            object_rot=pose.object_rot,
            object_pos=pose.object_pos,
        )
```

par :

```python
        ratio = grounded.body.stature / cfg.style.human_height_assumption
        s_xy, s_z = resolve_scale(cfg.scene_scale, ratio)
        ground_h = cfg.style.ground_height
        scaled_object_pos = apply_scene_scale(pose.object_pos, s_xy, s_z, ground_h)  # (N, 3) centre objet
        targets = FrameTargets(
            style=style_t,
            robot_interaction=robot_interaction_targets(robot_field),
            env_interaction=environment_interaction_targets(object_fields),
            object_rot=pose.object_rot,
            object_pos=scaled_object_pos,
        )
```

> NOTE : l'éval (`eval_fields`, `transport`) s'est faite AVANT sur `pose.object_pos` RÉEL — c'est voulu (assignation des contacts correcte). Seul le `object_pos` REMIS au solveur est scalé. Ces imports (`resolve_scale`, `apply_scene_scale`) et le calcul `ratio/s_xy/s_z/ground_h` servent aussi à Task 5 — les placer une fois.

- [ ] **Step 5 : Lancer, vérifier le succès + non-régression pipeline**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_pipeline_targets.py -q`
Expected: PASS (nouveau test + existants).

- [ ] **Step 6 : Commit**

```bash
git add src/targets/pipeline.py tests/test_pipeline_targets.py
git commit -m "feat(holov2): interaction — scale de la trajectoire objet (object_pos) sur les refs"
```

---

### Task 5 : Interaction — scaler les witness du canal SOL dans les refs

**Files:**
- Modify: `src/targets/scale.py` (helper `scale_ground_channels`)
- Modify: `src/targets/pipeline.py` (`_build_frame` : appliquer aux champs ref `robot_field` + `object_fields`)
- Test: `tests/test_scene_scale.py` + `tests/test_pipeline_targets.py`

**Interfaces:**
- Consumes : `MultiChannelField` (contracts), `Channel.object_idx` (pour repérer le sol), `apply_scene_scale`.
- Produces : `scale_ground_channels(field: MultiChannelField, ground_idx: tuple[int, ...], s_xy, s_z, ground_height) -> MultiChannelField` — pour les canaux sol : `witness` scalé (xy autour origine, z sur le sol) et `distance *= s_z` là où `active` ; canaux objet et `direction`/`active` inchangés.

- [ ] **Step 1 : Test d'échec — `scale_ground_channels`**

Ajouter dans `tests/test_scene_scale.py` :

```python
def test_scale_ground_channels():
    from src.targets.contracts import MultiChannelField
    from src.targets.scale import scale_ground_channels
    C, P = 2, 3
    distance = np.array([[0.1, 0.2, 5.0], [1.0, 1.0, 1.0]], np.float64)     # canal 0 = sol, 1 = objet
    witness = np.zeros((C, P, 3), np.float64)
    witness[0] = [[2.0, 4.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]        # sol : frame monde
    witness[1] = [[9.0, 9.0, 9.0]] * P                                       # objet : local, NE DOIT PAS bouger
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (C, P, 1))
    active = np.array([[True, True, False], [True, True, True]])
    f = MultiChannelField(distance=distance, direction=direction, witness=witness,
                          active=active, channels=("ground", "obj0"))
    out = scale_ground_channels(f, ground_idx=(0,), s_xy=0.5, s_z=0.25, ground_height=0.0)
    # sol : witness xy * 0.5, z sur le plan ; distance * 0.25 là où active (la 3e inactive inchangée)
    np.testing.assert_allclose(out.witness[0], [[1.0, 2.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(out.distance[0], [0.025, 0.05, 5.0])         # 3e (inactive) inchangée
    # canal objet intact ; direction/active intacts
    np.testing.assert_array_equal(out.witness[1], witness[1])
    np.testing.assert_allclose(out.distance[1], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out.active, active)
    np.testing.assert_allclose(out.direction, direction)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_scale.py::test_scale_ground_channels -q`
Expected: FAIL (`ImportError: cannot import name 'scale_ground_channels'`).

- [ ] **Step 3 : Implémenter `scale_ground_channels` dans `scale.py`**

Ajouter à `src/targets/scale.py` :

```python
from .contracts import MultiChannelField


def scale_ground_channels(field: MultiChannelField, ground_idx: tuple[int, ...],
                          s_xy: float, s_z: float, ground_height: float = 0.0) -> MultiChannelField:
    """Scale, pour les canaux SOL (frame monde, ``ground_idx``), le ``witness`` (similarité de scène)
    et la ``distance`` (= hauteur, ``*= s_z``) là où ``active``. Les canaux OBJET (witness local,
    qui suivent la pose objet scalée), ``direction`` et ``active`` sont inchangés. Retourne un nouveau
    ``MultiChannelField`` (frozen)."""
    distance = np.asarray(field.distance, np.float64).copy()           # (C, P)
    witness = np.asarray(field.witness, np.float64).copy()             # (C, P, 3)
    active = np.asarray(field.active, dtype=bool)                      # (C, P)
    for c in ground_idx:
        witness[c] = apply_scene_scale(witness[c], s_xy, s_z, ground_height)
        distance[c] = np.where(active[c], distance[c] * s_z, distance[c])
    return MultiChannelField(distance=distance, direction=field.direction, witness=witness,
                             active=field.active, channels=field.channels)
```

- [ ] **Step 4 : Lancer le test unitaire, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_scale.py -q`
Expected: PASS.

- [ ] **Step 5 : Appliquer aux champs ref dans `_build_frame`**

Dans `src/targets/pipeline.py`, **étendre l'import de Task 4** pour ajouter `scale_ground_channels` :

```python
from .scale import apply_scene_scale, resolve_scale, scale_ground_channels
```

Le calcul `ratio`, `s_xy, s_z`, `ground_h` (ajouté en Task 4 avant l'assemblage des `FrameTargets`) doit être **remonté AVANT** l'usage ci-dessous (juste après `robot_field = transport(...)` / la boucle `object_fields`). Puis insérer, avant l'assemblage des refs :

```python
        ground_idx = tuple(c for c, ch in enumerate(ctx.channels) if ch.object_idx is None)
        robot_field = scale_ground_channels(robot_field, ground_idx, s_xy, s_z, ground_h)
        object_fields = tuple(
            scale_ground_channels(of, ground_idx, s_xy, s_z, ground_h) for of in object_fields)
```

> L'éval (`eval_fields`/`transport`) reste sur le réel ; seuls les champs REF (`robot_field`, `object_fields`) sont scalés. `ratio/s_xy/s_z/ground_h` ne sont calculés qu'UNE fois et réutilisés par l'assemblage `object_pos` (Task 4).

- [ ] **Step 6 : Test d'intégration — canal sol scalé dans la ref robot (wiring)**

La fixture par défaut (`_Body22`) pose les os à z=0.9 → canal sol INACTIF (au-delà de `margin=0.1`) et xy=0 ⇒ witness trivial. On pose donc un corps PRÈS du sol (dans la marge) et on teste le scaling de la **distance** (= hauteur, non nulle), ce qui vérifie réellement le câblage. Ajouter dans `tests/test_pipeline_targets.py` :

```python
def test_ground_channel_scaled_in_robot_field():
    """Wiring : le pipeline scale le canal SOL des refs. Corps près du sol -> canal sol actif ;
    on vérifie distance(hauteur) *= s_z (le z du witness reste sur le plan, xy nuls ici)."""
    from src.targets.config import TargetsConfig, SceneScaleConfig

    class _BodyLow(_Body22):
        def bone_transforms(self, params, t):
            rot, pos = _Body22.bone_transforms(self, params, t)
            pos = pos.copy(); pos[:, 2] = 0.05            # dans la marge (0.1) du plan sol z~0
            return rot, pos

    g0 = _grounded(n_obj=1)
    g = GroundedScene(joint_pos=g0.joint_pos, joint_names=g0.joint_names, object_poses=g0.object_poses,
                      object_mesh_paths=g0.object_mesh_paths, calibration=g0.calibration, fps=g0.fps,
                      smpl_params=None, body=_BodyLow())
    ctx = _ctx(n_obj=1)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ft_id = process_frame(g, ctx, robot, f=0,
                          cfg=TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0)))
    ft_sc = process_frame(g, ctx, robot, f=0,
                          cfg=TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=2.0, scale_z=2.0)))
    a = np.asarray(ft_id.robot_interaction.field.active[0])           # canal sol = index 0
    did = np.asarray(ft_id.robot_interaction.field.distance[0])
    dsc = np.asarray(ft_sc.robot_interaction.field.distance[0])
    assert a.any()                                                    # sol actif (corps dans la marge)
    np.testing.assert_allclose(dsc[a], did[a] * 2.0, atol=1e-9)       # hauteur scalée par s_z
    # z du witness reste sur le plan (ground_height 0) -> invariant
    wid = np.asarray(ft_id.robot_interaction.field.witness[0])
    wsc = np.asarray(ft_sc.robot_interaction.field.witness[0])
    np.testing.assert_allclose(wsc[a][:, 2], wid[a][:, 2], atol=1e-9)
```

> Le scaling xy du witness sol est couvert par le test UNITAIRE `test_scale_ground_channels` (Step 1), où les xy sont non nuls ; la fixture pipeline a tout en xy=0, d'où le test sur la distance.

- [ ] **Step 7 : Lancer tous les tests targets, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_scale.py tests/test_pipeline_targets.py tests/test_style.py -q`
Expected: PASS.

- [ ] **Step 8 : Commit**

```bash
git add src/targets/scale.py src/targets/pipeline.py tests/test_scene_scale.py tests/test_pipeline_targets.py
git commit -m "feat(holov2): interaction — scale des witness du canal sol sur les refs"
```

---

### Task 6 : Nettoyage des notes périmées + docs + non-régression éval

**Files:**
- Modify: `src/targets/interaction/transport.py` (note périmée → pointe vers `targets`/SceneScaleConfig)
- Modify: `src/prepare/runner.py`, `src/prepare/scene.py`, `src/prepare/contracts.py` (notes « scale at the transport seam / robot_height/stature » → réf au scaling des refs targets)
- Modify: `docs/TARGETS.md` (documenter l'échelle de scène + l'ordre éval-réel→scale-refs)
- Test: suite éval (non-régression)

**Interfaces:** aucune nouvelle API ; cohérence documentaire + garde-fou éval.

- [ ] **Step 1 : Corriger la note de `transport.py`**

Dans `src/targets/interaction/transport.py`, remplacer le paragraphe « The human->robot metric `scale` is NOT applied here … belongs to scene placement in `prepare.runner` … » par :

```
The human->robot metric ``scale`` is NOT applied here: ``transport`` stays a frame-agnostic gather.
L'échelle de scène (placement) est appliquée en ÉTAPE FINALE sur les RÉFÉRENCES par le ``pipeline``
(``targets.config.SceneScaleConfig`` + ``targets.scale``), APRÈS l'évaluation sur la scène réelle —
jamais avant (sinon l'assignation des contacts est corrompue). Le witness objet reste en frame local
(l'objet garde sa taille réelle) et suit la pose objet scalée.
```

- [ ] **Step 2 : Corriger les notes de `prepare` (runner / scene / contracts)**

- `src/prepare/runner.py:14-15` : remplacer « owned by the downstream transport seam, NOT applied here » par une note pointant `targets` :

```
The scene stays at NATIVE (human) scale: l'échelle de scène (human->robot placement) est appliquée en
aval, sur les RÉFÉRENCES de ``targets`` (``targets.config.SceneScaleConfig``), après l'éval — PAS ici
(prepare reste à l'échelle réelle pour que la détection des contacts soit correcte).
```

- `src/prepare/scene.py:8-10` : même esprit (la note « HUMAN scale … NOT applied here » → préciser « appliqué sur les refs targets, pas en prepare »).
- `src/prepare/contracts.py:45` et `:218-219` : la mention « human->robot scale = robot_height / stature, composed at the transport seam » est PÉRIMÉE. Remplacer par : « l'échelle de scène des refs utilise ``ratio = stature / StyleConfig.human_height_assumption`` (le ratio du style), appliquée dans ``targets`` ; voir ``targets.config.SceneScaleConfig`` ».

> Ne PAS changer de logique ici — uniquement les commentaires (ces modules n'appliquent aucun scale).

- [ ] **Step 3 : Documenter dans `docs/TARGETS.md`**

Ajouter une sous-section (après la section flux per-frame) :

```markdown
## Échelle de scène (placement, configurable)

`SceneScaleConfig(scale_xy, scale_z)` (`targets/config.py`), `None → ratio = stature/human_height_assumption`,
ancre statique (xy origine, z sol). Partagée style + interaction, appliquée en **étape finale sur les
références** APRÈS l'éval sur la scène réelle (jamais avant : sinon la détection de contact est
corrompue — une boîte scalée passerait sous la table → faux contact sol).

- `style` : placement du root (`targets/scale.resolve_scale`) ; le morphologique `0.9/0.8` reste interne.
- `interaction` : `object_pos` (trajectoire) + witness du canal **sol** (`targets/scale.apply_scene_scale`
  / `scale_ground_channels`) ; witness objet en frame local → suit la pose objet scalée.
- Défaut `None,None` → `ratio` partout ; `scale_xy=1.0, scale_z=None` = comportement natif.
- Les packages d'**éval** (`evaluator`, `interaction/eval`, `style/eval`) ne sont PAS touchés
  (refs = sortie parallèle, cf. `evaluator.py`). Limitation : résidu de contact `(1-scale)·offset_objet`
  (objet gardé à taille réelle).
```

- [ ] **Step 4 : Non-régression éval — la suite d'éval reste verte**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_evaluator.py tests/test_contact_eval.py tests/test_style_eval.py tests/test_eval_fields.py tests/test_eval_contracts.py -q`
Expected: PASS (aucune modif des packages d'éval). Note : tests réel-data SKIP si données absentes — acceptable.

- [ ] **Step 5 : Compile + import sanity**

Run:
```bash
~/.holonew_deps/miniconda3/envs/holonew/bin/python -m py_compile src/targets/config.py src/targets/scale.py src/targets/style/build.py src/targets/pipeline.py
~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "from src.targets.config import SceneScaleConfig, TargetsConfig; from src.targets.scale import resolve_scale, apply_scene_scale, scale_ground_channels; print('OK')"
```
Expected: `OK`.

- [ ] **Step 6 : Commit**

```bash
git add src/targets/interaction/transport.py src/prepare/runner.py src/prepare/scene.py src/prepare/contracts.py docs/TARGETS.md
git commit -m "docs(holov2): échelle de scène — notes scale unifiées vers targets/SceneScaleConfig + TARGETS.md"
```

---

## Self-Review

**Spec coverage :**
- `SceneScaleConfig` (scale_xy/scale_z, None→ratio, ancre statique) → Task 1. ✓
- Éval réel → scale refs (style positions, trajectoire objet, witness sol) → Tasks 3, 4, 5. ✓
- Witness objet local suit la pose (rien à coder) → couvert (Task 4, note). ✓
- Morphologique style-only conservé → Task 3 (membres `morph·ratio`, root z garde `scale_torso_legs`). ✓
- Parité native + nouveau défaut → Task 3 (tests). ✓
- Refs-only / éval intacte → Task 6 Step 4 (non-régression). ✓
- Limitation résidu objet documentée → Task 6 (TARGETS.md). ✓
- Notes périmées (transport/prepare) + ratio = style ratio → Task 6. ✓ (ajout vs spec, justifié par l'exploration code)

**Placeholder scan :** code complet à chaque step. Fixtures réelles vérifiées dans `tests/test_pipeline_targets.py` : `_grounded(n_obj, T=3)` (corps `_Body22`, `stature=1.7`, objet `[0.2,0.3,0.5]`), `_ctx(n_obj, obj_off_z=0.0)`, `RobotSpec` inliné — utilisées telles quelles dans les tests du plan. Task 4 épingle le test existant `test_frame_targets_carry_object_poses_from_frame_pose` en identité (no-op) car le défaut scale désormais. Task 5 pose un `_BodyLow` près du sol (canal sol actif) et teste le scaling de la **distance** (xy nuls dans la fixture, donc xy couvert par le test unitaire).

**Type consistency :** `resolve_scale(cfg, ratio) -> (s_xy, s_z)`, `apply_scene_scale(points, s_xy, s_z, ground_height)`, `scale_ground_channels(field, ground_idx, s_xy, s_z, ground_height)`, `style.build(..., scene=SceneScaleConfig())`, `process_frame(..., cfg=TargetsConfig(scene_scale=...))` — cohérents entre Tasks 1→5.

**Identité vs natif (piège testé) :** « no-op » = `SceneScaleConfig(scale_xy=1.0, scale_z=1.0)` (identité, pour les tests qui veulent l'ancien `object_pos`/champ non scalé) ; « natif style » = `SceneScaleConfig(scale_xy=1.0, scale_z=None)` (xy brut, z par `ratio` — la parité style). Ne pas confondre.
