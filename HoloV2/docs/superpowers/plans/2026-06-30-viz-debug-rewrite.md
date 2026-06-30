# viz/debug — réécriture des viewers de debug sur `core/` (Phase D) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réécrire les **4 viewers de debug** (`scene`, `cloud`, `sdf`, `hoim3_multiperson`) + leur glue CLI (`_scene_args`) sur le socle partagé `viz/core/` (Phase A), en **préservant exactement** leur comportement actuel + leurs CLI, et en **tuant la duplication** (player ×4 + keep-alive, colormaps ×3, conversion quat ×3, hack de masquage triangle-dégénéré). Nouvelles localisations sous `src/viz/debug/` ; les 5 anciens fichiers `src/viz/*.py` sont **supprimés en dernier**, après parité.

**Architecture:** Chaque viewer reste un **runnable séparé** (`python -m src.viz.debug.<x>`) et garde son **droit de piloter les internes d'étage** qu'il visualise (exception ARCHITECTURE.md / CLAUDE.md : `load`/`calibration`/`sdf`/`point_cloud`, et le `build_person_params` interne du loader hoim3) pour montrer des intermédiaires **hors-contrat** (clearance de grounding, percentile pied live, parité vs surface SMPL pleine, tranches/bande/witness SDF). Mais : viser confiné à `core/viser_ops` + le module viewer lui-même ; toute la mécanique partagée vient de `core/`. La logique **pure** (point le plus bas d'un objet rigide, point le plus bas humain, reconstruction barycentrique de la surface de référence pour la parité, coords des nœuds de grille SDF) est extraite en **fonctions pures testées** dans un helper debug `src/viz/debug/_geometry.py` (numpy-only, sans viser).

**Tech Stack:** Python, numpy (compute float64, arrays affichés float32), scipy (déjà tiré par le pipeline), env python `~/.holonew_deps/miniconda3/envs/holonew/bin/python`. **viser + trimesh importés en LAZY** (dans la fonction `view_*`, jamais au chargement du module — sauf via `core/viser_ops` qui est le SEUL module viser confiné). pytest pour les helpers purs + smokes d'import.

## Dépendances de phase

- **DÉPEND de la Phase A (socle `viz/core/`) UNIQUEMENT.** Ce plan **consomme** `core/` par nom canonique et est écrit **comme si A était mergée**. Indépendant de B/C.
- **Surface canonique consommée (fournie par Phase A — NE PAS redéfinir) :**
  - `src/viz/core/player.py` : `Player` — possède le folder GUI **Playback** (slider `frame` / `play` / `fps`) + la boucle play/fps + le thread keep-alive. **Contrat consommé** (à confirmer avec Phase A, cf. Questions ouvertes) :
    ```python
    Player(server: "viser.ViserServer", n_frames: int,
           render: Callable[[int], None], *, fps: float = 20.0)
        .frame -> int            # valeur courante du slider Playback
        .request_render() -> None  # appelle render(self.frame) maintenant (toggles Display/Grounding)
        .run() -> None           # render initial + thread play/fps + keep-alive (bloquant)
    ```
  - `src/viz/core/colors.py` : `heat_distance`, `diverging`, `parity`, `active_mask`, `AXIS_COLORS`. **Consommés ici** : `diverging(values, vmax) -> (N,3) uint8` (bleu −vmax / blanc 0 / rouge +vmax, ex-`sdf._diverging`) et `parity(err, vmax) -> (N,3) uint8` (bleu 0 / rouge ≥ vmax, ex-`cloud._heat`).
  - `src/viz/core/viser_ops.py` : `quat_wxyz_to_R(quat) -> R` (`(...,4) wxyz -> (...,3,3)`), `hide(handle) -> None`, `point_cloud(server, name, points, colors, *, point_size, visible=True) -> handle`, `line_segments(server, name, segments, colors, *, line_width, visible=True) -> handle`.
  - `src/viz/core/layer.py` : `Layer`/`UiState` — **NON utilisés** par les viewers debug (intermédiaires hors-contrat ⇒ on garde les données par-frame bespoke, cf. spec « Viewers debug »).

## Global Constraints (verbatim)

- **Code / commentaires / docstrings en ANGLAIS.**
- Les viewers debug **PEUVENT piloter les internes d'étage** (l'exception CLAUDE.md) **mais viser reste confiné à `core/viser_ops` + le module viewer** (jamais dans un helper pur).
- **Réutiliser `core/`** : aucun re-duplication de `player` / `colors` / `quat` / `hide` (les 4 dettes nommées par la spec).
- **Tests dans `HoloV2/tests/`**, lancés depuis `HoloV2/` avec `~/.holonew_deps/miniconda3/envs/holonew/bin/python`, **`max_frames` bas**.
- Imports **relatifs** dans `src/` (attention : `debug/` ajoute **un niveau** ⇒ `..prepare` devient `...prepare`, `.. import paths` devient `... import paths`) ; **absolus** (`from src.…`) dans `tests/`.
- **Commits conventionnels, JAMAIS tagger Claude** (aucun `Co-Authored-By`/mention). Auteur `Guillaume-Bsst`.
- `trimesh`/`viser` **lazy** (dans la fonction). Helpers purs (`_geometry.py`) = **numpy-only, sans viser**.

## File Structure

```
src/viz/debug/
  __init__.py        # NEW — marqueur de package (docstring courte)
  _args.py           # NEW (← move src/viz/_scene_args.py) : add_scene_args / _g1_robot / scene_from_args
  _geometry.py       # NEW — helpers PURS testés : object_world_lowz / lowest_point / surface_points / parity_error / node_coords
  scene.py           # NEW (← rewrite src/viz/scene.py)              load/grounding debug
  cloud.py           # NEW (← rewrite src/viz/cloud.py)              point_cloud bake (parité)
  sdf.py             # NEW (← rewrite src/viz/sdf.py)                SDF pure-géométrie (pas de Player : aucune frame)
  hoim3.py           # NEW (← rewrite src/viz/hoim3_multiperson.py)  multi-personnes HOI-M3

tests/
  test_scene_args.py          # MODIFIED — repointé vers src.viz.debug._args (+ smoke _g1_robot)
  test_viz_debug_geometry.py  # NEW — TDD des helpers purs
  test_viz_debug_imports.py   # NEW — smokes d'import par viewer (consomme bien core/)

DELETE (Task 7, en dernier, après parité) :
  src/viz/scene.py · src/viz/cloud.py · src/viz/sdf.py · src/viz/hoim3_multiperson.py · src/viz/_scene_args.py
```

**Coexistence pendant la migration :** les nouveaux fichiers `debug/` sont créés AVANT la suppression des anciens ⇒ pendant les Tasks 1–6 les deux jeux coexistent et restent runnables (anciens via `python -m src.viz.scene`, nouveaux via `python -m src.viz.debug.scene`). On **copie puis supprime-en-dernier** (= le « move » fait proprement). Task 7 supprime les 5 anciens d'un coup après vérification de parité, et **repointe les deux importeurs restants** (`src/viz/viewer.py:45`, qui tire `_scene_args`).

---

### Task 1 : déplacer / construire `debug/_args.py` (+ package `debug/`)

**Files:**
- Create: `src/viz/debug/__init__.py`
- Create: `src/viz/debug/_args.py`  (contenu de `src/viz/_scene_args.py`, profondeur d'import corrigée)
- Modify: `tests/test_scene_args.py`  (repointer l'import vers le nouveau module + smoke `_g1_robot`)
- (NE PAS supprimer `src/viz/_scene_args.py` ici — les anciens viewers + `viewer.py` l'importent encore ; suppression en Task 7.)

**Interfaces:**
- Produces : `src.viz.debug._args.{add_scene_args, _g1_robot, scene_from_args}` (mêmes signatures qu'avant).
- Consumes : `...prepare.contracts.{RobotSpec, SceneSpec}`, `... paths` (profondeur `...` car `debug/` est un niveau plus bas).

- [ ] **Step 1 : Repointer le test (échec attendu)** — `tests/test_scene_args.py` ligne 4 :

```python
from src.viz.debug._args import add_scene_args, scene_from_args
```

et ajouter, à la fin du fichier, un smoke pour la 3ᵉ fonction publique déplacée :

```python
def test_g1_robot_smoke():
    from src.viz.debug._args import _g1_robot
    r = _g1_robot()
    assert r.name == "g1" and r.dof == 29
    assert r.urdf_path.name == "g1_29dof.urdf" and r.urdf_path.is_absolute()
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_args.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.viz.debug'`).

- [ ] **Step 3 : Créer `src/viz/debug/__init__.py`**

```python
"""``viz/debug`` — per-stage DEBUG viewers, rewritten onto ``viz/core``.

Each viewer is a standalone runnable (``python -m src.viz.debug.<x>``) that DRIVES the internals of
the stage it visualises (load / calibration / sdf / point_cloud) to surface non-contract
intermediates — the ARCHITECTURE.md debug-viewer exception. They consume the shared ``viz/core``
socle (Player, colors, viser_ops) and confine viser to ``core/viser_ops`` + the viewer module. The
PROD viewer path lives elsewhere (``viz/app.py``); these are debugging tools only."""
```

- [ ] **Step 4 : Créer `src/viz/debug/_args.py`** (copie de `_scene_args.py`, **imports approfondis d'un cran** : `..` → `...`)

```python
"""Shared CLI glue for the debug viz entry points: declare the common scene flags and assemble a
fully-resolved ``SceneSpec`` from them (filling defaults from the machine-local paths.toml).

EDGE-only: imported by the debug viewers' ``main()`` functions, never by the pure pipeline. Keeps the
viewers from each duplicating the RobotSpec/SceneSpec construction. (Moved from ``viz/_scene_args.py``
into ``viz/debug/`` during the viz redesign — same public surface, imports one level deeper.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ... import paths
from ...prepare.contracts import RobotSpec, SceneSpec


def add_scene_args(ap: argparse.ArgumentParser) -> None:
    """Add the scene-selection flags shared by the viz CLIs."""
    ap.add_argument("--dataset", default="hodome")
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolute, or relative to [datasets.<dataset>].motion in paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="SMPL-X model dir; default: paths.toml [models].smplx")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="release root for object/betas metadata; default: paths.toml [datasets.<dataset>].meta")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--person-id", type=int, default=None, help="multi-person: which person to retarget")
    ap.add_argument("--object-names", default=None, help="comma-separated subset of objects to load")


def _g1_robot() -> RobotSpec:
    """Default G1 RobotSpec for the viz entry points (single-sourced URDF/DOF/height)."""
    return RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                     link_names=("pelvis",), dof=29, height=1.3)


def scene_from_args(a: argparse.Namespace, *, paths_file: Path | None = None) -> SceneSpec:
    """Build a fully-resolved SceneSpec, filling missing paths from paths.toml.

    Explicit CLI args always win; paths.toml is read only when a default is needed (so fully
    explicit, absolute invocations work even without a paths.toml).
    """
    # paths.toml is only HARD-required when a default must come from it: a missing model-dir
    # or a relative motion path. A missing --dataset-root degrades to None (dataset_meta_root
    # returns None), so it must NOT force the file — absolute invocations work with no paths.toml.
    hard_need = (a.model_dir is None) or (not Path(a.motion_path).is_absolute())
    try:
        cfg = paths.load_paths(paths_file)
    except FileNotFoundError:
        if hard_need:
            raise
        cfg = {}

    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion(a.dataset, a.motion_path, cfg)
    droot = a.dataset_root if a.dataset_root is not None else paths.dataset_meta_root(a.dataset, cfg)

    objs = tuple(a.object_names.split(",")) if a.object_names else None
    return SceneSpec(dataset=a.dataset, motion_path=motion, robot=_g1_robot(),
                     smpl_model_dir=model_dir, dataset_root=droot,
                     person_id=a.person_id, object_names=objs,
                     smplh_dir=paths.smplh_dir(cfg), smpl2smplx_pkl=paths.smpl2smplx_pkl(cfg))
```

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_args.py -q`
Expected: PASS (8 anciens tests repointés + `test_g1_robot_smoke`).

- [ ] **Step 6 : Commit**

```bash
git add src/viz/debug/__init__.py src/viz/debug/_args.py tests/test_scene_args.py
git commit -m "refactor(holov2): viz/debug — déplace _scene_args -> debug/_args (move, imports +1 niveau) + test repointé"
```

---

### Task 2 : helpers PURS testés — `debug/_geometry.py`

Extrait la logique **pure** disséminée dans les viewers (point bas objet `scene._object_world_lowz`, point bas humain `scene` inline, reconstruction barycentrique de la surface de parité `cloud` inline, coords de nœuds `sdf._node_coords`) en fonctions **numpy-only, sans viser**, testées par valeur connue. Les viewers convertissent les quats via `core.viser_ops.quat_wxyz_to_R` et passent des **matrices** ici ⇒ `_geometry` reste viser-free **et** ne re-duplique pas la conversion quat.

**Files:**
- Create: `src/viz/debug/_geometry.py`
- Test: `tests/test_viz_debug_geometry.py`

**Interfaces:**
- Produces : `object_world_lowz(verts_local (V,3), rot (F,3,3), pos (F,3), cap=8000) -> (min_z (F,), low_point (F,3))`, `lowest_point(points (F,P,3)) -> (min_z (F,), low_point (F,3))`, `surface_points(verts (V,3), tri_idx (N,3), bary (N,3)) -> (N,3)`, `parity_error(posed (N,3), ref (N,3)) -> (N,)`, `node_coords(origin (3,), spacing: float, shape (3,)) -> (Nx,Ny,Nz,3)`.
- Consumes : numpy uniquement.

- [ ] **Step 1 : Écrire le test [full code]**

```python
# tests/test_viz_debug_geometry.py
"""Pure debug-geometry helpers: known input -> known value (no viser, headless)."""
import numpy as np

from src.viz.debug._geometry import (lowest_point, node_coords, object_world_lowz,
                                     parity_error, surface_points)


def test_object_world_lowz_identity_translation():
    # 8 unit-cube corners in [-1,1]^3, identity rotation, lifted by +5 in z -> world z in {4,6}.
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    rot = np.eye(3)[None]                                    # (1,3,3)
    pos = np.array([[0.0, 0.0, 5.0]])                        # (1,3)
    mz, lp = object_world_lowz(c, rot, pos, cap=8000)
    assert mz.shape == (1,) and lp.shape == (1, 3)
    assert np.isclose(mz[0], 4.0) and np.isclose(lp[0, 2], 4.0)


def test_object_world_lowz_cap_shape_only():
    v = np.random.default_rng(0).normal(size=(50, 3))
    mz, lp = object_world_lowz(v, np.eye(3)[None], np.zeros((1, 3)), cap=4)  # subsampled
    assert mz.shape == (1,) and lp.shape == (1, 3)


def test_lowest_point():
    pts = np.array([[[0, 0, 3.0], [0, 0, 1.0], [0, 0, 2.0]],
                    [[0, 0, -1.0], [0, 0, 5.0], [0, 0, 0.5]]])               # (2,3,3)
    mz, lp = lowest_point(pts)
    assert np.allclose(mz, [1.0, -1.0])
    assert np.allclose(lp[:, 2], [1.0, -1.0])


def test_surface_points_centroid_and_vertex():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    tri = np.array([[0, 1, 2]])
    assert np.allclose(surface_points(verts, tri, np.array([[1 / 3, 1 / 3, 1 / 3]])), [[1 / 3, 1 / 3, 0]])
    assert np.allclose(surface_points(verts, tri, np.array([[1.0, 0.0, 0.0]])), [[0, 0, 0]])


def test_parity_error():
    assert np.allclose(parity_error(np.array([[3.0, 4.0, 0.0]]), np.zeros((1, 3))), [5.0])


def test_node_coords():
    coords = node_coords(np.zeros(3), 0.5, (2, 2, 2))
    assert coords.shape == (2, 2, 2, 3)
    assert np.allclose(coords[0, 0, 0], [0, 0, 0])
    assert np.allclose(coords[1, 0, 0], [0.5, 0, 0])
    assert np.allclose(coords[1, 1, 1], [0.5, 0.5, 0.5])
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_geometry.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.debug._geometry`).

- [ ] **Step 3 : Écrire `src/viz/debug/_geometry.py` [full code]**

```python
"""Pure debug-only geometry helpers for the ``viz/debug`` viewers — numpy-only, NO viser, NO scipy.

These carry the non-contract debug math the viewers used to inline: the lowest WORLD point a rigid
object / the human must rest on z=0 (grounding overlay), the barycentric surface reference used to
colour the cloud by its parity error against the full SMPL forward, and the SDF grid node coords.
Callers convert quaternions to rotation matrices via ``core.viser_ops.quat_wxyz_to_R`` and pass the
matrices in — so this module stays viser-free AND does not re-roll the quat conversion."""
from __future__ import annotations

import numpy as np


def object_world_lowz(verts_local: np.ndarray, rot: np.ndarray, pos: np.ndarray,
                      cap: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame lowest WORLD point of a rigid object.

    ``verts_local`` (V,3) local vertices, ``rot`` (F,3,3) per-frame rotation, ``pos`` (F,3) per-frame
    translation. Returns ``(min_z (F,), low_point (F,3))``. ``verts_local`` is subsampled to ``cap``
    to bound cost on dense scans — a near-exact lowest point, enough for a debug marker."""
    v = verts_local
    if v.shape[0] > cap:
        v = v[np.random.default_rng(0).choice(v.shape[0], cap, replace=False)]
    world = np.einsum("fij,vj->fvi", rot, v) + pos[:, None, :]          # (F, V, 3)
    z = world[:, :, 2]
    lo = z.argmin(axis=1)
    return z.min(axis=1), world[np.arange(world.shape[0]), lo]


def lowest_point(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame lowest point of a moving point set ``points`` (F,P,3).
    Returns ``(min_z (F,), low_point (F,3))``."""
    z = points[:, :, 2]
    lo = z.argmin(axis=1)
    return z.min(axis=1), points[np.arange(points.shape[0]), lo]


def surface_points(verts: np.ndarray, tri_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Barycentric samples on a mesh: for each of N samples, ``bary``-weighted blend of its triangle's
    three ``verts``. ``tri_idx`` (N,3) vertex indices, ``bary`` (N,3) weights. Returns (N,3). This is
    the TRUE posed-surface reference the cloud viewer compares its mesh-free cloud against."""
    return np.einsum("nij,ni->nj", verts[tri_idx], bary)


def parity_error(posed: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-point L2 distance ``‖posed - ref‖`` (N,3),(N,3) -> (N,). The cloud's parity error."""
    return np.linalg.norm(posed - ref, axis=1)


def node_coords(origin: np.ndarray, spacing: float, shape) -> np.ndarray:
    """(Nx,Ny,Nz,3) local coords of every SDF grid node, given ``origin`` (3,), ``spacing`` and the
    grid ``shape`` (Nx,Ny,Nz)."""
    nx, ny, nz = shape
    xs = origin[0] + spacing * np.arange(nx)
    ys = origin[1] + spacing * np.arange(ny)
    zs = origin[2] + spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)
```

- [ ] **Step 4 : Lancer, vérifier le succès + numpy-only (sans viser)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_geometry.py -q`
Expected: PASS (6 tests).

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz.debug._geometry; assert 'viser' not in sys.modules; print('geometry viser-free ok')"`
Expected: `geometry viser-free ok`

- [ ] **Step 5 : Commit**

```bash
git add src/viz/debug/_geometry.py tests/test_viz_debug_geometry.py
git commit -m "feat(holov2): viz/debug/_geometry — helpers purs (lowz objet/humain, surface barycentrique, node_coords) testés"
```

---

### Task 3 : réécrire `scene.py` -> `debug/scene.py` (load / grounding debug)

Préserve : mesh SMPL posé, joints démo, squelette, objets, floor, **folder Grounding** (apply / **slider foot-pct** / floor / **lowest-point markers**), debug calibration (offsets human/object, médianes RAW). Player remplace la boucle Playback + keep-alive ; `viser_ops` remplace clouds/segments + le hack de masquage ; `_geometry` + `quat_wxyz_to_R` remplacent le `_object_world_lowz`/lowest-point inline.

**Files:**
- Create: `src/viz/debug/scene.py`
- Test: `tests/test_viz_debug_imports.py`  (créé ici — 1 fonction smoke par viewer, étendue aux Tasks 4–6)

**Interfaces:**
- Consumes : `core.player.Player`, `core.viser_ops.{quat_wxyz_to_R, hide, point_cloud, line_segments}`, `debug._geometry.{object_world_lowz, lowest_point}`, `debug._args.{add_scene_args, scene_from_args}`, internes `load`/`calibration` (drive délibéré).
- Produces : runnable `python -m src.viz.debug.scene` + `view_scene(spec, *, port, frame_step, max_frames)`.

- [ ] **Step 1 : Écrire le smoke d'import [full code]** — crée `tests/test_viz_debug_imports.py` :

```python
# tests/test_viz_debug_imports.py
"""Import smokes for the rewritten debug viewers: each module imports, exposes its entrypoint, and
references the CANONICAL core/ symbols (so the rewrite truly consumes the socle). Requires Phase A
(viz/core) merged. No screen needed — viser imports headless."""
from src.viz.core.colors import diverging, parity
from src.viz.core.player import Player


def test_scene_imports():
    from src.viz.debug import scene
    assert callable(scene.view_scene) and callable(scene.main)
    assert scene.Player is Player                       # consumes core/Player (no re-rolled player)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.debug.scene`).

- [ ] **Step 3 : Écrire `src/viz/debug/scene.py` [full code]**

```python
"""Scene preview viewer — visual debug of the ``load`` stage (rewritten onto ``viz/core``).

Given a ``SceneSpec`` it loads the ``RawMotion`` + builds the ``BodyModel`` and shows, per frame, the
posed SMPL-X mesh, the skeleton (FK bones + demo joints), the object(s) posed by their world poses,
the ground, AND the GROUNDING debug overlay (calibration offsets, a live foot-percentile slider, and
the lowest-point markers each entity must rest on z=0). Debug viewer: it deliberately DRIVES the
``load``/``calibration`` internals to surface these non-contract intermediates (the ARCHITECTURE.md
debug-viewer exception); viser stays confined to ``core/viser_ops`` + this module; the shared
Playback/keep-alive comes from ``core/Player``.

Run:
    python -m src.viz.debug.scene --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...prepare.contracts import SceneSpec
from ...prepare.config import CalibrationConfig
from ...prepare.calibration import build_calibration
from ...prepare.load import load
from ...prepare.load.smpl import build_body_model
from ..core.player import Player
from ..core.viser_ops import hide, line_segments, point_cloud, quat_wxyz_to_R
from ._args import add_scene_args, scene_from_args
from ._geometry import lowest_point, object_world_lowz


def view_scene(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200) -> None:
    import trimesh
    import viser

    raw = load(spec)
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"loaded {raw.source_format}: T={raw.n_frames}, showing {F} frames, "
          f"{len(raw.object_poses_raw)} object(s), parametric={raw.is_parametric}")

    # --- precompute per shown frame (bounded) ---
    body = build_body_model(raw.smpl_params, Path(spec.smpl_model_dir)) if raw.is_parametric else None
    faces = body.faces if body is not None else None
    parents = body.parents if body is not None else None
    n_demo = raw.joint_pos.shape[1]

    verts = None
    if body is not None:
        V = body.rest_vertices(raw.smpl_params).shape[0]
        verts = np.empty((F, V, 3), np.float32)
    demo_j = np.empty((F, n_demo, 3), np.float32)
    bones = np.empty((F, body.n_bones, 3), np.float32) if body is not None else None
    print("precomputing posed meshes/skeletons ...")
    for i, t in enumerate(frames):
        demo_j[i] = raw.joint_pos[t]
        if body is not None:
            verts[i] = body.posed_vertices(raw.smpl_params, t)
            bones[i] = body.bone_transforms(raw.smpl_params, t)[1]

    # objects are rigid: keep local mesh + per-frame world pose, update the transform per frame.
    objs = []  # (verts_local, faces, poses_frames (F, 7))
    for k in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[k]), force="mesh", process=False, skip_materials=True)
        vl = np.asarray(m.vertices, np.float32)
        fl = np.asarray(m.faces, np.int32)
        poses = np.asarray(raw.object_poses_raw[k], np.float32)[frames]
        objs.append((vl, fl, poses))

    bone_pairs = [(int(parents[j]), j) for j in range(body.n_bones) if parents[j] >= 0] if body else []

    # --- grounding debug: the calibration + per-frame floor clearances (RAW, pre-grounding) ---
    # Grounding is PER ENTITY: the human drops by calib.human_offset, ALL objects by the shared
    # calib.object_offset. These clearances let us SEE each entity land on z=0 (the human may float
    # while the objects already rest on the floor, hence the split human/object offsets).
    calib = build_calibration(raw, CalibrationConfig())                # body-free grounding
    human_offset = float(calib.human_offset)
    object_offset = float(calib.object_offset)                         # shared by all objects
    # Human lowest world z + lowest point per frame (surface if parametric, else demo joints).
    src = verts if verts is not None else demo_j
    human_minz, human_low = lowest_point(src)                          # (F,), (F, 3)
    # Human floor offset = a PERCENTILE of the lower mocap FOOT-JOINT height over the clip, dialled
    # live by a slider. The foot joint is robust to the SMPL sole penetration (toe-curl dips the mesh
    # BELOW the rest level, so chasing the lowest point over-lifts the human); the percentile targets
    # the RESTING/contact level instead. ``sole_med`` (the current method) is kept on-screen only.
    sole_med = float(np.median(human_minz))                            # current method, for contrast
    _foot = [i for i, n in enumerate(raw.joint_names) if n in ("L_Foot", "R_Foot")]
    lower_foot = raw.joint_pos[:, _foot, 2].min(axis=1) if _foot else human_minz   # (T,) lower foot z
    obj_minz, obj_low = [], []                                          # per object: (F,), (F, 3)
    for (vl, _fl, poses) in objs:
        rot = quat_wxyz_to_R(poses[:, 3:7])                            # (F,3,3) wxyz -> R
        mz, lp = object_world_lowz(vl, rot, poses[:, :3])
        obj_minz.append(mz); obj_low.append(lp)
    hz_med = float(np.median(human_minz))
    oz_med = [float(np.median(z)) for z in obj_minz]
    stature_str = f"{body.stature:.3f} m" if body is not None else "n/a"
    print(f"calibration: human_offset={human_offset:+.4f} m, human_stature={stature_str}, "
          f"object_offset={object_offset:+.4f}")
    print(f"  RAW clip-median lowest z: human={hz_med:+.4f}" +
          "".join(f", obj{k}={m:+.4f}" for k, m in enumerate(oz_med)))

    print("done. starting viser ...")
    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=4.0, height=4.0)
    with srv.gui.add_folder("Display"):
        show_mesh = srv.gui.add_checkbox("SMPL mesh", True)
        show_joints = srv.gui.add_checkbox("demo joints", True)
        show_bones = srv.gui.add_checkbox("skeleton", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    with srv.gui.add_folder("Grounding"):
        apply_ground = srv.gui.add_checkbox("apply grounding", True)
        foot_pct = srv.gui.add_slider("foot offset pct", 0, 100, 1, 50)   # percentile of lower foot z
        show_floor = srv.gui.add_checkbox("floor plane z=0", True)
        show_low = srv.gui.add_checkbox("lowest-point markers", True)
    info = srv.gui.add_markdown("")

    # persistent handles (Playback folder + keep-alive are owned by core.Player)
    floor_h = srv.scene.add_box("/floor", color=(170, 170, 178), dimensions=(4.0, 4.0, 0.004),
                                position=(0.0, 0.0, 0.0))
    hj = point_cloud(srv, "/joints", demo_j[0],
                     np.tile([[40, 200, 60]], (n_demo, 1)).astype(np.uint8), point_size=0.025)
    hobj = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
            for k, (vl, fl, _) in enumerate(objs)]
    nseg = max(len(bone_pairs), 1)
    skel_h = line_segments(srv, "/skeleton", np.zeros((nseg, 2, 3), np.float32),
                           np.tile([[[0, 120, 255]]], (nseg, 2, 1)).astype(np.uint8), line_width=3.0)
    # lowest-point markers: red = human sole, yellow = each object — the two things grounding rests
    # on z=0. Watching them separately is exactly how a shared offset reads vs a split one.
    low_h = point_cloud(srv, "/low_human", human_low[:1], np.array([[255, 40, 40]], np.uint8),
                        point_size=0.05)
    low_o = [point_cloud(srv, f"/low_obj{k}", obj_low[k][:1], np.array([[255, 210, 0]], np.uint8),
                         point_size=0.05) for k in range(len(objs))]
    body_h = [None]   # latest /body mesh handle (re-added per frame when shown: SMPL verts change)

    def render(f: int) -> None:
        on = apply_ground.value                              # grounding = drop each entity to z=0
        gh = float(np.percentile(lower_foot, foot_pct.value)) if on else 0.0   # foot-pct human z-shift
        go = [object_offset if on else 0.0 for _ in range(len(objs))]          # shared object z-shift
        dzh = np.array([0.0, 0.0, gh], np.float32)
        if body is not None and show_mesh.value:
            body_h[0] = srv.scene.add_mesh_simple("/body", verts[f] - dzh, faces,
                                                  color=(200, 200, 210), opacity=0.55, side="double")
        elif body_h[0] is not None:
            hide(body_h[0])
        hj.points = demo_j[f] - dzh
        hj.visible = show_joints.value
        if body is not None and show_bones.value and bone_pairs:
            seg = (np.stack([np.stack([bones[f, a], bones[f, b]]) for a, b in bone_pairs])
                   .astype(np.float32) - dzh)
            skel_h.points = seg
            skel_h.visible = True
        else:
            hide(skel_h)
        for k, (_, _, poses) in enumerate(objs):
            h = hobj[k]
            h.position = poses[f][:3] - np.array([0.0, 0.0, go[k]], np.float32)
            h.wxyz = poses[f][3:]
            h.visible = show_obj.value
        floor_h.visible = show_floor.value
        low_h.points = (human_low[f] - dzh)[None]
        low_h.visible = show_low.value
        for k in range(len(objs)):
            low_o[k].points = (obj_low[k][f] - np.array([0.0, 0.0, go[k]], np.float32))[None]
            low_o[k].visible = show_low.value

        hz = human_minz[f] - gh
        oz = [obj_minz[k][f] - go[k] for k in range(len(objs))]
        info.content = (
            f"**frame {frames[f]}** ({f + 1}/{F}) · grounding **{'ON' if on else 'OFF'}**\n\n"
            f"human offset = **foot-joint p{int(foot_pct.value)} = {gh:+.4f} m**  "
            f"(sole median for contrast: {sole_med:+.4f})\n\n"
            f"lowest z (this frame) — human sole **{hz:+.4f}**" +
            "".join(f", obj{k} **{z:+.4f}**" for k, z in enumerate(oz)) + " m\n\n"
            f"object offset (shared): {object_offset:+.4f} m" + ("" if objs else " (no objects)"))

    player = Player(srv, F, render, fps=20.0)
    for h in (show_mesh, show_joints, show_bones, show_obj, apply_ground, foot_pct,
              show_floor, show_low):
        h.on_update(lambda _=None: player.request_render())
    print(f"viser ready -> http://localhost:{port}")
    player.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le smoke, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py -q`
Expected: PASS (`test_scene_imports`).

- [ ] **Step 5 : Vérification VISUELLE manuelle (le viewer est visuel)** — lancer sur la démo avec `max_frames` bas et comparer à l'ancien `python -m src.viz.scene` :

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.debug.scene --motion-path <demo_smplx.npz> --model-dir <smplx_models> --dataset hodome --frame-step 4 --max-frames 30`
Vérifier dans le navigateur (`http://localhost:8080`) : (a) le slider **Playback** (core.Player) avance les frames, `play`/`fps` animent ; (b) folder **Display** : mesh/joints/skeleton/objects togglent ON/OFF (masquage propre via `hide`, pas de triangle résiduel) ; (c) folder **Grounding** : `apply grounding` baisse l'humain+objets sur z=0, le **slider foot-pct** déplace l'humain live, les **marqueurs rouge (sole)/jaune (objets)** se posent sur le plan, le panneau markdown affiche offsets+z courants ; (d) parité visuelle avec l'ancien viewer (mêmes couleurs/positions).

- [ ] **Step 6 : Commit**

```bash
git add src/viz/debug/scene.py tests/test_viz_debug_imports.py
git commit -m "feat(holov2): viz/debug/scene — réécrit sur core (Player + viser_ops + _geometry), grounding/foot-pct/markers préservés"
```

---

### Task 4 : réécrire `cloud.py` -> `debug/cloud.py` (bake point_cloud, parité)

Préserve : nuage humain skinné mesh-free posé par `pose_cloud`, **coloration par erreur de parité** vs surface SMPL pleine (via `core.colors.parity`), nuages objets rigides, ghost SMPL toggle, médian/p95. Garde **délibérément** l'import de l'op AVAL `targets.interaction.pose_cloud` (validation du chemin runtime).

**Files:**
- Create: `src/viz/debug/cloud.py`
- Modify: `tests/test_viz_debug_imports.py`  (append `test_cloud_imports`)

**Interfaces:**
- Consumes : `core.player.Player`, `core.colors.parity`, `core.viser_ops.{quat_wxyz_to_R, hide, point_cloud}`, `debug._geometry.{surface_points, parity_error}`, `debug._args.{add_scene_args, scene_from_args}`, `prepare.point_cloud.*` (drive du bake) **et** `targets.interaction.pose_cloud` (op aval, délibéré).
- Produces : runnable `python -m src.viz.debug.cloud` + `view_cloud(spec, corr_path, *, port, frame_step, max_frames, vmax)`.

- [ ] **Step 1 : Append du smoke (échec attendu)** — ajouter à `tests/test_viz_debug_imports.py` :

```python
def test_cloud_imports():
    from src.viz.debug import cloud
    assert callable(cloud.view_cloud) and callable(cloud.main)
    assert cloud.Player is Player and cloud.parity is parity    # consumes core/Player + core/colors.parity
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py::test_cloud_imports -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.debug.cloud`).

- [ ] **Step 3 : Écrire `src/viz/debug/cloud.py` [full code]**

```python
"""Point-cloud viewer — visual debug of the ``point_cloud`` bake (rewritten onto ``viz/core``).

Builds the subject's sparse-skinned human cloud (reusing the correspondence's sampling) and each
object's rigid cloud, then poses them per frame with the single ``pose_cloud`` op — the mesh-free,
torch-free runtime path. The human points are coloured by their parity error against the TRUE posed
SMPL surface (full forward) via ``core.colors.parity``, so one can SEE the LBS-on-cloud track the
body; the object points (rigid K=1) sit on the object surface. Debug viewer: it drives the bake AND
deliberately imports the DOWNSTREAM op ``targets.interaction.pose_cloud`` to exercise the exact
runtime path the solver uses. viser confined to ``core/viser_ops`` + this module.

Run:
    python -m src.viz.debug.cloud --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...prepare.contracts import SceneSpec
from ...prepare.config import CloudConfig
from ...prepare.load import load
from ...prepare.load.mesh import load_mesh
from ...prepare.load.smpl import build_body_model
from ...prepare.point_cloud import build_human_cloud, build_object_cloud
from ...prepare.point_cloud.correspondence import load_correspondence
from ...targets.interaction import pose_cloud          # DOWNSTREAM op (deliberate): the runtime cloud path
from ..core.colors import parity
from ..core.player import Player
from ..core.viser_ops import hide, point_cloud, quat_wxyz_to_R
from ._args import add_scene_args, scene_from_args
from ._geometry import parity_error, surface_points

_DEFAULT_CORR = Path(__file__).resolve().parents[3] / "cache" / "correspondence" / "corr_neutral.npz"


def _object_world(cloud, pose7: np.ndarray) -> np.ndarray:
    """(P,3) object cloud posed by one ``[x,y,z,qw,qx,qy,qz]`` world pose via the shared ``pose_cloud``."""
    R = quat_wxyz_to_R(np.asarray(pose7, np.float64)[3:7])              # (3,3), wxyz -> R
    return pose_cloud(cloud, R[None], np.asarray(pose7, np.float64)[:3][None])


def view_cloud(spec: SceneSpec, corr_path: Path, *, port: int = 8080, frame_step: int = 2,
               max_frames: int = 150, vmax: float = 0.02) -> None:
    import viser

    raw = load(spec)
    if not raw.is_parametric:
        raise ValueError("the human cloud needs a parametric body (SMPL params); this source has none")
    params = raw.smpl_params
    body = build_body_model(params, Path(spec.smpl_model_dir))
    _, sampling = load_correspondence(corr_path)
    human = build_human_cloud(body, sampling, CloudConfig())
    obj_clouds = [build_object_cloud(*load_mesh(p), CloudConfig()) for p in raw.object_mesh_paths]

    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F, N = len(frames), human.n_points
    V = body.rest_vertices(params).shape[0]
    tri_v = body.faces[sampling.tri_idx]                            # (N,3) for the surface reference
    posed = np.empty((F, N, 3), np.float32)
    verts = np.empty((F, V, 3), np.float32)
    colors = np.empty((F, N, 3), np.uint8)
    obj_posed = [np.empty((F, c.n_points, 3), np.float32) for c in obj_clouds]
    print(f"human cloud: {N} pts, K={human.n_influences}; {len(obj_clouds)} object cloud(s) "
          f"[{', '.join(str(c.n_points) for c in obj_clouds) or '-'}]; precomputing {F} frames ...")
    med = np.empty(F); p95 = np.empty(F)
    for i, t in enumerate(frames):
        v = body.posed_vertices(params, t)                          # (V,3) full SMPL forward (parity ref)
        ref = surface_points(v, tri_v, sampling.bary.astype(np.float64))
        pc = pose_cloud(human, *body.bone_transforms(params, t))    # (N,3) the mesh-free runtime path
        err = parity_error(pc, ref)
        verts[i], posed[i], colors[i] = v, pc, parity(err, vmax)
        med[i], p95[i] = np.median(err), np.percentile(err, 95)
        for k, c in enumerate(obj_clouds):
            obj_posed[k][i] = _object_world(c, raw.object_poses_raw[k][t])
    print(f"parity over clip: median {med.mean()*1000:.1f}mm, p95 {p95.mean()*1000:.1f}mm")

    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=4.0, height=4.0)
    with srv.gui.add_folder("Display"):
        show_human = srv.gui.add_checkbox("human cloud", True)
        show_objs = srv.gui.add_checkbox("object clouds", True)
        show_mesh = srv.gui.add_checkbox("SMPL surface (ghost)", False)
        size = srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
    info = srv.gui.add_markdown("")

    hum_h = point_cloud(srv, "/human", posed[0], colors[0], point_size=float(size.value))
    obj_h = [point_cloud(srv, f"/obj{k}", op[0],
                         np.tile([[255, 140, 0]], (op.shape[1], 1)).astype(np.uint8),
                         point_size=float(size.value)) for k, op in enumerate(obj_posed)]
    ghost_h = [None]   # latest /ghost handle (re-added per frame when shown: verts change)

    def render(f: int) -> None:
        hum_h.points, hum_h.colors, hum_h.point_size = posed[f], colors[f], float(size.value)
        hum_h.visible = show_human.value
        for k, h in enumerate(obj_h):
            h.points, h.point_size, h.visible = obj_posed[k][f], float(size.value), show_objs.value
        if show_mesh.value:
            ghost_h[0] = srv.scene.add_mesh_simple("/ghost", verts[f], body.faces,
                                                   color=(200, 200, 210), opacity=0.4, side="double")
        elif ghost_h[0] is not None:
            hide(ghost_h[0])
        info.content = (f"**frame {frames[f]}** ({f + 1}/{F})\n\n"
                        f"human parity err — median **{med[f]*1000:.1f}mm**, p95 **{p95[f]*1000:.1f}mm**\n\n"
                        f"human colour: blue 0 → red ≥ {vmax*1000:.0f}mm · objects: orange (rigid K=1)")

    player = Player(srv, F, render, fps=20.0)
    for h in (show_human, show_objs, show_mesh, size):
        h.on_update(lambda _=None: player.request_render())
    print(f"viser ready -> http://localhost:{port}")
    player.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument("--corr", type=Path, default=_DEFAULT_CORR, help="correspondence cache (.npz)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_cloud(spec, a.corr, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le smoke, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py -q`
Expected: PASS (`test_scene_imports`, `test_cloud_imports`).

- [ ] **Step 5 : Vérification VISUELLE manuelle** — comparer à l'ancien `python -m src.viz.cloud` :

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.debug.cloud --motion-path <demo_smplx.npz> --model-dir <smplx_models> --dataset hodome --frame-step 4 --max-frames 30`
Vérifier : (a) le nuage humain est coloré **bleu (0) → rouge (≥ vmax)** par l'erreur de parité (couleurs identiques à l'ancien `_heat` ⇒ `core.colors.parity`) ; (b) le nuage suit le corps (médian/p95 affichés cohérents) ; (c) nuages objets orange rigides ; (d) toggle `SMPL surface (ghost)` apparaît/se masque proprement ; (e) `point size` modifie la taille ; (f) Playback core.Player anime.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/debug/cloud.py tests/test_viz_debug_imports.py
git commit -m "feat(holov2): viz/debug/cloud — réécrit sur core (parity colormap + viser_ops), pose_cloud aval gardé délibérément"
```

---

### Task 5 : réécrire `sdf.py` -> `debug/sdf.py` (SDF pure-géométrie)

Préserve : son **propre parser `--mesh`/`--plane`**, la tranche mobile (axis/index), la bande shell, les lignes witness, le ghost mesh. **Aucune frame ⇒ pas de `core.Player`** (pas d'axe temps) : la spec elle-même liste pour sdf « `core/colors.diverging` + `viser_ops` » sans Player ; le keep-alive reste un bloc minimal (cf. Questions ouvertes : helper keep-alive frameless éventuel côté Phase A). Colormap signée via `core.colors.diverging` ; `node_coords` via `_geometry`.

**Files:**
- Create: `src/viz/debug/sdf.py`
- Modify: `tests/test_viz_debug_imports.py`  (append `test_sdf_imports`)

**Interfaces:**
- Consumes : `core.colors.diverging`, `core.viser_ops.{hide, line_segments, point_cloud}`, `debug._geometry.node_coords`, internes `prepare.sdf.build` / `prepare.load.mesh` (drive du builder).
- Produces : runnable `python -m src.viz.debug.sdf` (`--mesh`/`--plane`) + `view_sdf(sdf, margin, *, verts, faces, port)`.

- [ ] **Step 1 : Append du smoke (échec attendu)** — ajouter à `tests/test_viz_debug_imports.py` :

```python
def test_sdf_imports():
    from src.viz.debug import sdf
    assert callable(sdf.view_sdf) and callable(sdf.main)
    assert sdf.diverging is diverging                       # consumes core/colors.diverging
    assert callable(sdf.node_coords)                        # consumes the pure _geometry helper
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py::test_sdf_imports -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.debug.sdf`).

- [ ] **Step 3 : Écrire `src/viz/debug/sdf.py` [full code]**

```python
"""SDF viewer — visual debug of the ``prepare/sdf`` build (rewritten onto ``viz/core``).

Renders any ``SDF`` (object / terrain / flat ground) in its local frame:
  - a movable CROSS-SECTION slice of the grid, coloured by signed distance (blue = inside/negative,
    white = surface/zero, red = outside/positive): the zero-crossing must trace the surface,
  - the near-surface BAND shell (|d| < margin) coloured the same way,
  - WITNESS lines (band node -> its stored nearest surface point): every line must end ON the surface,
  - the source mesh as a ghost when there is one (a flat-ground plane SDF has no mesh).

Debug viewer with its OWN ``--mesh``/``--plane`` parser (it visualises a single SDF in its local
frame, not a motion). It drives the ``prepare/sdf`` builder once, then only reads the asset. There is
NO time axis -> it does NOT use ``core/Player``; it consumes ``core/colors.diverging`` +
``core/viser_ops`` + the pure ``_geometry.node_coords``. viser confined to ``core/viser_ops`` + this
module.

Run:
    python -m src.viz.debug.sdf --mesh <path.obj>   [--spacing 0.02] [--margin 0.05] [--port 8080]
    python -m src.viz.debug.sdf --plane <size_m>    [--spacing 0.05] [--margin 0.05] [--port 8080]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...prepare.contracts import SDF
from ...prepare.load.mesh import load_mesh
from ...prepare.sdf.build import build_plane_sdf, build_sdf
from ..core.colors import diverging
from ..core.viser_ops import line_segments, point_cloud
from ._geometry import node_coords


def view_sdf(sdf: SDF, margin: float, *, verts: np.ndarray | None = None,
             faces: np.ndarray | None = None, port: int = 8080) -> None:
    import viser

    spacing = sdf.spacing
    coords = node_coords(sdf.origin, spacing, sdf.grid.shape)     # (Nx,Ny,Nz,3)
    nx, ny, nz = sdf.grid.shape
    inside_pct = 100.0 * float((sdf.grid < 0).mean())

    # band shell (|d| < margin): coords, signed dist, stored witness — for the shell + witness lines
    mask = np.abs(sdf.grid) < margin
    band_xyz = coords[mask]; band_d = sdf.grid[mask]; band_w = sdf.witness[mask]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(band_xyz), min(500, len(band_xyz)), replace=False) if len(band_xyz) else []

    print(f"SDF '{sdf.name}': grid {nx}x{ny}x{nz} ({nx*ny*nz} nodes), spacing={spacing}, "
          f"inside%={inside_pct:.1f}, band nodes={int(mask.sum())}")

    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=2.0, height=2.0)

    with srv.gui.add_folder("Layers"):
        show_mesh = srv.gui.add_checkbox("mesh ghost", verts is not None)
        show_slice = srv.gui.add_checkbox("slice", True)
        show_band = srv.gui.add_checkbox("band shell", False)
        show_wit = srv.gui.add_checkbox("witness lines", False)
    with srv.gui.add_folder("Slice"):
        axis = srv.gui.add_dropdown("axis", ("X", "Y", "Z"), initial_value="Y")
        idx = srv.gui.add_slider("index", 0, ny - 1, 1, ny // 2)
    info = srv.gui.add_markdown("")

    band_h = point_cloud(srv, "/band", band_xyz.astype(np.float32),
                         diverging(band_d, margin), point_size=spacing * 0.6)
    if len(sub):
        seg = np.stack([band_xyz[sub], band_w[sub]], axis=1).astype(np.float32)   # (S,2,3) node->witness
        wcol = np.where((band_d[sub] < 0)[:, None, None],
                        np.array([[[60, 90, 255]]]), np.array([[[255, 80, 60]]]))
        wcol = np.broadcast_to(wcol, (len(sub), 2, 3)).astype(np.uint8)
    else:
        seg = np.zeros((1, 2, 3), np.float32); wcol = np.zeros((1, 2, 3), np.uint8)
    wit_h = line_segments(srv, "/witness", seg, wcol, line_width=1.5)

    def render(_=None) -> None:
        a = {"X": 0, "Y": 1, "Z": 2}[axis.value]
        idx.max = sdf.grid.shape[a] - 1
        i = min(int(idx.value), sdf.grid.shape[a] - 1)
        sl = [slice(None)] * 3; sl[a] = i
        pts = coords[tuple(sl)].reshape(-1, 3)
        dist = sdf.grid[tuple(sl)].reshape(-1)
        # slice geometry changes with axis/index -> re-add (justified) under the same name + visible.
        sh = point_cloud(srv, "/slice", pts.astype(np.float32), diverging(dist, margin),
                         point_size=spacing * 0.9)
        sh.visible = show_slice.value
        if verts is not None:
            srv.scene.add_mesh_simple("/mesh", verts.astype(np.float32), faces, color=(150, 150, 160),
                                      opacity=0.35 if show_mesh.value else 0.0, side="double")
        band_h.visible = show_band.value
        wit_h.visible = show_wit.value
        info.content = (
            f"**{sdf.name}** · grid {nx}×{ny}×{nz} · spacing {spacing} · margin {margin}\n\n"
            f"inside **{inside_pct:.1f}%**\n\n"
            f"slice **{axis.value}={i}**  ·  blue=inside (−) · white=surface (0) · red=outside (+)")

    for h in (show_mesh, show_slice, show_band, show_wit, axis, idx):
        h.on_update(render)
    render()
    print(f"viser ready -> http://localhost:{port}")
    while True:                       # frameless viewer: no Player; minimal keep-alive
        time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=Path, help="object/terrain mesh -> SDF")
    src.add_argument("--plane", type=float, metavar="SIZE", help="flat ground: SIZE×SIZE m plane SDF")
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if args.plane is not None:
        h = args.plane / 2.0
        sdf = build_plane_sdf([-h, -h], [h, h], args.spacing, args.margin, name="ground")
        view_sdf(sdf, args.margin, port=args.port)
    else:
        verts, faces = load_mesh(args.mesh)
        t0 = time.time()
        sdf = build_sdf(verts, faces, args.spacing, args.margin, name=args.mesh.stem)
        print(f"built mesh SDF in {time.time() - t0:.1f}s")
        view_sdf(sdf, args.margin, verts=verts, faces=faces, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le smoke, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py -q`
Expected: PASS (`test_scene_imports`, `test_cloud_imports`, `test_sdf_imports`).

- [ ] **Step 5 : Vérification VISUELLE manuelle** — plan + mesh, comparer à l'ancien `python -m src.viz.sdf` :

Run (plan exact, rapide) : `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.debug.sdf --plane 1.0 --spacing 0.1 --margin 0.05`
Vérifier : tranche colorée **bleu(−)/blanc(0)/rouge(+)** (= `core.colors.diverging`, identique à l'ancien `_diverging`), slider axis/index déplace la tranche, toggle band shell + witness lines (chaque ligne finit sur la surface), markdown info à jour. (Optionnel mesh : `--mesh <obj>` ⇒ ghost + zéro-crossing sur la surface.)

- [ ] **Step 6 : Commit**

```bash
git add src/viz/debug/sdf.py tests/test_viz_debug_imports.py
git commit -m "feat(holov2): viz/debug/sdf — réécrit sur core (diverging + viser_ops + node_coords), parser --mesh/--plane gardé"
```

---

### Task 6 : réécrire `hoim3_multiperson.py` -> `debug/hoim3.py` (multi-personnes)

Préserve : son **propre parser** (hoim3 construit son `SceneSpec` autrement que les flags partagés), le chargement **multi-personnes** + tous les objets, le pilotage de l'interne loader `build_person_params` (exception debug). Player remplace la boucle/keep-alive ; `viser_ops.hide` remplace le masquage degenerate des personnes.

**Files:**
- Create: `src/viz/debug/hoim3.py`
- Modify: `tests/test_viz_debug_imports.py`  (append `test_hoim3_imports`)

**Interfaces:**
- Consumes : `core.player.Player`, `core.viser_ops.hide`, `debug._args._g1_robot`, `... paths`, internes `prepare.load.load` + `prepare.load.datasets.hoim3.build_person_params` (drive délibéré).
- Produces : runnable `python -m src.viz.debug.hoim3` + `view(spec, *, port, frame_step, max_frames)`.

- [ ] **Step 1 : Append du smoke (échec attendu)** — ajouter à `tests/test_viz_debug_imports.py` :

```python
def test_hoim3_imports():
    from src.viz.debug import hoim3
    assert callable(hoim3.view) and callable(hoim3.main)
    assert hoim3.Player is Player                           # consumes core/Player
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py::test_hoim3_imports -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.debug.hoim3`).

- [ ] **Step 3 : Écrire `src/viz/debug/hoim3.py` [full code]**

```python
"""Multi-person debug view for HOI-M3 — validate the loading end-to-end (rewritten onto ``viz/core``).

HOI-M3 scenes have several people, each manipulating different objects; the single-human loader keeps
one person + all objects, which looks incoherent (the other objects are driven by people we don't
show). This view renders ALL people + all objects together (as the official toolbox does), so the
per-entity loading can be checked against a coherent scene. Debug viewer: it reuses the loader
(objects + one person) AND drives the load-stage internal ``build_person_params`` (the other people)
+ the SMPL-X body model — a deliberate stage-internal drive (ARCHITECTURE.md debug-viewer exception).
It keeps its OWN parser (hoim3 builds its SceneSpec differently from the shared scene flags). viser
confined to ``core/viser_ops`` + this module; Playback/keep-alive from ``core/Player``.

Run:
    python -m src.viz.debug.hoim3 --motion-path <..._human.npz> --model-dir <smplx_models>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...prepare.contracts import SceneSpec
from ... import paths
from ...prepare.load import load
from ...prepare.load.datasets.hoim3 import build_person_params
from ..core.player import Player
from ..core.viser_ops import hide
from ._args import _g1_robot

_PALETTE = [(70, 130, 220), (220, 90, 90), (90, 200, 120), (210, 170, 60), (170, 110, 210)]


def view(spec: SceneSpec, *, port: int = 8080, frame_step: int = 30, max_frames: int = 150) -> None:
    import trimesh
    import viser

    raw = load(spec)                                          # objects (+ one person) for free
    hd = np.load(str(spec.motion_path), allow_pickle=True)
    smpl_params = hd["smpl_params"]
    gender = str(hd["gender"])
    ids = [int(np.asarray(p["id"])) for p in smpl_params[0]]  # people present at frame 0
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"HOI-M3 multi-person: {len(ids)} people {ids}, {len(raw.object_poses_raw)} objects, "
          f"showing {F} frames")

    # Per-person posed SMPL-X meshes for the shown frames (real forward).
    persons = []  # (verts (F,V,3), faces, color)
    for k, pid in enumerate(ids):
        params, body = build_person_params(smpl_params, pid, gender, Path(spec.smpl_model_dir),
                                           spec.smplh_dir, spec.smpl2smplx_pkl)
        verts = np.stack([body.posed_vertices(params, t) for t in frames]).astype(np.float32)
        persons.append((verts, body.faces, _PALETTE[k % len(_PALETTE)]))
        print(f"  person {pid}: posed {F} frames")

    # Objects: centred local mesh + per-frame Z-up pose, exactly as the loader produced them.
    objs = []  # (verts_local, faces, poses (F,7))
    for kk in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[kk]), force="mesh", process=False, skip_materials=True)
        objs.append((np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                     np.asarray(raw.object_poses_raw[kk], np.float32)[frames]))

    print("starting viser ...")
    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=6.0, height=6.0)
    with srv.gui.add_folder("Display"):
        show_people = srv.gui.add_checkbox("people", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    info = srv.gui.add_markdown("")

    oh = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
          for k, (vl, fl, _) in enumerate(objs)]
    person_h = [None] * len(persons)   # latest /person{k} handle (re-added per frame: verts change)

    def render(f: int) -> None:
        for k, (verts, faces, color) in enumerate(persons):
            if show_people.value:
                person_h[k] = srv.scene.add_mesh_simple(f"/person{k}", verts[f], faces,
                                                        color=color, side="double")
            elif person_h[k] is not None:
                hide(person_h[k])
        for k, (_, _, poses) in enumerate(objs):
            oh[k].position = poses[f][:3]
            oh[k].wxyz = poses[f][3:]
            oh[k].visible = show_obj.value
        info.content = f"**frame {frames[f]}** ({f + 1}/{F}) — {len(persons)} people"

    player = Player(srv, F, render, fps=20.0)
    for h in (show_people, show_obj):
        h.on_update(lambda _=None: player.request_render())
    print(f"viser ready -> http://localhost:{port}")
    player.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolute, or relative to [datasets.hoim3].motion in paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="SMPL-X model dir; default: paths.toml [models].smplx")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=150)
    a = ap.parse_args()
    try:
        cfg = paths.load_paths()
    except FileNotFoundError:
        # paths.toml only HARD-required for a default: missing model-dir or a relative motion.
        if a.model_dir is None or not Path(a.motion_path).is_absolute():
            raise
        cfg = {}
    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion("hoim3", a.motion_path, cfg)
    spec = SceneSpec(dataset="hoim3", motion_path=motion, robot=_g1_robot(), smpl_model_dir=model_dir,
                     smplh_dir=paths.smplh_dir(cfg), smpl2smplx_pkl=paths.smpl2smplx_pkl(cfg))
    view(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le smoke, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_debug_imports.py -q`
Expected: PASS (`test_scene_imports`, `test_cloud_imports`, `test_sdf_imports`, `test_hoim3_imports`).

- [ ] **Step 5 : Vérification VISUELLE manuelle** — comparer à l'ancien `python -m src.viz.hoim3_multiperson` :

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.debug.hoim3 --motion-path <hoim3_..._human.npz> --model-dir <smplx_models> --frame-step 60 --max-frames 10`
Vérifier : **toutes les personnes** (couleurs distinctes de la palette) + **tous les objets** posés en scène cohérente, toggles people/objects propres (masquage via `hide`), Playback core.Player anime, markdown affiche le nombre de personnes.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/debug/hoim3.py tests/test_viz_debug_imports.py
git commit -m "feat(holov2): viz/debug/hoim3 — réécrit sur core (Player + viser_ops.hide), parser+multi-personnes préservés"
```

---

### Task 7 : supprimer les 5 anciens fichiers (en dernier) + repointer les importeurs + doc

Après parité confirmée (Tasks 3–6), supprimer les 5 anciens `src/viz/*.py`, **repointer les importeurs restants** de `_scene_args` (le viewer prod `viz/viewer.py` + son test `test_viewer_bake.py` le tirent encore — cf. Questions ouvertes sur le couplage Phase C), et mettre à jour la **sous-section « debug viewers » de VIZ.md** (PAS la section archi, propriété Phase A).

**Files:**
- Delete: `src/viz/scene.py`, `src/viz/cloud.py`, `src/viz/sdf.py`, `src/viz/hoim3_multiperson.py`, `src/viz/_scene_args.py`
- Modify: `src/viz/viewer.py`  (ligne 45 : repointer l'import `_scene_args` -> `debug/_args`, pour garder `viewer.py`/`test_viewer_bake.py` verts jusqu'à ce que Phase C remplace `viewer.py`)
- Modify: `docs/VIZ.md`  (sous-section debug viewers -> nouveaux chemins `viz/debug/…`)

**Interfaces:**
- Consumes : rien de nouveau.
- Produces : arbre `viz/` sans dette debug ; seuls `viz/debug/*` + `viz/core/*` (Phase A) + `viz/app.py`/`viewer.py` (prod, Phase C) subsistent.

- [ ] **Step 1 : Vérifier qu'aucun code (hors les anciens fichiers eux-mêmes + `viewer.py:45`) ne référence les anciens modules**

Run: `cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2 && grep -rn --include=*.py -E "viz\.(scene|cloud|sdf|hoim3_multiperson|_scene_args)|from \._scene_args|from \.(scene|cloud|sdf|hoim3_multiperson) import" src tests`
Expected: seules lignes restantes = `src/viz/viewer.py:45` (repointé au Step 2) + les anciens fichiers à supprimer. (Si un AUTRE importeur apparaît : le repointer de même avant suppression.)

- [ ] **Step 2 : Repointer `src/viz/viewer.py` ligne 45**

```python
from .debug._args import add_scene_args, scene_from_args
```

- [ ] **Step 3 : Supprimer les 5 anciens fichiers**

```bash
cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2
git rm src/viz/scene.py src/viz/cloud.py src/viz/sdf.py src/viz/hoim3_multiperson.py src/viz/_scene_args.py
```

- [ ] **Step 4 : Mettre à jour la sous-section « Visualiseurs de debug » de VIZ.md** (NE PAS toucher la section archi/seam) — remplacer la liste actuelle par les nouveaux chemins + entrypoints, ex. :

```markdown
## Visualiseurs de debug incrémentaux (par étape) — réécrits sur `viz/core`

Des viewers viser **focalisés** valident chaque étape (consommateurs purs, viser confiné à
`core/viser_ops` + le viewer, socle `core/` partagé : Player/colors/viser_ops). Ils gardent le droit
de **piloter les internes de l'étage** qu'ils visualisent (exception ARCHITECTURE.md) :
- `viz/debug/scene.py` : étape **load/grounding** (mesh SMPL posé, squelette, objets, sol, debug
  grounding : offsets, slider foot-percentile, marqueurs point-bas). `python -m src.viz.debug.scene`.
- `viz/debug/cloud.py` : bake **`point_cloud`** — nuage humain posé par `pose_cloud` (coloré par la
  parité vs surface SMPL pleine, `core.colors.parity`) + nuages objets rigides.
  `python -m src.viz.debug.cloud`.
- `viz/debug/sdf.py` : build **`prepare/sdf`** — tranche/bande/witness, `core.colors.diverging`
  (parser propre `--mesh`/`--plane`, pas de frames donc pas de Player). `python -m src.viz.debug.sdf`.
- `viz/debug/hoim3.py` : **load multi-personnes** HOI-M3 (toutes les personnes + objets).
  `python -m src.viz.debug.hoim3`.
- `viz/debug/_args.py` : glue CLI partagée (ex-`_scene_args.py`) ; `viz/debug/_geometry.py` : helpers
  géométriques purs testés (point-bas, surface barycentrique de parité, node_coords).
```

- [ ] **Step 5 : Vérifier l'arbre vert (tests debug + importeurs repointés)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_scene_args.py tests/test_viz_debug_geometry.py tests/test_viz_debug_imports.py -q`
Expected: PASS (tout vert).

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.viz.viewer; print('viewer import ok (repointed _scene_args)')"`
Expected: `viewer import ok (repointed _scene_args)`

Run: `cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2 && grep -rn --include=*.py -E "viz\.(scene|cloud|sdf|hoim3_multiperson|_scene_args)|from \._scene_args" src tests || echo "no dangling refs"`
Expected: `no dangling refs`.

- [ ] **Step 6 : Commit**

```bash
git add -A
git commit -m "refactor(holov2): viz — supprime les anciens viewers debug (scene/cloud/sdf/hoim3/_scene_args), repointe viewer.py + VIZ.md"
```

---

## Self-Review

**1. Spec coverage** (étape 6 de migration `2026-06-30-viz-architecture-design.md` + le périmètre du prompt) :
- `_scene_args.py` -> `debug/_args.py` (move, `add_scene_args`/`_g1_robot`/`scene_from_args` gardés, import-smoke) → Task 1. ✅
- `scene.py` -> `debug/scene.py` : grounding folder + slider foot-pct + lowest-point markers préservés ; sur `core/Player`+`colors?`+`viser_ops` ; `_object_world_lowz`/lowest-point portés en pur testé → Task 2 + Task 3. ✅
- `cloud.py` -> `debug/cloud.py` : parité via `core.colors.parity` ; `targets.pose_cloud` aval gardé délibérément (noté) → Task 4. ✅
- `sdf.py` -> `debug/sdf.py` : parser `--mesh`/`--plane` propre, slice/band/witness, `core.colors.diverging` + `viser_ops` + `_geometry.node_coords` → Task 5. ✅
- `hoim3_multiperson.py` -> `debug/hoim3.py` : multi-personnes + parser propre, drive interne `build_person_params` → Task 6. ✅
- DELETE des 5 anciens **en dernier**, après parité → Task 7. ✅
- Chaque viewer réécrit consomme `core/` (Player sauf sdf frameless, colors, viser_ops) + garde son trimesh lazy + son droit de piloter les internes d'étage → Tasks 3–6. ✅
- Helpers purs testés (le prompt : `_object_world_lowz` + parity/lowest-point math) portés en fonctions pures → `debug/_geometry.py` (5 fns) + `test_viz_debug_geometry.py` → Task 2. ✅
- VIZ.md : seule la sous-section debug viewers mise à jour (archi intouchée) → Task 7. ✅
- Séquence imposée : `_args` d'abord, viewers un par un (chacun runnable + committable indépendamment), anciens supprimés en dernier → Tasks 1→7. ✅

**2. Placeholder scan** : aucun `TBD`/`TODO`/`...` non-Python ; tout step porte code réel + commande exacte (FAIL/PASS) ; les viewers réécrits sont du code complet (pas d'abrégé). ✅

**3. Type-name consistency avec core/** : `Player(server, n_frames, render, *, fps)` + `.frame`/`.request_render()`/`.run()` ; `quat_wxyz_to_R` ; `hide` ; `point_cloud(server, name, points, colors, *, point_size)` ; `line_segments(server, name, segments, colors, *, line_width)` ; `parity(err, vmax)` ; `diverging(values, vmax)` — noms canoniques Phase A, identiques dans les 4 viewers et les smokes. Helpers debug : `object_world_lowz`/`lowest_point`/`surface_points`/`parity_error`/`node_coords` cohérents test↔impl↔appelants. ✅

**4. Profondeur d'import** (piège `debug/` = +1 niveau) : tous les `..prepare`/`..targets`/`.. import paths` des fichiers déplacés réécrits en `...prepare`/`...targets`/`... import paths` ; `..core.*` pour le socle ; `._args`/`._geometry` (même niveau). ✅

**5. Runnable à chaque étape** : copie-puis-supprime-en-dernier ⇒ anciens + nouveaux coexistent durant Tasks 1–6 ; `viewer.py`/`test_viewer_bake.py` (importeurs de `_scene_args`) restent verts car la suppression de `_scene_args.py` n'a lieu qu'en Task 7, accompagnée du repointage de `viewer.py`. ✅

## Questions ouvertes (coordination Phase A / Phase C)

1. **Signature exacte de `core.Player`** : ce plan suppose `Player(server, n_frames, render, *, fps=20.0)` avec `.frame`, `.request_render()`, `.run()` (render-callback `render(f:int)` ; le Player possède le folder Playback + keep-alive). Si Phase A expose plutôt un Player orienté `Layer`/`UiState` sans render-callback brut, les 3 viewers à frames (scene/cloud/hoim3) devront passer un mini-adaptateur. À **confirmer avec Phase A** avant Tasks 3/4/6.
2. **Signatures `viser_ops.point_cloud`/`line_segments`** (kwargs `point_size`/`line_width`, handles à `.points/.colors/.visible` mutables) et **`hide(handle)`** (fixe `.visible=False`) : supposées comme ci-dessus ; à aligner sur l'API réelle Phase A (sinon ajuster les appels — purement mécanique).
3. **`sdf.py` frameless sans Player** : la spec impose « chaque viewer utilise core/Player » mais sdf n'a **pas d'axe temps** (sliders axis/index, pas de play). Choix retenu : sdf consomme `colors`+`viser_ops` mais garde un keep-alive minimal `while True: sleep(1)`. Si Phase A juge la dette keep-alive devant aussi être tuée pour les viewers frameless, exposer un `core.player.serve_forever(server)` (ou `Player.keep_alive(server)`) et sdf l'appellera — sinon le bloc 2-lignes reste.
4. **Couplage Phase C (`viz/viewer.py`)** : le viewer **prod** importe encore `_scene_args` (ligne 45) et sera **supprimé** par Phase C (remplacé par `app.py`). Task 7 le repointe vers `debug/_args` pour rester vert si Phase D merge avant Phase C ; si Phase C a déjà supprimé `viewer.py`, ce repointage est **sans objet** (ignorer le Step 2 de Task 7). À séquencer au merge.
5. **`node_coords` candidat `core/`** : promu ici dans `debug/_geometry.py` (pour ne pas empiéter sur `core/` que possède Phase A). La couche prod roadmap `sdf_iso` en aura aussi besoin ⇒ Phase A pourra plus tard le hisser dans `core/` et `debug/sdf.py` ré-importera depuis là. Aucune action requise tant que la couche `sdf_iso` n'existe pas.
