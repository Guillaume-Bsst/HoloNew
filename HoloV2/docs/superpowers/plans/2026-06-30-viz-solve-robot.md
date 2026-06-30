# viz — seam solve + couche robot (PHASE B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **DÉPEND DE PHASE A** (`2026-06-30-viz-architecture-design.md`, étape de migration 1-3 : `core/`, `model.py`, `sources.py`, `layers/` prod, `app.py`). Ce plan CONSOMME les types/modules de Phase A par leurs noms canoniques (cf. « Dépendances Phase A » plus bas) ; il ne les redéfinit jamais. Écrit en supposant Phase A implémentée + mergée.

**Goal:** Brancher l'étage `solve/` dans le view-model du viz (étape de migration #4 + roadmap #1/#2 du design). Concrètement : (1) une fonction PURE `build_solved_frame(traj, ev, ctx, f) -> SolvedFrame` qui LIT `SolveTrajectory[f]` (q, object_poses, FrameInfo) et RÉUTILISE `targets.Evaluator` (+ FK `ctx.robot`) pour l'« atteint » — **zéro nouvelle logique de retargeting** ; (2) étendre `BakeSource(..., solve=True)` pour exécuter `solve.runner.solve` une fois et remplir `VizFrame.solved` ; (3) une couche 3D `RobotLayer` (ViserUrdf) qui affiche le G1 résolu ; (4) un panel 2D `CostDashboard` (matplotlib → image viser) ; (5) câbler un flag `--solve` dans `app.py`. Première valeur neuve : **le robot résolu superposé + les diagnostics solve**.

**Architecture:** `viz/sources.py` gagne `build_solved_frame` (numpy-only, pur) et un chemin `solve=True` dans `BakeSource` (exécute `solve` 1×, construit `Evaluator` 1×, attache un `SolvedFrame` par frame ; `solved=None` sinon — tout le pré-solve marche sans solveur). `viz/layers/robot.py` (couche 3D) et `viz/panels/cost_dashboard.py` (panel 2D) sont des CONSOMMATEURS minces du view-model (viser/ViserUrdf/matplotlib confinés ici). `app.py` câble `--solve` → `BakeSource(solve=True)` + `RobotLayer` + `CostDashboard`. Fil rouge Phase A inchangé : `Source → VizFrame → Layers/Panels`, `Player` orchestre, `core/` partagé.

**Tech Stack:** Python, numpy (float64), scipy (`Rotation`, conversion quat wxyz→matrice — déjà le pattern dans `viz/scene.py`/`viz/cloud.py`), pinocchio (caché dans `ctx.robot`, jamais importé par `viz`), **viser 1.0.30 + `viser.extras.ViserUrdf` + yourdfpy** (rendu robot — meshes complets, comme V1 ; vérifié présents dans l'env), **matplotlib 3.10.9 backend `Agg`** (graphe coût → image, car **plotly est ABSENT de l'env** : on n'utilise PAS `gui.add_plotly`), pytest. Env python : `~/.holonew_deps/miniconda3/envs/holonew/bin/python`. Tests data-gated (HODome/SMPL-X/corr/URDF, comme `tests/test_solve_runner.py`).

## Global Constraints

- **Code/commentaires/docstrings en ANGLAIS** (les fichiers sous `docs/` sont la seule exception au français de CLAUDE.md ; le code de Phase A/B suit la règle générale — ce plan met le code en anglais, conforme à la consigne de phase).
- **`viz` = consommateur pur** (règle d'or 6) : viser confiné à `core/viser_ops`, `layers/`, `panels/`, `app.py`, `debug/`. `model.py`/`sources.py` restent **numpy-only** (importables/testables sans écran ; scipy OK, viser/matplotlib INTERDITS).
- **`SolvedFrame` ne fait que LIRE `SolveTrajectory` et RÉUTILISER `Evaluator`** — aucune nouvelle logique de retargeting (l'« atteint » vit dans `targets.Evaluator`, déjà testé). FK robot via `ctx.robot` (protocole `RobotModel`) + kernel public `targets.pose_cloud`.
- `viz` n'importe que des **surfaces publiques** : `prepare.contracts`, `targets.Evaluator`/`targets.contracts`/`targets.pose_cloud`, `solve.runner.solve`/`solve.contracts`/`solve.config` ; jamais un interne de l'étage (`solve.loop`, `solve.retract`, `targets.interaction.*`…).
- **`targets`/`solve` INCHANGÉS** par ce redesign.
- Compute en **float64** ; arrays de view-model en float64 (numpy-only).
- Imports **relatifs** dans `src/` ; **absolus** (`from src.…`) dans `tests/`.
- Tests dans **`HoloV2/tests/`**, lancés depuis `HoloV2/` avec l'env `holonew`, **`max_frames` très bas** (`T = min(2, n_frames)`, cf. mémoire `run-tests-low-max-frames`).
- Commits **conventionnels** (préfixe `feat(holov2): viz/…`). **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic). Auteur : `Guillaume-Bsst`.

## Dépendances Phase A (CONSOMMÉES, jamais redéfinies)

- `src/viz/model.py` : `VizContext` (champs incl. `robot_urdf_path: Path`, `has_solve: bool`), `VizFrame` (frozen ; champ `solved: SolvedFrame | None`), **`SolvedFrame`** (frozen) avec EXACTEMENT les champs : `q:(nq,)`, `object_poses:(N,7)`, `robot_points_world:(M,3)`, `link_transforms:(L,4,4)`, `style_achieved:StyleEval`, `contact_achieved:ContactEval`, `cost:float`, `cost_by_term:dict`, `n_iters:int`, `status:str`.
- `src/viz/core/layer.py` : `Layer` Protocol (`folder: str` ; `setup(self, server, gui, ctx: VizContext) -> None` ; `update(self, frame: VizFrame, ui: UiState) -> None`), `UiState`.
- `src/viz/core/{colors,viser_ops,player}.py` ; `src/viz/sources.py` : **`BakeSource(spec, config, *, solve=False, ...)`** (exécute `prepare` 1×, bake `trace_frame` par frame, expose `get(i)->VizFrame`, `n_frames`, `context`). `src/viz/app.py` : `run_app(..., solve=False)`.

**Hypothèses assumées sur l'API Phase A (à confirmer au merge ; isolées dans la glu `BakeSource`, jamais dans `build_solved_frame`) :**
1. `BakeSource` détient après `prepare` : `grounded` (GroundedScene), `ctx` (InteractionContext), `spec` (SceneSpec), et bake une liste interne de `FrameTrace` (un par frame montrée, frames CONTIGUËS `range(n_frames)` — requis pour le warm-start séquentiel de `solve`). Chaque `FrameTrace.targets` est le `FrameTargets` de la frame → réutilisé tel quel pour `solve` (pas de recompute).
2. `BakeSource` expose la liste complète des `VizFrame` bakées (pour `CostDashboard.setup`), via une propriété `frames` (ou équivalent). Si Phase A ne l'expose pas, l'ajouter est une 1-ligne (les frames sont déjà en mémoire).
3. `VizContext.robot_urdf_path = spec.robot.urdf_path` et `VizContext.has_solve = solve` sont remplis par `BakeSource` (Phase A).

## File Structure

```
HoloV2/
  src/viz/
    sources.py            # MODIFY : + build_solved_frame() (pur) + chemin BakeSource(solve=True)
    layers/
      robot.py            # CREATE : RobotLayer (ViserUrdf) — couche 3D du G1 résolu
    panels/
      __init__.py         # CREATE si absent
      cost_dashboard.py   # CREATE : stack_cost_terms() (pur) + CostDashboard (matplotlib→image viser)
    app.py                # MODIFY : flag --solve -> BakeSource(solve=True) + RobotLayer + CostDashboard
  tests/
    test_viz_solved_frame.py   # CREATE : build_solved_frame (déterminisme/formes/atteint) + smoke BakeSource(solve=True)
    test_viz_cost_dashboard.py # CREATE : stack_cost_terms (pur, sans matplotlib/viser)
  docs/VIZ.md             # MODIFY : APPEND une section DISTINCTE (robot + cost_dashboard)
```

---

### Task 1 : `build_solved_frame` + `BakeSource(solve=True)` — la seam solve

**Files:**
- Modify: `src/viz/sources.py`
- Test: `tests/test_viz_solved_frame.py`

**Interfaces:**
- Consumes (Phase A) : `SolvedFrame`, `VizContext`, `VizFrame`, `BakeSource`.
- Consumes (déjà mergé) : `solve.runner.solve`, `solve.config.SolveConfig`, `solve.contracts.SolveTrajectory`/`FrameInfo`, `targets.Evaluator`, `targets.pose_cloud`, `targets.contracts.StyleEval`/`ContactEval`, `prepare.contracts.InteractionContext`, `ctx.robot.link_transforms`.
- Produces : `build_solved_frame(traj, ev, ctx, f) -> SolvedFrame` (pur, numpy-only) ; `BakeSource(..., solve=True)` qui remplit `VizFrame.solved`.

**Notes de contrat (load-bearing) :**
- `SolveTrajectory.object_poses[f]` est `(N, 7)` = `[x,y,z, qw,qx,qy,qz]` (**quat wxyz**). `Evaluator.contacts` attend `(q, object_rot (N,3,3), object_pos (N,3))` — PAS un pose7 packé. La conversion wxyz→matrice se fait via scipy (`R.from_quat(q[:, [1,2,3,0]]).as_matrix()`, le pattern de `viz/scene.py:37`), IDENTIQUE à ce que fait `solve.loop.evaluate` en interne — on ne l'importe pas (interne solve), on reproduit la même conversion publique-side.
- `style_achieved = ev.style(q)` est EXACTEMENT le recompute prouvé dans `tests/test_solve_runner.py:69` (`ev.style(traj.qpos[f]).position`).
- `robot_points_world` : pose le `ctx.robot_cloud` (K=1, parts en ordre FK des liens) via `pose_cloud(ctx.robot_cloud, rot, pos)` avec `(rot, pos) = ctx.robot.link_transforms(q)` — exactement comme `targets/interaction/eval.py:87`.
- `link_transforms` : empilement homogène `(L,4,4)` depuis `ctx.robot.link_transforms(q) -> (rot (L,3,3), pos (L,3))`.

- [ ] **Step 1 : Écrire le test** (`tests/test_viz_solved_frame.py`)

```python
"""build_solved_frame : la seam solve du viz. Data-gated (HODome / SMPL-X / corr / G1 URDF, comme
test_solve_runner). Vérifie déterminisme (build x2 identique), formes/dtypes, et que l'« atteint »
(style_achieved/contact_achieved) provient bien d'un appel Evaluator direct (pattern test_solve_runner).
Un smoke-test BakeSource(solve=True) verrouille la glu (dépend de Phase A : BakeSource)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets.evaluator import Evaluator
from src.targets.pipeline import process_frame
from src.solve.config import SolveConfig
from src.solve.runner import solve
from src.viz.model import SolvedFrame
from src.viz.sources import build_solved_frame
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX

_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick():
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome / SMPL-X / corr / G1 URDF absent")


def _prepare_solve(tmp_path):
    """prepare -> quelques FrameTargets -> solve -> (g, ctx, spec, traj, ev, T). T bas (suite rapide)."""
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())
    T = min(2, g.n_frames)
    frame_targets = [process_frame(g, ctx, spec.robot, f) for f in range(T)]
    traj = solve(g, ctx, frame_targets, SolveConfig(), robot_name="g1")
    ev = Evaluator(ctx, "g1")
    return g, ctx, spec, traj, ev, T


@_SKIP
def test_build_solved_frame_shapes_and_achieved(tmp_path):
    g, ctx, spec, traj, ev, T = _prepare_solve(tmp_path)
    L = len(ctx.robot.link_names)
    M = ctx.robot_cloud.n_points
    N = len(ctx.object_clouds)
    for f in range(T):
        sf = build_solved_frame(traj, ev, ctx, f)
        assert isinstance(sf, SolvedFrame)
        # formes / dtypes
        assert sf.q.shape == (ctx.robot.nq,) and sf.q.dtype == np.float64
        assert sf.object_poses.shape == (N, 7)
        assert sf.robot_points_world.shape == (M, 3)
        assert np.isfinite(sf.robot_points_world).all()
        assert sf.link_transforms.shape == (L, 4, 4)
        assert np.allclose(sf.link_transforms[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]))  # ligne homogène
        # diagnostics FrameInfo
        assert isinstance(sf.cost, float)
        assert isinstance(sf.cost_by_term, dict)
        assert isinstance(sf.n_iters, int) and isinstance(sf.status, str)
        # « atteint » == appel Evaluator direct (le pattern test_solve_runner)
        assert np.allclose(sf.style_achieved.position, ev.style(traj.qpos[f]).position)
        assert sf.contact_achieved.field.n_points == M


@_SKIP
def test_build_solved_frame_deterministic(tmp_path):
    g, ctx, spec, traj, ev, T = _prepare_solve(tmp_path)
    for f in range(T):
        a = build_solved_frame(traj, ev, ctx, f)
        b = build_solved_frame(traj, ev, ctx, f)
        assert np.array_equal(a.q, b.q)
        assert np.array_equal(a.object_poses, b.object_poses)
        assert np.array_equal(a.robot_points_world, b.robot_points_world)
        assert np.array_equal(a.link_transforms, b.link_transforms)
        assert np.array_equal(a.style_achieved.position, b.style_achieved.position)
        assert np.array_equal(a.contact_achieved.field.distance, b.contact_achieved.field.distance)


@_SKIP
def test_bake_source_solve_fills_solved(tmp_path):
    """Smoke : BakeSource(solve=True).get(f).solved est non-None + cohérent (dépend de Phase A)."""
    from src.viz.sources import BakeSource
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    src = BakeSource(spec, PrepareConfig(), solve=True)
    assert src.context.has_solve is True
    vf = src.get(0)
    assert vf.solved is not None
    assert vf.solved.q.shape[0] == src.context.__class__ is not None  # q présent
    assert vf.solved.status != ""
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_solved_frame.py -q`
Expected: FAIL (`ImportError: cannot import name 'build_solved_frame' from src.viz.sources`). *(Si les données sont absentes : tests SKIPPED — l'import échoue quand même au collect, donc ERROR sur l'import manquant ; après Step 3 ils passent/skippent proprement.)*

- [ ] **Step 3 : Implémenter `build_solved_frame` dans `src/viz/sources.py`**

Ajouter ces imports en tête de `sources.py` (à fusionner avec les imports Phase A ; tous numpy-only / scipy — AUCUN viser) :

```python
import numpy as np
from scipy.spatial.transform import Rotation as _R

from ..prepare.contracts import InteractionContext
from ..targets import Evaluator, pose_cloud
from ..solve.contracts import SolveTrajectory
from .model import SolvedFrame
```

Ajouter la fonction pure (placement : après les imports, niveau module) :

```python
def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions wxyz -> (N, 3, 3) rotations. Reordonne wxyz -> xyzw pour scipy.
    Identique à la conversion interne de ``solve.loop.evaluate`` (qu'on NE importe pas — interne solve)."""
    q = np.asarray(quat_wxyz, np.float64).reshape(-1, 4)
    if q.shape[0] == 0:
        return np.zeros((0, 3, 3), np.float64)
    return _R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def build_solved_frame(traj: SolveTrajectory, ev: Evaluator, ctx: InteractionContext,
                       f: int) -> SolvedFrame:
    """Build the post-solve ``SolvedFrame`` for frame ``f``. PURE: only READS ``SolveTrajectory`` and
    REUSES ``targets.Evaluator`` (+ FK ``ctx.robot``) — no new retargeting logic.

    - ``q`` / ``object_poses`` / cost-diagnostics : straight from ``traj``.
    - ``style_achieved`` = ``ev.style(q)`` (the proven recompute, cf. test_solve_runner).
    - ``contact_achieved`` = ``ev.contacts(q, object_rot, object_pos)`` (object_poses wxyz -> matrices).
    - ``robot_points_world`` : the M control points placed by FK @ q (``pose_cloud(robot_cloud, R, t)``).
    - ``link_transforms`` : (L, 4, 4) homogeneous world placements of the robot links @ q."""
    q = np.asarray(traj.qpos[f], np.float64)                       # (nq,)
    poses = np.asarray(traj.object_poses[f], np.float64)           # (N, 7) pos + quat wxyz
    info = traj.info[f]                                            # FrameInfo

    # « atteint » — REUSE the Evaluator (no recompute of retargeting).
    style_achieved = ev.style(q)                                  # StyleEval
    object_pos = poses[:, :3]                                     # (N, 3)
    object_rot = _quat_wxyz_to_R(poses[:, 3:7])                   # (N, 3, 3)
    contact_achieved = ev.contacts(q, object_rot, object_pos)     # ContactEval

    # FK : link transforms (L,4,4) + robot control points (M,3).
    rot, pos = ctx.robot.link_transforms(q)                       # (L,3,3), (L,3)
    L = rot.shape[0]
    link_transforms = np.zeros((L, 4, 4), np.float64)
    link_transforms[:, :3, :3] = rot
    link_transforms[:, :3, 3] = pos
    link_transforms[:, 3, 3] = 1.0
    robot_points_world = pose_cloud(ctx.robot_cloud, rot, pos)    # (M, 3)

    return SolvedFrame(
        q=q,
        object_poses=poses,
        robot_points_world=np.asarray(robot_points_world, np.float64),
        link_transforms=link_transforms,
        style_achieved=style_achieved,
        contact_achieved=contact_achieved,
        cost=float(info.cost),
        cost_by_term=dict(info.cost_by_term),
        n_iters=int(info.n_iters),
        status=str(info.status),
    )
```

- [ ] **Step 4 : Câbler `BakeSource(solve=True)`** (glu Phase A — chemin solve optionnel)

Dans `BakeSource.__init__` (Phase A), AJOUTER le paramètre solve-config et le chemin solve. Les imports lazy (`solve.runner`, `SolveConfig`) restent locaux pour garder `import src.viz.sources` léger côté `cvxpy` :

```python
# signature (Phase A canonical) : def __init__(self, spec, config, *, solve=False, solve_config=None, ...):
# ... après prepare(...) -> (self._grounded, self._ctx) et le bake des FrameTrace (Phase A) ...
self._solved: list = [None] * self._n_frames
if solve:
    from ..solve.runner import solve as _solve_seq          # lazy : cvxpy n'arrive qu'ici
    from ..solve.config import SolveConfig
    from ..targets import Evaluator
    # Réutilise les FrameTargets DÉJÀ calculés par le bake trace (pas de recompute) :
    frame_targets = [self._traces[f].targets for f in range(self._n_frames)]
    traj = _solve_seq(self._grounded, self._ctx, frame_targets,
                      solve_config or SolveConfig(), robot_name=spec.robot.name)
    ev = Evaluator(self._ctx, spec.robot.name)
    self._solved = [build_solved_frame(traj, ev, self._ctx, f)
                    for f in range(self._n_frames)]
# VizContext (Phase A) : has_solve=solve, robot_urdf_path=spec.robot.urdf_path
```

Dans `BakeSource.get(i)` (Phase A), passer `solved=self._solved[i]` au constructeur `VizFrame`.

> Adapter aux noms exacts de Phase A (`self._traces`, `self._n_frames`, `self._grounded`, `self._ctx`) au merge — la LOGIQUE (réutiliser `trace.targets`, solve 1×, Evaluator 1×, `build_solved_frame` par frame) est stable.

- [ ] **Step 5 : Lancer, vérifier le succès + l'invariant numpy-only**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_solved_frame.py -q`
Expected: PASS (3 tests, ou SKIPPED si données absentes localement — l'import `build_solved_frame` doit réussir quoi qu'il arrive).

Run (sources.py reste numpy-only à l'import — pas de viser) :
`~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz.sources; assert 'viser' not in sys.modules, 'viser leaked into sources!'; print('sources numpy-only ok')"`
Expected: `sources numpy-only ok`

- [ ] **Step 6 : Commit**

```bash
git add src/viz/sources.py tests/test_viz_solved_frame.py
git commit -m "feat(holov2): viz/sources — build_solved_frame + BakeSource(solve=True) (seam solve -> SolvedFrame)"
```

---

### Task 2 : `RobotLayer` — la couche 3D du G1 résolu (ViserUrdf)

**Files:**
- Create: `src/viz/layers/robot.py`
- (manuel) `app.py` non requis ici ; la couche est câblée en Task 4.

**Interfaces:**
- Consumes (Phase A) : `Layer` protocol, `UiState`, `VizContext` (`robot_urdf_path`, `has_solve`), `VizFrame` (`frame.solved`).
- Produces : `RobotLayer` (couche 3D ; no-op/masquée quand `frame.solved is None`).

**Note load-bearing (piège #1) — convention quaternion :** `solved.q` est un vecteur free-flyer **pinocchio** : `q[:3]` = position base, **`q[3:7]` = quat xyzw**, `q[7:7+dof]` = joints (cf. `prepare/load/robot.py` : « q = [pelvis(7: pos + quat **xyzw**), joints] » et `solve/init.py` : `q[3:7] = quat_wxyz_to_xyzw(...)`). Viser attend un quaternion **wxyz** sur le frame de base ⇒ réordonner `q[[6, 3, 4, 5]]` (qw, qx, qy, qz). C'est le seul écart vs V1 (`viewer.draw_q`, dont le `q` Holosoma portait déjà wxyz en `q[3:7]`). `update_cfg(q[7:7+dof])` et la pose de base sont sinon le portage direct de `viewer._add_robot`/`draw_q`.

- [ ] **Step 1 : Implémenter `src/viz/layers/robot.py`** (couche mince ; pas de test unitaire — `update` = pure affectation de handles, cf. design « couches assez fines pour ne pas avoir à les tester »)

```python
"""Layer ``robot`` — the SOLVED G1 rendered with ViserUrdf (full meshes), roadmap #1. Reads
``frame.solved.q`` (pinocchio free-flyer config). No-op/hidden when ``frame.solved is None``
(pre-solve / solve disabled). Ported LOGIC from V1 ``viewer._add_robot`` / ``draw_q``; viser/ViserUrdf
are confined to this layer (consumer, golden rule 6)."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame

_FOLDER = "Robot (solved)"
_ROOT = "/world/robot_solved"


class RobotLayer:
    """``Layer`` for the solved robot. ``setup`` loads the URDF once (ViserUrdf) + a toggle; ``update``
    drives ``update_cfg`` (joints) and the base frame pose (pinocchio xyzw -> viser wxyz)."""

    folder = _FOLDER

    def __init__(self) -> None:
        self._urdf = None      # ViserUrdf handle
        self._base = None      # base frame handle
        self._dof = 0
        self._toggle = None

    def setup(self, server, gui, ctx: VizContext) -> None:
        import yourdfpy
        from viser.extras import ViserUrdf

        self._base = server.scene.add_frame(_ROOT, show_axes=False)
        urdf = yourdfpy.URDF.load(str(ctx.robot_urdf_path), load_meshes=True, build_scene_graph=True)
        self._urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name=_ROOT)
        self._dof = len(self._urdf.get_actuated_joint_limits())
        self._urdf.update_cfg(np.zeros(self._dof))
        with gui.add_folder(_FOLDER):
            self._toggle = gui.add_checkbox("Show solved G1", bool(ctx.has_solve))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        solved = frame.solved
        show = bool(self._toggle.value) and solved is not None
        self._urdf.show_visual = show
        self._base.visible = show
        if not show:
            return
        q = np.asarray(solved.q, np.float64)
        self._urdf.update_cfg(q[7:7 + self._dof])
        self._base.position = q[:3]
        self._base.wxyz = q[[6, 3, 4, 5]]            # pinocchio xyzw (q[3:7]) -> viser wxyz
```

- [ ] **Step 2 : Vérifier l'import (sanity, sans écran)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.viz.layers.robot as m; assert hasattr(m, 'RobotLayer') and m.RobotLayer.folder; print('RobotLayer import ok')"`
Expected: `RobotLayer import ok`

- [ ] **Step 3 : Vérification visuelle manuelle** (différée à Task 4, où `app.py --solve` câble la couche)

Critère manuel (à Task 4) : le robot G1 apparaît superposé à la scène ; le toggle « Show solved G1 » le masque/affiche ; sur une frame `solved is None` (source sans solve) la couche reste masquée ; la base bouge cohéremment avec le pelvis (orientation correcte ⇒ la réordonnance xyzw→wxyz est bonne).

- [ ] **Step 4 : Commit**

```bash
git add src/viz/layers/robot.py
git commit -m "feat(holov2): viz/layers/robot — couche RobotLayer (G1 résolu via ViserUrdf), masquée sans solve"
```

---

### Task 3 : `CostDashboard` — panel 2D des diagnostics solve

**Files:**
- Create: `src/viz/panels/__init__.py` (si absent)
- Create: `src/viz/panels/cost_dashboard.py`
- Test: `tests/test_viz_cost_dashboard.py`

**Interfaces:**
- Consumes (Phase A) : `VizFrame` (`frame.solved.cost_by_term`/`cost`/`status`/`n_iters`).
- Produces : `stack_cost_terms(frames) -> (names, matrix, total, status, n_iters)` (pur, testable) ; `CostDashboard` (panel 2D, matplotlib `Agg` → `gui.add_image` + `gui.add_markdown`).

**Note dépendances :** **plotly est ABSENT** de l'env (`gui.add_plotly` indisponible) ⇒ on rend un graphe **matplotlib (backend `Agg`)** en image RGB `(H,W,3)` uint8 passée à `gui.add_image`, + un `gui.add_markdown` listant les frames non convergées (status/n_iters). matplotlib/viser confinés au panel.

- [ ] **Step 1 : Écrire le test pur** (`tests/test_viz_cost_dashboard.py`) — verrouille l'empilement des coûts, sans matplotlib ni viser

```python
"""stack_cost_terms : empilement pur du cost_by_term sur la séquence (union des termes, frames None
-> ligne nulle). Pas de matplotlib/viser ici (logique pure du panel)."""
import numpy as np

from src.viz.panels.cost_dashboard import stack_cost_terms


class _Solved:                                          # double minimal d'un SolvedFrame
    def __init__(self, cost, by, status, n):
        self.cost, self.cost_by_term, self.status, self.n_iters = cost, by, status, n


class _Frame:
    def __init__(self, solved):
        self.solved = solved


def test_stack_cost_terms_union_and_none():
    frames = [
        _Frame(_Solved(3.0, {"S-pos": 1.0, "C-D": 2.0}, "optimal", 4)),
        _Frame(None),                                   # frame non résolue -> ligne nulle
        _Frame(_Solved(5.0, {"C-D": 5.0}, "max_iter", 10)),
    ]
    names, mat, total, status, n_iters = stack_cost_terms(frames)
    assert set(names) == {"S-pos", "C-D"}               # union des termes
    assert mat.shape == (3, len(names))
    j_cd = names.index("C-D")
    assert mat[0, j_cd] == 2.0 and mat[1, j_cd] == 0.0 and mat[2, j_cd] == 5.0
    assert np.array_equal(total, np.array([3.0, 0.0, 5.0]))
    assert status[0] == "optimal" and status[1] == "" and status[2] == "max_iter"
    assert n_iters[2] == 10 and n_iters[1] == 0


def test_stack_cost_terms_all_none():
    names, mat, total, status, n_iters = stack_cost_terms([_Frame(None), _Frame(None)])
    assert names == () and mat.shape == (2, 0)
    assert np.array_equal(total, np.zeros(2)) and status == ("", "")
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_cost_dashboard.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.panels.cost_dashboard`).

- [ ] **Step 3 : Créer `src/viz/panels/__init__.py`** (si absent)

```python
"""2D panels (non-3D diagnostics over the whole sequence) — distinct from the 3D ``layers/``.
Each panel reads ``solved.*`` across all baked frames (provided by the Source) and renders into the
viser GUI. viser/matplotlib are confined here (consumer, golden rule 6)."""
```

- [ ] **Step 4 : Implémenter `src/viz/panels/cost_dashboard.py`**

```python
"""Panel ``cost_dashboard`` (roadmap #2) — total cost + stacked ``cost_by_term`` over ALL solved
frames, with status/n_iters markers for the non-converged ones. Rendered with matplotlib (Agg) into a
viser image (plotly is absent from the env, so ``gui.add_plotly`` is NOT used). Reads ``frame.solved``
across the whole sequence (provided by the Source at setup). matplotlib/viser confined to this module."""
from __future__ import annotations

import numpy as np

_OK = ("optimal", "optimal_inaccurate")
_FOLDER = "Cost dashboard"


def stack_cost_terms(frames):
    """Stack the per-term costs over the sequence. PURE (testable, no matplotlib/viser).

    Returns ``(names, matrix (F, K), total (F,), status (F,), n_iters (F,))``: ``names`` = union of
    ``cost_by_term`` keys in first-seen order; a frame with ``solved is None`` -> all-zero row +
    empty status; a term absent from a frame -> 0."""
    solved = [getattr(fr, "solved", None) for fr in frames]
    names: list[str] = []
    for s in solved:
        if s is not None:
            for k in s.cost_by_term:
                if k not in names:
                    names.append(k)
    F, K = len(frames), len(names)
    mat = np.zeros((F, K), np.float64)
    total = np.zeros(F, np.float64)
    status = [""] * F
    n_iters = np.zeros(F, np.int64)
    for i, s in enumerate(solved):
        if s is None:
            continue
        total[i] = float(s.cost) if np.isfinite(s.cost) else 0.0
        status[i] = str(s.status)
        n_iters[i] = int(s.n_iters)
        for j, k in enumerate(names):
            mat[i, j] = float(s.cost_by_term.get(k, 0.0))
    return tuple(names), mat, total, tuple(status), n_iters


def _render_chart(names, mat, total, status) -> np.ndarray:
    """matplotlib (Agg) stacked-area of cost_by_term + total line + non-converged markers -> RGB uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = mat.shape[0]
    x = np.arange(F)
    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=100)
    if names and F > 0:
        ax.stackplot(x, mat.T, labels=list(names))
    ax.plot(x, total, color="k", lw=1.5, label="total")
    bad = [i for i, st in enumerate(status) if st not in _OK]
    if bad:
        ax.scatter(bad, total[np.asarray(bad)], color="r", zorder=5, s=20, label="non-converged")
    ax.set_xlabel("frame")
    ax.set_ylabel("cost (Σ residuals²)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())          # (H, W, 4) uint8
    plt.close(fig)
    return rgba[..., :3].copy()                          # (H, W, 3) RGB


def _summary_md(status, n_iters) -> str:
    bad = [(i, st, int(n)) for i, (st, n) in enumerate(zip(status, n_iters)) if st not in _OK and st]
    if not bad:
        return "**solve** — all frames converged (optimal)."
    lines = ["**solve — non-converged frames**", "", "| frame | status | n_iters |", "|---|---|---|"]
    lines += [f"| {i} | {st} | {n} |" for i, st, n in bad]
    return "\n".join(lines)


class CostDashboard:
    """2D panel. ``setup`` once with the whole sequence -> a cost image + a non-converged summary."""

    folder = _FOLDER

    def setup(self, server, gui, frames) -> None:
        names, mat, total, status, n_iters = stack_cost_terms(frames)
        img = _render_chart(names, mat, total, status)
        with gui.add_folder(self.folder):
            gui.add_image(img, label="cost / cost_by_term")
            gui.add_markdown(_summary_md(status, n_iters))
```

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_cost_dashboard.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6 : Vérification visuelle manuelle** (différée à Task 4)

Critère manuel (à Task 4) : avec `--solve`, le folder « Cost dashboard » montre une image (aires empilées par terme + courbe `total` noire) et, si des frames n'ont pas convergé, un tableau markdown frame/status/n_iters ; sinon le message « all frames converged ».

- [ ] **Step 7 : Commit**

```bash
git add src/viz/panels/__init__.py src/viz/panels/cost_dashboard.py tests/test_viz_cost_dashboard.py
git commit -m "feat(holov2): viz/panels/cost_dashboard — coût empilé par terme + non-convergées (matplotlib->image viser)"
```

---

### Task 4 : `app.py --solve` — câbler la seam + couche + panel

**Files:**
- Modify: `src/viz/app.py`

**Interfaces:**
- Consumes : `BakeSource(solve=True)` (Task 1), `RobotLayer` (Task 2), `CostDashboard` (Task 3), `run_app(..., solve=False)` (Phase A).
- Produces : un flag CLI `--solve` qui (a) construit `BakeSource(..., solve=True)`, (b) ajoute `RobotLayer` à la liste des couches, (c) instancie `CostDashboard` et l'alimente avec toutes les frames bakées.

- [ ] **Step 1 : Câbler le flag `--solve`** (additif, sur la structure Phase A de `app.py`)

```python
# 1) CLI : ajouter un flag (tyro/argparse selon Phase A), défaut False :
#       solve: bool = False        # run solve/ and overlay the solved robot + cost dashboard
#
# 2) Source : propager le flag :
#       source = BakeSource(spec, config, solve=solve)
#
# 3) Couches : RobotLayer est SÛRE même sans solve (elle se masque si frame.solved is None) ->
#    on l'ajoute toujours à la liste des couches prod :
#       layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
#                 ObjectsLayer(), FieldsLayer(), StyleLayer(), RobotLayer()]   # <- + RobotLayer
#
# 4) Panel : seulement utile avec solve (lit solved.* sur toute la séquence) :
#       panels = []
#       if solve:
#           panels.append(CostDashboard())
#
# 5) Setup des panels : le Player/app appelle panel.setup(server, gui, source.frames) une fois
#    (source.frames = toutes les VizFrame bakées, cf. hypothèse Phase A #2). Si Phase A n'expose pas
#    encore source.frames, l'ajouter (les frames sont déjà en mémoire dans BakeSource).
```

Imports à ajouter en tête de `app.py` :

```python
from .layers.robot import RobotLayer
from .panels.cost_dashboard import CostDashboard
```

- [ ] **Step 2 : Vérifier l'import / le parsing CLI (sans écran)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.viz.app as a; print('app import ok')"`
Expected: `app import ok`

Run (le flag est reconnu) : `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.app --help 2>&1 | grep -i solve`
Expected : une ligne décrivant `--solve`.

- [ ] **Step 3 : Vérification visuelle manuelle** (le critère du redesign : « robot résolu superposé »)

Commande (adapter les chemins de données locaux, cf. CLAUDE.md / `path.yaml`) :
```bash
cd HoloV2
~/.holonew_deps/miniconda3/envs/holonew/bin/python -m src.viz.app --solve \
    --motion-path <…/hodome/smplx/<seq>.npz> --model-dir <…/models_smplx>  # + flags Phase A
```
Critères (ouvrir l'URL viser) :
1. La scène pré-solve (humain/cibles) s'affiche comme avant (Phase A non régressée).
2. Le **G1 résolu** apparaît superposé, dans l'espace solve ; le toggle « Show solved G1 » fonctionne.
3. La base du robot suit le pelvis avec la BONNE orientation (valide la réordonnance xyzw→wxyz du Task 2).
4. Le folder « Cost dashboard » montre l'image des coûts empilés + (si applicable) le tableau des frames non convergées.
5. Sans `--solve` : aucun robot, aucun panel coût ; le pré-solve reste intact (solve optionnel de bout en bout).

- [ ] **Step 4 : Commit**

```bash
git add src/viz/app.py
git commit -m "feat(holov2): viz/app — flag --solve (BakeSource solve + RobotLayer + CostDashboard)"
```

---

### Task 5 : `docs/VIZ.md` — documenter la seam solve (section DISTINCTE)

**Files:**
- Modify: `docs/VIZ.md`

**Interfaces:** documentation seulement (aucun code).

> **Important :** Phase A RÉÉCRIT `VIZ.md` (nouvelle archi couches). Pour éviter de clobberer cette réécriture, AJOUTER une section **distincte en fin de fichier** (titre dédié), sans toucher au reste. Au merge, si Phase A a déjà une liste de couches, n'y intégrer que les 2 lignes robot/cost (cf. tableau roadmap du design) — la section ci-dessous reste l'ancrage détaillé.

- [ ] **Step 1 : APPEND la section à `docs/VIZ.md`**

```markdown
## Seam solve : couche `robot` + panel `cost_dashboard`

Le viz consomme aussi `solve/` (optionnel). `BakeSource(spec, config, solve=True)` exécute
`solve.runner.solve` UNE fois sur la séquence montrée, construit un `targets.Evaluator` UNE fois, et
remplit `VizFrame.solved` par frame via `build_solved_frame(traj, ev, ctx, f)` (pur, numpy-only) :

- `q` / `object_poses` / `cost`/`cost_by_term`/`n_iters`/`status` : lus directement dans
  `SolveTrajectory[f]` (+ `FrameInfo[f]`).
- `style_achieved = ev.style(q)`, `contact_achieved = ev.contacts(q, object_rot, object_pos)` :
  l'« atteint » RÉUTILISE l'`Evaluator` (le recompute prouvé dans `test_solve_runner`), aucune
  nouvelle logique de retargeting. `object_poses` (wxyz) -> matrices via scipy.
- `robot_points_world = pose_cloud(robot_cloud, R, t)` et `link_transforms (L,4,4)` : FK @ q via
  `ctx.robot`.

Sans `solve` (`solve=False`), `VizFrame.solved is None` et tout le pré-solve marche ; les
consommateurs solve se masquent. Couches/panels concernés :

- **`layers/robot.py` (`RobotLayer`, roadmap #1)** : le G1 résolu en meshes complets via
  `viser.extras.ViserUrdf` (yourdfpy). `update_cfg(q[7:7+dof])` + pose de base `q[:3]` /
  `q[[6,3,4,5]]` (le `q` pinocchio porte le quat en **xyzw** sur `q[3:7]` ⇒ réordonné en wxyz pour
  viser). No-op/masquée quand `frame.solved is None`. URDF depuis `VizContext.robot_urdf_path`.
- **`panels/cost_dashboard.py` (`CostDashboard`, roadmap #2)** : panel 2D (matplotlib `Agg` -> image
  viser, car plotly absent de l'env) — `cost_by_term` empilé + `cost` total sur toutes les frames,
  + marqueurs/tableau des frames non convergées (`status`/`n_iters`). Lit `solved.*` sur toute la
  séquence (fourni par la Source au `setup`).

`app.py --solve` câble `BakeSource(solve=True)` + `RobotLayer` + `CostDashboard`. Invariant tenu :
`viz` n'importe que les surfaces PUBLIQUES (`solve.runner`/`solve.contracts`, `targets.Evaluator`) ;
`targets`/`solve` restent inchangés.
```

- [ ] **Step 2 : Commit**

```bash
git add docs/VIZ.md
git commit -m "docs(holov2): VIZ.md — section seam solve (RobotLayer + CostDashboard)"
```

---

## Self-Review

**1. Couverture de la spec (Phase B = étape de migration #4 + roadmap #1/#2) :**
- `BakeSource(solve=True)` exécute `solve` 1× + `Evaluator` 1× + remplit `SolvedFrame` par frame (q, object_poses, cost/cost_by_term/n_iters/status + `ev.style`/`ev.contacts` + FK) → **Task 1**. ✅
- `VizFrame.solved` non-None avec `--solve`, `None` sinon (solve optionnel de bout en bout) → Task 1 (glu) + Task 4 (flag). ✅
- Couche `robot` (ViserUrdf, `update_cfg` + pose base ; masquée sans solve) → **Task 2**. ✅
- Panel 2D `cost_dashboard` (cost_by_term empilé + cost + status/n_iters non convergées) → **Task 3**. ✅
- `app.py --solve` câble source+couche+panel → **Task 4**. ✅
- VIZ.md : section distincte robot/cost (n'écrase pas la réécriture Phase A) → **Task 5**. ✅
- Focus testable = `build_solved_frame` : déterminisme (build ×2 identique), formes/dtypes, « atteint » == appel `Evaluator` direct (pattern `test_solve_runner`), scène démo `max_frames` bas → Task 1 (2 tests data-gated) + smoke `BakeSource(solve=True)`. Panel : test pur `stack_cost_terms` ; couche/panel/wiring = manual viser check explicité (Task 2/3/4). ✅

**2. Placeholder scan :** aucun `TBD`/`TODO`/`...` non résolu ; tout step porte du code réel ou une commande réelle. Les seuls « adapter aux noms Phase A » sont des points d'intégration EXPLICITES (glu `BakeSource` / `app.py`), isolés de la logique testée (`build_solved_frame`, `stack_cost_terms`, `RobotLayer.update`). ✅

**3. Cohérence des noms de types** (vs API canonique Phase A + types déjà mergés) :
- `SolvedFrame(q, object_poses, robot_points_world, link_transforms, style_achieved, contact_achieved, cost, cost_by_term, n_iters, status)` — exactement les 10 champs du design. ✅
- `VizContext.robot_urdf_path` / `.has_solve` ; `VizFrame.solved` ; `Layer.folder/setup(server,gui,ctx)/update(frame,ui)` ; `UiState` ; `BakeSource(spec, config, *, solve=...)`. ✅
- `solve.runner.solve(grounded, ctx, frame_targets, SolveConfig, robot_name=...)`, `SolveTrajectory.qpos`/`.object_poses`/`.info`, `FrameInfo.cost/cost_by_term/n_iters/status` — conformes au code mergé (`solve/runner.py`, `solve/contracts.py`). ✅
- `Evaluator.style(q)` / `Evaluator.contacts(q, object_rot, object_pos)` — signature réelle (PAS `contacts(q, poses)` : conversion wxyz→(rot,pos) explicitée). `pose_cloud(cloud, part_rot, part_pos)` — réel. ✅

**4. Invariants viz tenus :** `sources.py` numpy-only (assert d'import sans viser, Task 1 Step 5) ; viser/ViserUrdf confinés à `layers/robot.py` ; matplotlib/viser confinés à `panels/cost_dashboard.py` ; `targets`/`solve` non modifiés ; `build_solved_frame` ne fait que lire/réutiliser. plotly évité (absent de l'env) au profit de matplotlib `Agg`. ✅
