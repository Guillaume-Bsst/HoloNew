# Plan C : solve — init + retract + assemble + loop + runner (end-to-end SQP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Câbler l'orchestration end-to-end de l'étage `solve/` : la boucle SQP/trust-region par frame (`evaluate → assemble → backend.solve → retract → converge`) plus le seed `init` (Holosoma : base = cible pelvis + joints neutres), la rétraction numpy (robot `integrate` + exp SE(3) objet), l'assemblage `Problem` (appelle les builders Plan B) et l'entrée publique `solve(...) -> SolveTrajectory`.

**Architecture:** Trois incréments composent l'étage `solve/` : **A** (cœur : `Problem`/`Step`/`TrustRegion`/`SolveBackend`/`CvxpyBackend` + `make_backend`), **B** (`config.SolveConfig` + `terms/` builders + `terms/constraints`), **C = CE PLAN** (orchestration). C consomme A (contrats + backend) et B (config + builders) verbatim, plus la surface publique amont `targets` (`Evaluator`, types Eval/Ref, `process_frame`) et `prepare.contracts` (`RobotModel`). Flux **linéaire** par frame, classe-dieu interdite : `loop.solve_frame` itère `evaluate → assemble → solve → retract`, `runner.solve` boucle les frames (f=0 `compute_q_init`, f>0 `warm_start`) et collecte `SolveTrajectory`. `solve` reste **pinocchio/torch-free** (cinématique via le Protocol `RobotModel`) ; l'exp SE(3) objet est du **numpy pur** ; cvxpy reste confiné au backend (Plan A).

**Tech Stack:** Python, numpy (float64), cvxpy 1.9.1 (via le backend Plan A), pytest. Env `holonew`.

## Global Constraints

- **Plan A est DONE** : `solve/contracts.py` (`Problem`, `ResidualBlock`, `LinearConstraint`, `TrustRegion`, `Step`), `solve/backend/` (`SolveBackend` Protocol, `CvxpyBackend`, `make_backend(name) -> SolveBackend`), `solve/__init__.py` exporte déjà `Problem`/`Step`/`make_backend`.
- **Plan B est DONE** : `solve/config.py` (`SolveConfig`) et les builders, signatures **CANONIQUES utilisées verbatim** :
  - `build_style(style_eval, style_targets, cfg) -> list[ResidualBlock]`
  - `build_contact(contact_eval, robot_field_ref, geo, cfg) -> list[ResidualBlock]`
  - `build_object(contact_eval, env_refs, object_rot, object_pos, cfg) -> list[ResidualBlock]`
  - `build_reg(nv, cfg) -> list[ResidualBlock]`
  - `build_constraints(robot, cfg) -> (list[LinearConstraint], list[TrustRegion])`
- **`solve` reste pinocchio/torch-free** : cinématique via le Protocol `RobotModel` (`neutral`/`integrate`/`nq`/`nv`/`link_names`) ; l'exp SE(3) objet est **plain numpy** (quaternion + Rodrigues) ; cvxpy uniquement via le backend Plan A.
- Compute en **float64**. Quaternions **wxyz** ; poses objet `(x, y, z, qw, qx, qy, qz)`. La config `q` robot = pinocchio free-flyer `[pos(3), quat **xyzw**(4), joints]` de dim `nq` (conversion wxyz→xyzw au seed).
- Imports **relatifs** dans `src/` (l'aval importe la surface PUBLIQUE de l'amont : `from ..targets import Evaluator`, `from ..prepare.contracts import ...`, jamais un interne) ; **absolus** (`from src.…`) dans `tests/`.
- Invariants de contrat → `raise ValueError` explicite au `__post_init__` (style `MultiChannelField`).
- Tests dans `HoloV2/tests/`, lancés depuis `HoloV2/` avec
  `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<f> -q`. `max_frames` bas.
- Data-gated tests : `from datapaths import HODOME, SMPLX_MODELS` + `pytest.mark.skipif` quand la donnée/URDF/corr absente (cf. `tests/test_evaluator.py`).
- Commits **conventionnels, français**. **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`/mention). Auteur : `Guillaume-Bsst`.

### Seam upstream consommé (rappel des types réels)

- `targets.Evaluator(ctx, robot_name)` → `.style(q) -> StyleEval`, `.contacts(q, object_rot, object_pos) -> ContactEval`. **Pas** de `evaluate(q, poses)` combiné : Plan C ajoute un **wrapper `evaluate`** (fonction libre dans `solve/loop.py`) qui convertit `object_poses (N,7) → (object_rot (N,3,3), object_pos (N,3))` et appelle les deux méthodes.
- `StyleEval(position (L,3), rotation (L,3,3), jac_pos (L,3,nv), jac_rot (L,3,nv), link_names)`.
- `ContactEval(field, point_jac (M,3,nv), probe_jac_obj (C,M,3,6), env)`.
- `FrameTargets(style: StyleTargets, robot_interaction: RobotInteractionTargets, env_interaction: EnvironmentInteractionTargets, object_rot (N,3,3), object_pos (N,3))`.
  - `StyleTargets(link_names (L,), position (L,3), orientation (L,4) wxyz | None)`. Le **pelvis** est `link_names[0]` pour G1 (recette `style_table("g1")`, `prepare`/`targets`).
  - `RobotInteractionTargets(field)` ; `EnvironmentInteractionTargets(per_object)`.
- `prepare.contracts.RobotModel` : `neutral() -> (nq,)` (base identité quat xyzw + joints 0), `integrate(q, v) -> (nq,)`, `link_names`, `nq`, `nv`.
- `prepare.contracts.InteractionContext` : `.robot` (RobotModel), `.channels` (tuple `Channel`, chacun porte `.sdf` + `.geodesic`), `.object_clouds`.

---

### Task 1 : `solve/contracts.py` — `FrameEval`, `FrameInfo`, `SolveTrajectory`

**Files:**
- Modify: `src/solve/contracts.py` (APPEND après les types Plan A)
- Test: `tests/test_solve_traj_contracts.py`

**Interfaces:**
- Consumes (Plan A) : rien (autonome). Annotations `StyleEval`/`ContactEval` en `TYPE_CHECKING` seulement (pas d'import runtime → `contracts` reste numpy-only autonome).
- Produces :
  - `FrameEval(style, contact)` — conteneur des deux Eval `targets` (sortie du wrapper `evaluate`, entrée de `assemble`).
  - `FrameInfo(n_iters: int, status: str, cost: float, cost_by_term: dict[str, float])`.
  - `SolveTrajectory(qpos (T,nq), object_poses (T,N,7), info: tuple[FrameInfo, ...])` — validé.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_traj_contracts.py
"""Contrats d'orchestration solve : construction de FrameEval / FrameInfo / SolveTrajectory + rejet
des formes incohérentes au __post_init__ de SolveTrajectory."""
import types

import numpy as np
import pytest

from src.solve.contracts import FrameEval, FrameInfo, SolveTrajectory


def test_frame_eval_is_a_plain_container():
    fe = FrameEval(style=types.SimpleNamespace(position=np.zeros((1, 3))),
                   contact=types.SimpleNamespace(point_jac=np.zeros((1, 3, 6))))
    assert fe.style.position.shape == (1, 3)
    assert fe.contact.point_jac.shape == (1, 3, 6)


def test_frame_info_fields():
    fi = FrameInfo(n_iters=3, status="optimal", cost=1.5, cost_by_term={"S-pos": 1.0, "C-D": 0.5})
    assert fi.n_iters == 3 and fi.status == "optimal" and fi.cost == 1.5
    assert fi.cost_by_term["S-pos"] == 1.0


def test_solve_trajectory_valid_with_objects():
    T, nq, N = 2, 8, 1
    info = (FrameInfo(1, "optimal", 0.0, {}), FrameInfo(2, "optimal", 0.0, {}))
    traj = SolveTrajectory(qpos=np.zeros((T, nq)), object_poses=np.zeros((T, N, 7)), info=info)
    assert traj.qpos.shape == (T, nq) and traj.object_poses.shape == (T, N, 7)
    assert traj.n_frames == T


def test_solve_trajectory_valid_no_objects():
    T, nq = 3, 5
    info = tuple(FrameInfo(1, "optimal", 0.0, {}) for _ in range(T))
    traj = SolveTrajectory(qpos=np.zeros((T, nq)), object_poses=np.zeros((T, 0, 7)), info=info)
    assert traj.object_poses.shape == (T, 0, 7)


def test_solve_trajectory_bad_object_pose_width_raises():
    with pytest.raises(ValueError):                       # last dim must be 7
        SolveTrajectory(qpos=np.zeros((2, 5)), object_poses=np.zeros((2, 1, 6)),
                        info=(FrameInfo(1, "optimal", 0.0, {}), FrameInfo(1, "optimal", 0.0, {})))


def test_solve_trajectory_info_length_mismatch_raises():
    with pytest.raises(ValueError):                       # len(info) must equal T
        SolveTrajectory(qpos=np.zeros((3, 5)), object_poses=np.zeros((3, 0, 7)),
                        info=(FrameInfo(1, "optimal", 0.0, {}),))


def test_solve_trajectory_object_frames_mismatch_raises():
    with pytest.raises(ValueError):                       # object_poses T axis must equal qpos T axis
        SolveTrajectory(qpos=np.zeros((2, 5)), object_poses=np.zeros((3, 0, 7)),
                        info=(FrameInfo(1, "optimal", 0.0, {}), FrameInfo(1, "optimal", 0.0, {})))
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_traj_contracts.py -q`
Expected: FAIL (`ImportError: cannot import name 'FrameEval' from src.solve.contracts`).

- [ ] **Step 3 : APPEND à `src/solve/contracts.py`** (après le `Step` de Plan A ; le module garde `from __future__ import annotations` en tête)

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # annotations only -> contracts stays numpy-only at runtime
    from ..targets import StyleEval, ContactEval


@dataclass(frozen=True)
class FrameEval:
    """Combined per-frame evaluator output: the style FK + the contact field/Jacobians at the current
    ``(q, object_poses)``. Produced by the ``evaluate`` wrapper (``solve/loop.py``), consumed by
    ``assemble``. A plain container (no shape logic) — the two members validate themselves."""

    style: "StyleEval"
    contact: "ContactEval"


@dataclass(frozen=True)
class FrameInfo:
    """Per-frame solve diagnostic (weight tuning + benchmark). ``cost_by_term`` is the squared residual
    norm per term (``S-pos`` / ``C-D`` / …) at the converged step — the #1 tuning tool."""

    n_iters: int
    status: str
    cost: float
    cost_by_term: dict[str, float]


@dataclass(frozen=True)
class SolveTrajectory:
    """Runner output: the retargeted ``qpos`` trajectory + the per-frame object poses + diagnostics.
    ``object_poses`` is ``(T, N, 7)`` (pos + quat wxyz) ; ``N = 0`` keeps the ``(T, 0, 7)`` shape."""

    qpos: np.ndarray          # (T, nq)
    object_poses: np.ndarray  # (T, N, 7)  pos + quat wxyz
    info: tuple[FrameInfo, ...]

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2:
            raise ValueError(f"qpos must be 2-D (T, nq), got shape {self.qpos.shape}")
        T = self.qpos.shape[0]
        if self.object_poses.ndim != 3 or self.object_poses.shape[2] != 7:
            raise ValueError(
                f"object_poses must be (T, N, 7), got shape {self.object_poses.shape}")
        if self.object_poses.shape[0] != T:
            raise ValueError(
                f"object_poses has {self.object_poses.shape[0]} frames but qpos has {T}")
        if len(self.info) != T:
            raise ValueError(f"info has {len(self.info)} entries but qpos has {T} frames")

    @property
    def n_frames(self) -> int:
        return self.qpos.shape[0]
```

- [ ] **Step 4 : Lancer, vérifier le succès + import numpy-only**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_traj_contracts.py -q`
Expected: PASS (7 tests).

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.solve.contracts; assert 'cvxpy' not in sys.modules and 'torch' not in sys.modules; print('contracts numpy-only ok')"`
Expected: `contracts numpy-only ok`

- [ ] **Step 5 : Commit**

```bash
git add src/solve/contracts.py tests/test_solve_traj_contracts.py
git commit -m "feat(holov2): solve/contracts — FrameEval/FrameInfo/SolveTrajectory (orchestration, validés)"
```

---

### Task 2 : `solve/retract.py` — math SE(3)/quaternion numpy + rétraction

**Files:**
- Create: `src/solve/retract.py`
- Test: `tests/test_solve_retract.py`

**Interfaces:**
- Consumes (Plan A) : `Step` (de `..contracts`). `RobotModel.integrate`.
- Produces (réutilisés par `init`/`loop`) :
  - `so3_exp(w (3,)) -> (3,3)` (Rodrigues, garde petit-angle).
  - `quat_wxyz_to_mat(q (4,)) -> (3,3)` ; `mat_to_quat_wxyz(R (3,3)) -> (4,)` ; `quat_wxyz_to_xyzw(q (4,)) -> (4,)`.
  - `retract(q (nq,), object_poses (N,7), step: Step, robot: RobotModel) -> (q (nq,), object_poses (N,7))`.
  - Convention objet (world-aligned, cohérente avec `ContactEval.probe_jac_obj` `(δt, δθ)` LOCAL_WORLD_ALIGNED) : `new_pos = pos + δt` ; `new_R = exp(δθ) · R` (rotation appliquée à GAUCHE, frame monde).

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_retract.py
"""retract : délégation robot.integrate (round-trip) + exp SE(3) objet (rotation/translation connues),
et les helpers quaternion/Rodrigues (vs valeurs analytiques)."""
import numpy as np

from src.solve.contracts import Step
from src.solve.retract import (mat_to_quat_wxyz, quat_wxyz_to_mat, quat_wxyz_to_xyzw, retract, so3_exp)


class _StubRobot:
    """RobotModel minimal, Euclidien (integrate additif) — teste la DÉLÉGATION de retract, pas la FK
    free-flyer (couverte par les tests pinocchio de prepare)."""

    def __init__(self, n):
        self.nq = self.nv = n

    def integrate(self, q, v):
        return np.asarray(q, np.float64) + np.asarray(v, np.float64)


def test_so3_exp_known_rotation():
    R = so3_exp(np.array([0.0, 0.0, np.pi / 2]))          # +90° autour de z
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_so3_exp_small_angle_is_near_identity():
    R = so3_exp(np.array([1e-10, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3), atol=1e-8)


def test_quat_mat_round_trip():
    R = so3_exp(np.array([0.3, -0.7, 1.1]))
    assert np.allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-12)


def test_quat_wxyz_to_xyzw_reorders():
    assert np.allclose(quat_wxyz_to_xyzw(np.array([0.1, 0.2, 0.3, 0.4])), [0.2, 0.3, 0.4, 0.1])


def test_retract_robot_round_trip():
    robot = _StubRobot(3)
    q0 = np.array([0.1, 0.2, 0.3])
    poses = np.zeros((0, 7))
    q1, p1 = retract(q0, poses, Step(dv=np.array([1.0, 1.0, 1.0]), dxi=None, value=0.0, status="optimal"), robot)
    assert np.allclose(q1, [1.1, 1.2, 1.3])
    q2, _ = retract(q1, p1, Step(dv=np.array([-1.0, -1.0, -1.0]), dxi=None, value=0.0, status="optimal"), robot)
    assert np.allclose(q2, q0)                            # round-trip exact (stub Euclidien)
    assert p1.shape == (0, 7)


def test_retract_object_exp_known_motion():
    robot = _StubRobot(2)
    poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])   # pose identité (quat wxyz)
    dxi = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2]])    # δt + δθ (+90° z)
    _, p = retract(np.zeros(2), poses, Step(dv=np.zeros(2), dxi=dxi, value=0.0, status="optimal"), robot)
    assert np.allclose(p[0, :3], [0.1, 0.2, 0.3], atol=1e-12)
    assert np.allclose(p[0, 3:7], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-10)


def test_retract_object_pure_translation_keeps_orientation():
    robot = _StubRobot(2)
    poses = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    dxi = np.array([[2.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
    _, p = retract(np.zeros(2), poses, Step(dv=np.zeros(2), dxi=dxi, value=0.0, status="optimal"), robot)
    assert np.allclose(p[0], [3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-12)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_retract.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.retract`).

- [ ] **Step 3 : Écrire `src/solve/retract.py`**

```python
"""retract — applique le pas du solveur sur les variables de décision. PUR, numpy-only,
pinocchio/torch-free : ``q ⊕ dv`` est délégué au ``RobotModel.integrate`` (le seul détenteur de la
cinématique free-flyer), ``pose_objet ⊕ dξ`` est une exp SE(3) en numpy pur.

Convention de la tangente objet (cohérente avec ``ContactEval.probe_jac_obj``, world-aligned ``(δt,δθ)``,
LOCAL_WORLD_ALIGNED) : ``new_pos = pos + δt`` (translation monde), ``new_R = exp(δθ) · R`` (incrément de
rotation appliqué à GAUCHE, dans le frame monde). Quaternions wxyz ; pose objet ``[x,y,z,qw,qx,qy,qz]``."""
from __future__ import annotations

import numpy as np

from .contracts import Step


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exponentielle SO(3) (Rodrigues) d'un vecteur de rotation ``w (3,)`` -> ``R (3,3)``. Garde
    petit-angle (série de Taylor) pour rester stable et différentiable près de 0."""
    w = np.asarray(w, np.float64)
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / (th * th)) * (K @ K)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion wxyz (supposé unitaire) -> matrice de rotation ``(3,3)``."""
    qw, qx, qy, qz = (float(v) for v in q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ])


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Matrice de rotation ``(3,3)`` -> quaternion wxyz unitaire (méthode de Shepperd, stable)."""
    R = np.asarray(R, np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Réordonne wxyz -> xyzw (la convention pinocchio du free-flyer ``q``)."""
    q = np.asarray(q, np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


def retract(q: np.ndarray, object_poses: np.ndarray, step: Step, robot) -> tuple[np.ndarray, np.ndarray]:
    """``q ⊕ dv`` via ``robot.integrate`` (free-flyer, pinocchio-free côté solve) + ``pose ⊕ dξ`` via
    exp SE(3) numpy par objet. Ne mute pas les inputs."""
    q_new = robot.integrate(np.asarray(q, np.float64), np.asarray(step.dv, np.float64))
    poses = np.array(object_poses, np.float64, copy=True)
    if step.dxi is not None and poses.shape[0] > 0:
        dxi = np.asarray(step.dxi, np.float64)
        for i in range(poses.shape[0]):
            dt, dth = dxi[i, :3], dxi[i, 3:6]
            R = so3_exp(dth) @ quat_wxyz_to_mat(poses[i, 3:7])    # exp à GAUCHE (frame monde)
            poses[i, :3] = poses[i, :3] + dt
            poses[i, 3:7] = mat_to_quat_wxyz(R)
    return q_new, poses
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_retract.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/retract.py tests/test_solve_retract.py
git commit -m "feat(holov2): solve/retract — q⊕dv (robot.integrate) + exp SE(3) objet (numpy pur)"
```

---

### Task 3 : `solve/init.py` — seed Holosoma (f=0) + warm_start (f>0)

**Files:**
- Create: `src/solve/init.py`
- Test: `tests/test_solve_init.py`

**Interfaces:**
- Consumes : `retract.mat_to_quat_wxyz`, `retract.quat_wxyz_to_xyzw` (Task 2). `RobotModel.neutral`. `FrameTargets`-like (lit `.style.{link_names, position, orientation}`, `.object_rot (N,3,3)`, `.object_pos (N,3)`).
- Produces :
  - `compute_q_init(frame_targets_0, robot, base_link: str = "pelvis") -> (q (nq,), object_poses (N,7))` — joints neutres ; base free-flyer = cible pelvis du style (pos + orientation wxyz→xyzw) ; objets à leur pose observée.
  - `warm_start(prev_q (nq,), prev_poses (N,7)) -> (q (nq,), object_poses (N,7))` — carry (copies) de f-1.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_init.py
"""init : le seed f=0 place la base free-flyer à la cible pelvis du style (pos + orientation), joints
neutres, objets à leur pose observée ; warm_start recopie l'état de f-1."""
import types

import numpy as np

from src.solve.init import compute_q_init, warm_start
from src.solve.retract import so3_exp, mat_to_quat_wxyz


class _StubRobot:
    """RobotModel free-flyer minimal : neutral() = base identité (quat xyzw, qw à l'index 6) + joints 0."""

    def __init__(self, n_joints):
        self.nq = 7 + n_joints
        self.nv = 6 + n_joints

    def neutral(self):
        q = np.zeros(self.nq)
        q[6] = 1.0                                       # quat xyzw identité : qw = 1
        return q


def _ft(link_names, position, orientation, object_rot, object_pos):
    return types.SimpleNamespace(
        style=types.SimpleNamespace(link_names=link_names, position=position, orientation=orientation),
        object_rot=object_rot, object_pos=object_pos)


def test_base_placed_at_pelvis_target_no_objects():
    robot = _StubRobot(n_joints=2)
    quat_pelvis = mat_to_quat_wxyz(so3_exp(np.array([0.0, 0.0, np.pi / 2])))   # +90° z, wxyz
    ft0 = _ft(link_names=("pelvis", "torso_link"),
              position=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.9]]),
              orientation=np.array([quat_pelvis, [1.0, 0.0, 0.0, 0.0]]),
              object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    q, poses = compute_q_init(ft0, robot)
    assert np.allclose(q[0:3], [1.0, 2.0, 3.0])                                  # base pos = cible pelvis
    assert np.allclose(q[3:7], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-10)  # xyzw
    assert np.allclose(q[7:], 0.0)                                               # joints neutres
    assert poses.shape == (0, 7)


def test_base_keeps_identity_when_orientation_none():
    robot = _StubRobot(n_joints=1)
    ft0 = _ft(link_names=("pelvis",), position=np.array([[5.0, 6.0, 7.0]]), orientation=None,
              object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    q, _ = compute_q_init(ft0, robot)
    assert np.allclose(q[0:3], [5.0, 6.0, 7.0])
    assert np.allclose(q[3:7], [0.0, 0.0, 0.0, 1.0])                             # identité xyzw conservée


def test_objects_seeded_at_observed_pose():
    robot = _StubRobot(n_joints=0)
    ft0 = _ft(link_names=("pelvis",), position=np.zeros((1, 3)),
              orientation=np.array([[1.0, 0.0, 0.0, 0.0]]),
              object_rot=np.stack([np.eye(3)]), object_pos=np.array([[8.0, 9.0, 10.0]]))
    _, poses = compute_q_init(ft0, robot)
    assert poses.shape == (1, 7)
    assert np.allclose(poses[0], [8.0, 9.0, 10.0, 1.0, 0.0, 0.0, 0.0])           # pos + quat wxyz identité


def test_warm_start_copies_state():
    q = np.arange(8.0)
    poses = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    q2, p2 = warm_start(q, poses)
    assert np.allclose(q2, q) and np.allclose(p2, poses)
    q2[0] = 99.0                                                                 # copie défensive
    assert q[0] == 0.0
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_init.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.init`).

- [ ] **Step 3 : Écrire `src/solve/init.py`**

```python
"""init — le seed des variables de décision. PUR, pinocchio/torch-free (consomme une référence
``FrameTargets`` + le ``RobotModel``, jamais l'Evaluator).

``compute_q_init`` (frame 0, idiome Holosoma) : base flottante = cible pelvis du style (position +
orientation), joints **neutres**, objets à leur pose observée — bien meilleur seed qu'une base à
l'origine. Pour G1 le lien root URDF = ``pelvis`` donc base ≡ cible pelvis directe ; un offset
root↔pelvis se composerait ICI (un seul endroit) via ``base_link``. ``warm_start`` : carry de f-1."""
from __future__ import annotations

import numpy as np

from .retract import mat_to_quat_wxyz, quat_wxyz_to_xyzw


def compute_q_init(frame_targets_0, robot, base_link: str = "pelvis") -> tuple[np.ndarray, np.ndarray]:
    """Seed f=0 : ``q = [base_pos = cible pelvis, base_quat = orient pelvis (xyzw), joints = 0]`` +
    objets ``(N,7)`` à leur pose observée (rot -> quat wxyz). ``base_link`` = lien root (G1 : pelvis)."""
    style = frame_targets_0.style
    q = np.array(robot.neutral(), np.float64, copy=True)          # base identité (xyzw) + joints 0
    try:
        idx = tuple(style.link_names).index(base_link)
    except ValueError:
        raise ValueError(
            f"base link {base_link!r} absent de StyleTargets.link_names {tuple(style.link_names)!r}")
    q[0:3] = np.asarray(style.position[idx], np.float64)          # base pos = cible pelvis
    if style.orientation is not None:
        q[3:7] = quat_wxyz_to_xyzw(np.asarray(style.orientation[idx], np.float64))  # wxyz -> xyzw

    rot = np.asarray(frame_targets_0.object_rot, np.float64)      # (N, 3, 3)
    pos = np.asarray(frame_targets_0.object_pos, np.float64)      # (N, 3)
    n = rot.shape[0]
    object_poses = np.zeros((n, 7), np.float64)
    for i in range(n):
        object_poses[i, :3] = pos[i]
        object_poses[i, 3:7] = mat_to_quat_wxyz(rot[i])           # pose objet = quat wxyz
    return q, object_poses


def warm_start(prev_q: np.ndarray, prev_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry de f-1 vers f>0 : copies défensives de l'état précédent."""
    return (np.array(prev_q, np.float64, copy=True), np.array(prev_poses, np.float64, copy=True))
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_init.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/init.py tests/test_solve_init.py
git commit -m "feat(holov2): solve/init — seed Holosoma (base=cible pelvis, joints neutres) + warm_start"
```

---

### Task 4 : `solve/assemble.py` — `evals + refs -> Problem` (appelle les builders Plan B)

**Files:**
- Create: `src/solve/assemble.py`
- Test: `tests/test_solve_assemble.py`

**Interfaces:**
- Consumes (Plan B, verbatim) : `build_style`, `build_contact`, `build_object`, `build_reg`, `build_constraints`, `SolveConfig`. (Plan A) `Problem`. (Task 1) `FrameEval`.
- Produces : `assemble(evals: FrameEval, frame_targets: FrameTargets, geo, robot, cfg) -> Problem`. `geo` = le contexte géodésique par canal lu par `build_contact` (sourcé `ctx.channels` dans le runner). `n_obj = frame_targets.object_rot.shape[0]`.

- [ ] **Step 1 : Écrire le test** (les builders Plan B sont MONKEYPATCHÉS : on teste la concaténation/`Problem` propre à `assemble`, pas la correction des builders — couverte par loop/end-to-end)

```python
# tests/test_solve_assemble.py
"""assemble : concatène build_style/contact/object/reg + build_constraints en UN Problem bien formé
(nv/n_obj corrects, comptes de blocs). Les builders Plan B sont monkeypatchés -> on teste la logique
d'orchestration d'assemble, isolée des internes Plan B."""
import types

import numpy as np

from src.solve.contracts import FrameEval, LinearConstraint, Problem, ResidualBlock, TrustRegion
import src.solve.assemble as A


class _StubRobot:
    def __init__(self, nv):
        self.nv = nv
        self.nq = nv + 1


def _blocks(nv, names):
    return [ResidualBlock(A=np.zeros((1, nv)), c=np.zeros(1), A_obj=None, name=n) for n in names]


def test_assemble_concatenates_into_well_formed_problem(monkeypatch):
    nv, N = 5, 2
    monkeypatch.setattr(A, "_geo_field", lambda ch, orot, opos: object())   # isolate from GeoField internals
    monkeypatch.setattr(A, "build_style", lambda se, st, cfg: _blocks(nv, ["S-pos", "S-rot"]))
    monkeypatch.setattr(A, "build_contact", lambda ce, rf, geo, cfg: _blocks(nv, ["C-D", "C-X"]))
    monkeypatch.setattr(A, "build_object", lambda ce, ev, orot, opos, cfg: _blocks(nv, ["O"]))
    monkeypatch.setattr(A, "build_reg", lambda nv_, cfg: _blocks(nv, ["reg"]))
    monkeypatch.setattr(A, "build_constraints", lambda robot, cfg: (
        [LinearConstraint(A=np.zeros((1, nv)), lb=np.zeros(1), ub=None, A_obj=None, name="jl")],
        [TrustRegion(var="dv", radius=np.ones(nv), norm=-1)]))

    evals = FrameEval(style=object(), contact=types.SimpleNamespace())
    ft = types.SimpleNamespace(
        style=object(),
        robot_interaction=object(),
        env_interaction=object(),
        object_rot=np.zeros((N, 3, 3)),
        object_pos=np.zeros((N, 3)))
    cfg = types.SimpleNamespace(tr_object_pos=0.05, tr_object_rot=0.10)

    prob = A.assemble(evals, ft, geo=("chan0",), robot=_StubRobot(nv), cfg=cfg)
    assert isinstance(prob, Problem)
    assert prob.nv == nv and prob.n_obj == N
    assert len(prob.residuals) == 6                       # 2 style + 2 contact + 1 object + 1 reg
    assert {b.name for b in prob.residuals} == {"S-pos", "S-rot", "C-D", "C-X", "O", "reg"}
    assert len(prob.constraints) == 1 and len(prob.trust_regions) == 2  # dv (build_constraints) + dxi (assemble)
    dxi_tr = next(tr for tr in prob.trust_regions if tr.var == "dxi")
    assert dxi_tr.radius.shape == (N * 6,)


def test_assemble_no_objects_sets_n_obj_zero(monkeypatch):
    nv = 3
    monkeypatch.setattr(A, "_geo_field", lambda ch, orot, opos: object())   # isolate from GeoField internals
    monkeypatch.setattr(A, "build_style", lambda se, st, cfg: _blocks(nv, ["S-pos"]))
    monkeypatch.setattr(A, "build_contact", lambda ce, rf, geo, cfg: [])
    monkeypatch.setattr(A, "build_object", lambda ce, ev, orot, opos, cfg: [])
    monkeypatch.setattr(A, "build_reg", lambda nv_, cfg: _blocks(nv, ["reg"]))
    monkeypatch.setattr(A, "build_constraints", lambda robot, cfg: ([], []))

    evals = FrameEval(style=object(), contact=object())
    ft = types.SimpleNamespace(style=object(), robot_interaction=object(), env_interaction=object(),
                               object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    prob = A.assemble(evals, ft, geo=(), robot=_StubRobot(nv), cfg=object())
    assert prob.n_obj == 0 and len(prob.residuals) == 2 and len(prob.trust_regions) == 0  # no dxi when n_obj=0
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_assemble.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.assemble`).

- [ ] **Step 3 : Écrire `src/solve/assemble.py`**

```python
"""assemble — (evals + refs + geo + robot + cfg) -> ``Problem``. Le pont entre les ÉVALUATIONS
courantes (``FrameEval`` : style FK + champ de contact) et le sous-problème QP linéarisé : appelle les
builders ``terms/`` (poids repliés) et ``terms/constraints`` (limites articulaires + trust-region box),
concatène en UN ``Problem``. PUR — aucune cinématique ici (déléguée aux Eval), aucun cvxpy.

``geo`` = le contexte géodésique par canal que ``build_contact`` lit pour le résidu witness C-X (sourcé
``ctx.channels`` côté runner — chaque ``Channel`` porte son ``geodesic`` + ``sdf``)."""
from __future__ import annotations

import numpy as np

from .contracts import Problem, TrustRegion
from .config import SolveConfig
from .terms._ops import GeoField
from .terms.style import build_style
from .terms.contact import build_contact
from .terms.object import build_object
from .terms.reg import build_reg
from .terms.constraints import build_constraints


def _geo_field(channels, object_rot, object_pos) -> GeoField:
    """Per-frame bundle build_contact needs: object-local frames + geodesic tables per channel.
    Channel.object_idx is None for ground -> -1 / identity frame (GeoField convention)."""
    tables     = tuple(ch.geodesic for ch in channels)
    rot        = np.stack([np.eye(3) if ch.object_idx is None
                           else object_rot[ch.object_idx] for ch in channels])   # (C,3,3)
    pos        = np.stack([np.zeros(3) if ch.object_idx is None
                           else object_pos[ch.object_idx] for ch in channels])   # (C,3)
    object_idx = tuple(-1 if ch.object_idx is None else ch.object_idx for ch in channels)
    return GeoField(tables=tables, rot=rot, pos=pos, object_idx=object_idx)


def assemble(evals, frame_targets, geo, robot, cfg: SolveConfig) -> Problem:
    """Construit le ``Problem`` d'UNE itération SQP. ``n_obj`` est dérivé des poses objet de la frame ;
    si ``n_obj = 0`` les builders objet ne produisent simplement aucun bloc ``A_obj``."""
    se, ce = evals.style, evals.contact
    blocks = []
    blocks += list(build_style(se, frame_targets.style, cfg))
    geo_field = _geo_field(geo, frame_targets.object_rot, frame_targets.object_pos)
    blocks += list(build_contact(ce, frame_targets.robot_interaction, geo_field, cfg))
    blocks += list(build_object(ce, frame_targets.env_interaction,
                                frame_targets.object_rot, frame_targets.object_pos, cfg))
    blocks += list(build_reg(robot.nv, cfg))
    constraints, trust_regions = build_constraints(robot, cfg)
    n_obj = int(frame_targets.object_rot.shape[0])
    trust_regions = list(trust_regions)
    if n_obj > 0:
        obj_r = np.tile(np.concatenate([np.full(3, cfg.tr_object_pos),
                                        np.full(3, cfg.tr_object_rot)]), n_obj)   # (n_obj*6,)
        trust_regions.append(TrustRegion(var="dxi", radius=obj_r, norm=-1))
    return Problem(nv=robot.nv, n_obj=n_obj,
                   residuals=tuple(blocks),
                   constraints=tuple(constraints),
                   trust_regions=tuple(trust_regions))
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_assemble.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/assemble.py tests/test_solve_assemble.py
git commit -m "feat(holov2): solve/assemble — terms + constraints -> Problem (orchestration QP)"
```

---

### Task 5 : `solve/loop.py` — wrapper `evaluate` + itéré SQP `solve_frame`

**Files:**
- Create: `src/solve/loop.py`
- Test: `tests/test_solve_loop.py`

**Interfaces:**
- Consumes : `assemble.assemble` (Task 4), `retract.retract` + `retract.quat_wxyz_to_mat` (Task 2), `FrameInfo`/`FrameEval` (Task 1), `Evaluator` (instance, méthodes `.style`/`.contacts`), `SolveBackend.solve`, `obs.NULL`.
- Produces :
  - `evaluate(evaluator, q (nq,), object_poses (N,7)) -> FrameEval` — convertit `object_poses -> (object_rot, object_pos)` (quat wxyz -> matrice) puis appelle `.style(q)` + `.contacts(q, object_rot, object_pos)`.
  - `cost_breakdown(problem: Problem, step: Step) -> dict[str, float]` — ‖résidu‖² par nom de terme au pas résolu.
  - `solve_frame(evaluator, frame_targets_f, geo, robot, backend, cfg, q0, poses0, n_iter: int | None = None, prof=NULL) -> (q (nq,), poses (N,7), FrameInfo)` — itère `evaluate → assemble → backend.solve → retract`, converge sur `max|dv| < cfg.step_tol` ou `n_iter` (par défaut `cfg.n_iter_per_frame`).

- [ ] **Step 1 : Écrire le test** (synthétique : `evaluate`/`assemble` monkeypatchés en un tracking quadratique `min‖dv + (q − target)‖²` sous box ; backend CVXPY réel Plan A + `StubRobot` Euclidien ; on prouve convergence `q→target`, `‖dv‖→0`, coût décroissant)

```python
# tests/test_solve_loop.py
"""solve_frame : un itéré SQP synthétique (tracking quadratique sous box) converge — q -> target,
‖dv‖ -> 0 (arrêt sur step_tol avant n_iter), et le coût final décroît avec le budget d'itérations.
evaluate/assemble sont monkeypatchés ; le backend CVXPY (Plan A) résout réellement chaque QP."""
import types

import numpy as np

from src.solve.backend.cvxpy import CvxpyBackend
from src.solve.contracts import Problem, ResidualBlock, TrustRegion
import src.solve.loop as L

_TARGET = np.array([1.0])


class _StubRobot:
    def __init__(self):
        self.nq = self.nv = 1

    def integrate(self, q, v):
        return np.asarray(q, np.float64) + np.asarray(v, np.float64)


def _patch(monkeypatch):
    # evaluate renvoie q tel quel ("evals" = la config courante) ; assemble construit le QP de tracking.
    monkeypatch.setattr(L, "evaluate", lambda ev, q, poses: q)

    def fake_assemble(evals, ft, geo, robot, cfg):
        q = np.asarray(evals, np.float64)
        r = ResidualBlock(A=np.eye(1), c=(q - _TARGET), A_obj=None, name="track")
        tr = TrustRegion(var="dv", radius=np.array([0.3]), norm=-1)
        return Problem(nv=1, n_obj=0, residuals=(r,), constraints=(), trust_regions=(tr,))

    monkeypatch.setattr(L, "assemble", fake_assemble)


def _run(n_iter):
    cfg = types.SimpleNamespace(step_tol=1e-6, n_iter_per_frame=n_iter)
    return L.solve_frame(evaluator=None, frame_targets_f=None, geo=None, robot=_StubRobot(),
                         backend=CvxpyBackend(), cfg=cfg, q0=np.array([0.0]),
                         poses0=np.zeros((0, 7)), n_iter=n_iter)


def test_solve_frame_converges_to_target(monkeypatch):
    _patch(monkeypatch)
    q, poses, info = _run(n_iter=50)
    assert abs(q[0] - 1.0) < 1e-3                          # convergence vers la cible (‖dv‖ -> 0)
    assert info.n_iters < 50                               # arrêt anticipé sur step_tol
    assert poses.shape == (0, 7)
    assert "track" in info.cost_by_term


def test_solve_frame_cost_decreases_with_budget(monkeypatch):
    _patch(monkeypatch)
    costs = [_run(n_iter=k)[2].cost for k in (1, 2, 3)]
    assert costs[0] >= costs[1] >= costs[2] - 1e-9         # coût final non croissant avec le budget
    assert costs[2] < costs[0]                             # strictement amélioré
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_loop.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.loop`).

- [ ] **Step 3 : Écrire `src/solve/loop.py`**

```python
"""loop — l'itéré SQP/trust-region par frame. Flux LINÉAIRE explicite (le point que V1 ratait) :
``evaluate -> assemble -> backend.solve -> retract -> converge``. Une seule passe, pas de classe-dieu.

``evaluate`` est le wrapper du seam : l'``Evaluator`` (targets) expose ``.style(q)`` + ``.contacts(q,
object_rot, object_pos)`` mais pas un appel combiné — on convertit ``object_poses (N,7) -> (object_rot,
object_pos)`` puis on assemble un ``FrameEval``. ``prof.span`` vit ici (orchestrateur), jamais dans les
ops pures."""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from .assemble import assemble
from .contracts import FrameEval, FrameInfo, Problem, Step
from .retract import quat_wxyz_to_mat, retract


def evaluate(evaluator, q: np.ndarray, object_poses: np.ndarray) -> FrameEval:
    """Géométrie courante au ``(q, object_poses)`` : style FK + champ de contact. Convertit les poses
    objet ``(N,7)`` (pos + quat wxyz) en ``(object_rot (N,3,3), object_pos (N,3))`` attendus par
    ``Evaluator.contacts``."""
    poses = np.asarray(object_poses, np.float64)
    n = poses.shape[0]
    object_rot = np.empty((n, 3, 3))
    object_pos = np.empty((n, 3))
    for i in range(n):
        object_pos[i] = poses[i, :3]
        object_rot[i] = quat_wxyz_to_mat(poses[i, 3:7])
    return FrameEval(style=evaluator.style(q),
                     contact=evaluator.contacts(q, object_rot, object_pos))


def cost_breakdown(problem: Problem, step: Step) -> dict[str, float]:
    """‖A·dv + A_obj·dξ + c‖² par NOM de terme au pas résolu (l'outil n°1 de tuning des poids).
    Les blocs de même nom s'agrègent."""
    dv = np.asarray(step.dv, np.float64)
    dxi = None if step.dxi is None else np.asarray(step.dxi, np.float64).reshape(-1)
    out: dict[str, float] = {}
    for b in problem.residuals:
        e = b.A @ dv + b.c
        if b.A_obj is not None and dxi is not None:
            e = e + b.A_obj @ dxi
        out[b.name] = out.get(b.name, 0.0) + float(e @ e)
    return out


def solve_frame(evaluator, frame_targets_f, geo, robot, backend, cfg, q0, poses0,
                n_iter: int | None = None, prof=NULL) -> tuple[np.ndarray, np.ndarray, FrameInfo]:
    """Un itéré SQP sur UNE frame depuis le seed ``(q0, poses0)``. Trust-region FIXE (adaptatif =
    incrément futur). Convergence : ``max|dv| < cfg.step_tol`` ou ``n_iter`` atteint (par défaut
    ``cfg.n_iter_per_frame`` ; le runner passe ``cfg.n_iter_first`` au cold start f=0)."""
    max_iter = cfg.n_iter_per_frame if n_iter is None else n_iter
    q = np.array(q0, np.float64, copy=True)
    poses = np.array(poses0, np.float64, copy=True)
    status, cost, cost_by_term, it = "no_iter", float("nan"), {}, 0
    with prof.span("frame_solve"):
        for it in range(1, max_iter + 1):
            evals = evaluate(evaluator, q, poses)
            problem = assemble(evals, frame_targets_f, geo, robot, cfg)
            step = backend.solve(problem)
            status, cost = step.status, step.value
            cost_by_term = cost_breakdown(problem, step)
            if step.dv is None or not np.all(np.isfinite(step.dv)):
                break                                       # backend non-optimal -> arrêt + diagnostic
            q, poses = retract(q, poses, step, robot)
            if float(np.max(np.abs(step.dv))) < cfg.step_tol:
                break
    return q, poses, FrameInfo(n_iters=it, status=status, cost=cost, cost_by_term=cost_by_term)
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_loop.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/loop.py tests/test_solve_loop.py
git commit -m "feat(holov2): solve/loop — wrapper evaluate + itéré SQP solve_frame (convergence + breakdown)"
```

---

### Task 6 : `solve/runner.py` — entrée publique `solve(...) -> SolveTrajectory` (end-to-end)

**Files:**
- Create: `src/solve/runner.py`
- Modify: `src/solve/__init__.py` (étendre la surface publique)
- Test: `tests/test_solve_runner.py`

**Interfaces:**
- Consumes : `targets.Evaluator` (surface publique), `make_backend` (Plan A), `compute_q_init`/`warm_start` (Task 3), `solve_frame` (Task 5), `SolveConfig` (Plan B), `SolveTrajectory`/`FrameInfo` (Task 1), `InteractionContext` (`.robot`, `.channels`), `GroundedScene`.
- Produces : `solve(grounded, ctx, frame_targets, config, *, robot_name: str | None = None, prof=NULL) -> SolveTrajectory`. Construit l'`Evaluator` 1×, le backend 1×, `geo = ctx.channels`, boucle les frames (f=0 `compute_q_init` + `config.n_iter_first` ; f>0 `warm_start` + `config.n_iter_per_frame`), collecte `qpos (T,nq)` / `object_poses (T,N,7)` / `info`.

- [ ] **Step 1 : Écrire le test** (data-gated HODome ; calque `test_evaluator.py` pour `prepare`, construit quelques `FrameTargets` via `process_frame`, résout, assert qpos fini + statut optimal + pelvis ≈ cible style)

```python
# tests/test_solve_runner.py
"""solve end-to-end (data-gated HODome) : prepare -> quelques FrameTargets -> solve. Vérifie une qpos
finie de bonne forme, des objets (T,N,7), un statut optimal par frame, la sanité des joints, et le
pelvis solveur ~ cible pelvis du style (le seed Holosoma + le suivi de style le tiennent proche)."""
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
from src.solve.contracts import SolveTrajectory
from src.solve.runner import solve
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
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X / corr / G1 URDF absent")


@_SKIP
def test_solve_end_to_end_on_real_assets(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    T = min(2, g.n_frames)                                   # max_frames bas (suite rapide)
    frame_targets = [process_frame(g, ctx, spec.robot, f) for f in range(T)]

    traj = solve(g, ctx, frame_targets, SolveConfig(), robot_name="g1")

    assert isinstance(traj, SolveTrajectory)
    assert traj.qpos.shape == (T, ctx.robot.nq)
    assert np.isfinite(traj.qpos).all()
    assert traj.object_poses.shape == (T, len(ctx.object_clouds), 7)
    assert np.isfinite(traj.object_poses).all()
    assert len(traj.info) == T

    # joints : finis et dans une bande de sanité (les limites dures vivent dans build_constraints ; le
    # protocol RobotModel n'expose pas les bornes -> on borne grossièrement + on exige un statut feasible).
    assert np.all(np.abs(traj.qpos[:, 7:]) < 2 * np.pi)
    for fi in traj.info:
        assert fi.status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(fi.cost)

    # pelvis solveur ~ cible pelvis du style (seed = cible pelvis exacte + suivi de style).
    ev = Evaluator(ctx, "g1")
    pidx = tuple(frame_targets[0].style.link_names).index("pelvis")
    for f in range(T):
        pos_solver = ev.style(traj.qpos[f]).position[pidx]
        pos_target = frame_targets[f].style.position[pidx]
        assert np.linalg.norm(pos_solver - pos_target) < 0.15      # < 15 cm
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_runner.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.runner`) — ou SKIP si la donnée HODome est absente (le gate doit néanmoins importer `src.solve.runner`, donc l'erreur d'import prime tant que le module n'existe pas).

- [ ] **Step 3 : Écrire `src/solve/runner.py`**

```python
"""runner — ENTRÉE PUBLIQUE de l'étage solve : ``solve(grounded, ctx, frame_targets, config)`` ->
``SolveTrajectory``. Construit l'``Evaluator`` (targets) 1× et le backend (Plan A) 1×, puis boucle les
frames : f=0 -> ``compute_q_init`` (seed Holosoma) + budget ``n_iter_first`` ; f>0 -> ``warm_start``
(carry de f-1) + budget ``n_iter_per_frame``. ``prof.span`` (séquence) vit ici. Reste pinocchio/torch-free
(cinématique cachée dans ``ctx.robot``) ; cvxpy n'arrive que via le backend (lazy).

``robot_name`` clé la recette de style (links suivis) de l'``Evaluator`` : il est lu de l'argument
explicite, sinon de ``config.robot_name`` (cf. assomption d'intégration Plan B)."""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from ..targets import Evaluator
from .backend import make_backend
from .config import SolveConfig
from .contracts import SolveTrajectory
from .init import compute_q_init, warm_start
from .loop import solve_frame


def solve(grounded, ctx, frame_targets, config: SolveConfig, *, robot_name: str | None = None,
          prof=NULL) -> SolveTrajectory:
    """Boucle online sur les frames -> ``SolveTrajectory``. ``frame_targets`` = ``list[FrameTargets]``
    (sortie ``targets.pipeline``). ``grounded`` est accepté pour la cohérence du seam public (provenance /
    futurs targets centroïdaux) ; la boucle n'en dépend pas directement (tout passe par
    ``frame_targets`` + ``ctx``)."""
    name = robot_name if robot_name is not None else getattr(config, "robot_name", None)
    if name is None:
        raise ValueError("robot_name requis (argument explicite ou config.robot_name) pour l'Evaluator")

    evaluator = Evaluator(ctx, name)
    backend = make_backend(config.backend)
    robot = ctx.robot
    geo = ctx.channels                                     # contexte géodésique/SDF par canal (build_contact)

    q = None
    poses = None
    qpos_rows: list[np.ndarray] = []
    poses_rows: list[np.ndarray] = []
    info_rows = []
    with prof.span("sequence", T=len(frame_targets)):
        for f, ft in enumerate(frame_targets):
            if f == 0:
                q, poses = compute_q_init(ft, robot)
                n_iter = config.n_iter_first
            else:
                q, poses = warm_start(q, poses)
                n_iter = config.n_iter_per_frame
            q, poses, fi = solve_frame(evaluator, ft, geo, robot, backend, config, q, poses,
                                       n_iter=n_iter, prof=prof)
            qpos_rows.append(q)
            poses_rows.append(poses)
            info_rows.append(fi)

    T = len(frame_targets)
    qpos = np.asarray(qpos_rows, np.float64) if qpos_rows else np.zeros((0, robot.nq))
    n_obj = poses_rows[0].shape[0] if poses_rows else 0
    object_poses = (np.asarray(poses_rows, np.float64) if n_obj
                    else np.zeros((T, 0, 7), np.float64))
    return SolveTrajectory(qpos=qpos, object_poses=object_poses, info=tuple(info_rows))
```

- [ ] **Step 4 : Étendre `src/solve/__init__.py`** (APPEND après les exports Plan A)

```python
from .config import SolveConfig
from .contracts import FrameEval, FrameInfo, SolveTrajectory
from .runner import solve

__all__ += ["solve", "SolveConfig", "SolveTrajectory", "FrameInfo", "FrameEval"]
```

- [ ] **Step 5 : Lancer, vérifier le succès (ou SKIP propre si donnée absente)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_runner.py -q`
Expected: PASS (1 test) si la donnée HODome est présente, sinon SKIP (« HODome data / … absent »).

Run (sanity import de toute la surface) : `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; from src.solve import solve, SolveConfig, SolveTrajectory; assert 'cvxpy' not in sys.modules and 'torch' not in sys.modules; print('solve runner import torch/cvxpy-free ok')"`
Expected: `solve runner import torch/cvxpy-free ok`

- [ ] **Step 6 : Commit**

```bash
git add src/solve/runner.py src/solve/__init__.py tests/test_solve_runner.py
git commit -m "feat(holov2): solve/runner — entrée publique solve(...) -> SolveTrajectory (boucle SQP end-to-end)"
```

---

## Self-Review

**1. Spec coverage** (vs `2026-06-30-solve-stage-design.md`, périmètre Plan C = orchestration) :
- `contracts.py` `SolveTrajectory` + `FrameInfo` (+ `FrameEval` pour le seam evaluate→assemble) → Task 1. ✅
- `init.py` `compute_q_init` (seed Holosoma : base = cible pelvis pos+orient, joints neutres, objets pose observée ; G1 root = pelvis via `base_link`) + `warm_start` → Task 3. ✅
- `retract.py` `q ⊕ dv` (robot.integrate) + exp SE(3) objet (quaternion + Rodrigues numpy pur) → Task 2. ✅
- `assemble.py` terms (`build_style/contact/object/reg`) + `constraints` -> `Problem`, `n_obj=0` => pas de bloc objet → Task 4. ✅
- `loop.py` itéré SQP `evaluate -> assemble -> backend.solve -> retract -> converge` (‖dv‖<step_tol ou n_iter), `prof.span`, gestion statut non-optimal (arrêt + diagnostic, pas de crash) → Task 5. ✅
- `runner.py` entrée publique `solve(grounded, ctx, frame_targets, config)`, Evaluator 1× + backend 1×, f=0 init / f>0 warm_start, `n_iter_first`/`n_iter_per_frame`, collecte qpos/object_poses/info → Task 6. ✅
- `__init__.py` surface publique (`solve`, `SolveConfig`, `SolveTrajectory`) → Task 6. ✅
- Pinocchio/torch-free (cinématique via `RobotModel`, exp SE(3) numpy, cvxpy confiné au backend) → assert d'import Task 6 Step 5 + `retract`/`init` purs numpy. ✅
- `cost_by_term` = norme résidu par terme → `cost_breakdown` (Task 5). ✅
- Tests : `retract` round-trip + exp SE(3) connue (Task 2) ; `loop` convergence + coût décroît (Task 5) ; end-to-end data-gated HODome (qpos fini, statut optimal, pelvis ≈ cible) (Task 6). ✅ (Le `evaluate` combiné absent de `targets` est fourni en wrapper — Task 5.)

**2. Placeholder scan** : aucun `TBD`/`TODO`/`...`. Chaque step de code porte le code complet ; chaque step de run porte la commande + l'attendu. ✅

**3. Type consistency** :
- `compute_q_init(ft0, robot, base_link="pelvis") -> (q, object_poses)`, `warm_start(prev_q, prev_poses)` — appelés tels quels par `runner`. ✅
- `retract(q, object_poses, step, robot) -> (q, poses)` ; helpers `so3_exp/quat_wxyz_to_mat/mat_to_quat_wxyz/quat_wxyz_to_xyzw` — réutilisés par `init` (`mat_to_quat_wxyz`, `quat_wxyz_to_xyzw`) et `loop` (`quat_wxyz_to_mat`). ✅
- `assemble(evals, frame_targets, geo, robot, cfg) -> Problem` consomme `FrameEval.style/contact` (produit par `evaluate`) et les builders Plan B **verbatim** (`build_style/contact/object/reg`, `build_constraints`). ✅
- `solve_frame(evaluator, frame_targets_f, geo, robot, backend, cfg, q0, poses0, n_iter=None, prof=NULL) -> (q, poses, FrameInfo)` ; `evaluate(evaluator, q, object_poses) -> FrameEval` cohérent avec `Evaluator.style/.contacts`. ✅
- `solve(grounded, ctx, frame_targets, config, *, robot_name=None, prof=NULL) -> SolveTrajectory` ; `Step.dv (nv,)`/`Step.dxi (n_obj,6)` ; `object_poses (T,N,7)` partout. ✅
- `Problem(nv, n_obj, residuals, constraints, trust_regions)`, `ResidualBlock(A,c,A_obj,name)`, `TrustRegion(var,radius,norm)`, `Step(dv,dxi,value,status)` — utilisés conformes à Plan A. ✅

**4. Open assumptions à réconcilier avec Plan B (signalées dans le plan)** :
- **Wrapper `evaluate`** : `targets.Evaluator` n'expose PAS de `evaluate(q, poses)` combiné — Plan C fournit le wrapper (`solve/loop.py:evaluate`) qui convertit `(N,7) -> (object_rot, object_pos)` et appelle `.style`/`.contacts`. Pas de modif de `targets`.
- **`geo` → bundle `GeoField`** (RÉSOLU vs Plan B) : `build_contact(contact_eval, robot_field_ref, geo, cfg)` attend un `GeoField(tables, rot, pos, object_idx)` (défini dans `terms/_ops.py`), PAS le tuple brut de `Channel`. `runner.solve` passe le tuple statique `ctx.channels` ; c'est `assemble._geo_field(channels, object_rot, object_pos)` qui construit le `GeoField` PER-FRAME (rot/pos depuis `FrameTargets.object_rot/pos` ; `ch.object_idx is None` → `-1` / frame identité = sol). NE JAMAIS passer `ctx.channels` brut à `build_contact`.
- **Convention exp SE(3) objet** : tangente world-aligned `(δt, δθ)` (LOCAL_WORLD_ALIGNED) ; `new_pos = pos + δt`, `new_R = exp(δθ)·R` (exp à GAUCHE, frame monde). CONVENTION VÉRIFIÉE contre `eval.py` : `cloud_jac_self` (∂p/∂δt=I, ∂p/∂δθ=−[p−t]_×) et `probe_jac_obj` (∂x/∂δt=−Rᵀ, ∂x/∂δθ=Rᵀ[p−t]_×) sont exactement cohérents avec l'exp à GAUCHE — NE PAS inverser en `R·exp(δθ)`. Le retract est correct tel quel.
- **`robot_name`** : `runner.solve` le lit de l'argument explicite, sinon `config.robot_name`. Si `SolveConfig` (Plan B) n'expose pas `robot_name`, le passer explicitement (le test end-to-end le fait : `solve(..., robot_name="g1")`).
- **Test data-gated end-to-end** : limites articulaires non vérifiées strictement (le Protocol `RobotModel` n'expose pas les bornes) — on borne grossièrement `|joints| < 2π` et on exige un statut `optimal`/`optimal_inaccurate` (faisabilité du QP, donc box/limites respectées par construction du backend). Le pelvis est vérifié à < 15 cm (seed exact + suivi de style) ; tolérance à resserrer une fois les poids Plan B figés.
