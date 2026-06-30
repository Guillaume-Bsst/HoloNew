# solve — terms/_ops + residual builders + config + constraints (Plan B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the **residual layer** of the `solve/` stage — `solve/config.py` (`SolveConfig` knobs), the reusable residual ops `solve/terms/_ops.py`, the per-concern term builders (`style`/`contact`/`object`/`reg`), and `constraints.py` (joint limits + box trust region) — each a pure numpy function returning `ResidualBlock`/`LinearConstraint`/`TrustRegion` (the Plan A contracts), with weights **folded into `A` and `c`**.

**Architecture:** Plan A (DONE) locked `Problem`/`ResidualBlock(A (m,nv), c (m,), A_obj (m,n_obj*6)|None, name)`/`LinearConstraint`/`TrustRegion`/`Step` + the `CvxpyBackend`. Plan B fills `solve/terms/`: `_ops.py` holds the complex residual-only ops (`world_normal`, `dist_jac`, `geo_chain`, `so3_log`, `se3_log_world`, `quat_to_rot`, object scatter, the `GeoField` channel bundle); the `build_*` builders linearise each concern into `ResidualBlock`s consuming the `targets` Eval/Ref surface (`StyleEval`/`StyleTargets`, `ContactEval`/`RobotInteractionTargets`/`EnvironmentInteractionTargets`, `geo_value_grad`). Plan C (later) calls these from `assemble.py`; Plan B keeps `terms` **internal** (not exported by `solve/__init__`). The objective is Gauss-Newton: `‖A·δv + A_obj·δξ + c‖²` per block.

**Tech Stack:** Python, numpy (float64), pytest. No cvxpy/torch/pinocchio in any Plan B module (the backend owns cvxpy; `targets`/`prepare` contracts pulled in are numpy-only).

## Global Constraints

- `solve/config.py` is **stdlib-only** (frozen dataclasses); `solve/terms/*` are **numpy-only** (may import the numpy-only `targets`/`prepare` contracts + `targets.interaction.geodesic.geo_value_grad`). **No cvxpy/torch/pinocchio.**
- Compute in **float64**.
- Imports **relative** inside `src/` (`from ..contracts import ...`, `from ...targets.contracts import ...`); **absolute** (`from src.…`) in `tests/`.
- Tests live in **`HoloV2/tests/`**, run from `HoloV2/` with `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<f> -q`. (Plan B tests are pure-numpy → fast, no `max_frames`.)
- Contract invariants → `raise ValueError` at construction (style of `MultiChannelField.__post_init__`); `assert` only for internal invariants.
- **Weights are FOLDED into `A` and `c`** — `ResidualBlock` carries no separate weight field.
- Tangent layout (locked by `RobotModel`/`ContactEval`): `v` is the pinocchio free-flyer tangent, `nv = 6 + dof`, ordered `[base_trans(0:3), base_rot(3:6), joints(6:6+dof)]`. Object tangent `δξ` is world-aligned `(δt(0:3), δθ(3:6))` per object, length `n_obj*6`.
- Quaternions **wxyz**; object/channel directions & witnesses for OBJECT channels are **object-LOCAL** (per `MultiChannelField` docstring) → mapped to world with `world_normal(R_i, …)`; the GROUND channel is world (`R_i = I`).
- Commits **conventional, French**. **NEVER tag Claude** (no `Co-Authored-By`/mention). Author `Guillaume-Bsst`.

### Canonical builder signatures (Plan C consumes these verbatim)

```text
build_style(style_eval, style_targets, cfg)                       -> list[ResidualBlock]   # S-pos, S-rot
build_contact(contact_eval, robot_field_ref, geo, cfg)           -> list[ResidualBlock]   # C-D, C-X
build_object(contact_eval, env_refs, object_rot, object_pos, cfg) -> list[ResidualBlock]  # CO-D, O
build_reg(nv, cfg)                                                -> list[ResidualBlock]   # reg
build_constraints(robot, cfg)            -> tuple[list[LinearConstraint], list[TrustRegion]]
```

### Open assumptions (resolved here, flagged for Plan C / review)

1. **Geodesic gradient sign/frame** — `geo_value_grad(table, source_idx, query_xyz)` returns `(value, ∇value)` where `∇value` is the gradient of the SAME scalar returned as `value`, expressed in the **channel frame** (object-LOCAL for object channels, world for ground). C-X uses `c = w_cx·value` and `A = w_cx·geo_chain(world_normal(R_i, ∇value), point_jac)` — Gauss-Newton is sign-consistent **because the same `value` feeds both `c` and the gradient**. The object-local→world map `world_normal(R_i, ∇value)` is the **frame guard**; the object δξ coupling uses the **raw local** gradient `dist_jac(∇value_local, probe_jac_obj[c])`. The gradient is APPROXIMATE for non-linear fields (~1e-2 per `geodesic.py`) → C-X FD tolerance is loose (~2e-2).
2. **Joint-limit DOFs → δv** — joint DOF `j` maps to `v[6+j]` and to `q[7+j]` (free-flyer is `q[:7]=pos(3)+quat(4)`, `v[:6]`), 1:1 for G1's revolute joints. The step box is `lower_j − q_j ≤ δv[6+j] ≤ upper_j − q_j`. `build_constraints(robot, cfg)` has **no live `q`**, so v1 linearises at the **neutral** joints `q0 = robot.neutral()[7:]` (EXACT at the cold-start frame, where joints START neutral; an over/under-tight static box afterwards). Live-q rebasing is DEFERRED to a future increment (would add a `q_joints` param to `build_constraints`); v1 linearises joint limits at neutral q0 and relies on the per-DOF joint trust region to keep steps in range.
3. **Object-channel frame `R_i` in `build_contact`** — object-channel directions/geodesic gradients are object-local and need `R_i` for the δv Jacobian, but `ContactEval` does not carry object poses. We thread per-channel world frames through the **`geo` argument**, defined as a `GeoField(tables, rot, pos, object_idx)` bundle (Plan C assembles it from `InteractionContext` geodesic tables + `FrameTargets.object_rot/pos`; ground frame = I). This keeps the positional signature intact and makes `world_normal(geo.rot[c], …)` uniform across channels.
4. **Channel → object index** — channel 0 = ground (`object_idx=-1`, frame I); channels `1..N` → object `0..N-1` (frame = that object's world pose). Carried explicitly in `GeoField.object_idx` (contact) / derived as `c-1` (object env fields).
5. **Active mask** — C/CO terms emit a row only for pairs `active` in the **reference** field (the demonstrated contact: `robot_field_ref.field.active` for C, `env_refs.per_object[i].field.active` for CO). The self-diagonal of env fields is already neutralised upstream (`active=False`).
6. **`build_object` scope** — implements **CO-D** + **O** fully. **CO-X (geodesic) is DEFERRED**: it needs the per-channel geodesic tables, which the canonical 5-arg `build_object` signature does not carry (unlike `build_contact`'s `geo`). It is identical to C-X once a `geo` bundle is threaded; flagged for Plan C.

---

### Task 1 : `solve/config.py` — `SolveConfig`

**Files:**
- Create: `src/solve/config.py`
- Test: `tests/test_solve_config.py`

**Interfaces:**
- Consumes: nothing (stdlib-only).
- Produces: `SolveConfig` (frozen) with per-term weights `w_pos/w_rot/w_cd/w_cx/w_cod/w_cox/w_obj/w_reg`, contact activation `contact_gate: bool` + `contact_d_ref_scale: float`, per-DOF trust radii `tr_base_pos/tr_base_rot/tr_joints/tr_object_pos/tr_object_rot`, `n_iter_first/n_iter_per_frame: int`, `step_tol: float`, `backend: str`.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_config.py
"""SolveConfig : défaut valide + override inline + validation ValueError."""
import pytest

from src.solve.config import SolveConfig


def test_defaults_construct():
    c = SolveConfig()
    assert c.w_pos > 0.0 and c.w_reg > 0.0
    assert c.tr_joints > 0.0 and c.tr_base_pos > 0.0
    assert c.n_iter_first >= c.n_iter_per_frame >= 1
    assert c.backend == "cvxpy"


def test_inline_override():
    c = SolveConfig(w_cd=5.0, tr_joints=0.2, n_iter_first=12)
    assert c.w_cd == 5.0 and c.tr_joints == 0.2 and c.n_iter_first == 12
    assert c.w_pos == SolveConfig().w_pos          # untouched fields keep defaults


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        SolveConfig(w_pos=-1.0)


def test_nonpositive_radius_raises():
    with pytest.raises(ValueError):
        SolveConfig(tr_joints=0.0)


def test_bad_backend_raises():
    with pytest.raises(ValueError):
        SolveConfig(backend="ipopt")


def test_bad_iter_raises():
    with pytest.raises(ValueError):
        SolveConfig(n_iter_per_frame=0)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_config.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.config`).

- [ ] **Step 3 : Écrire `src/solve/config.py`**

```python
"""Config of the ``solve`` stage — the QP KNOBS (one frozen, stdlib-only dataclass), co-located with
the stage (rule #2). ``SolveConfig()`` IS the default; override inline
(``SolveConfig(w_cd=5.0, tr_joints=0.2)``), exactly like ``targets.config.TargetsConfig``.

Three knob families:
  * per-term WEIGHTS (``w_pos`` … ``w_reg``) — the cost gains folded into each ``ResidualBlock``'s
    ``A``/``c`` by the ``solve/terms`` builders (the #1 tuning lever; cf. ``FrameInfo.cost_by_term``);
  * contact ACTIVATION — ``contact_gate`` (rows only for demonstrated-active pairs) + a soft
    ``contact_d_ref_scale`` falloff that down-weights far demonstrated contacts (the V1 ``alpha``);
  * trust region + loop — per-DOF box radii (heterogeneous units: base m / base rad / joints rad /
    object m+rad) and the SQP iteration budget / convergence tol / backend name.

Per-link / per-channel weight VECTORS are a future refinement; v1 weights are scalars."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolveConfig:
    """All knobs of the ``solve`` QP loop. Frozen, stdlib-only, importable anywhere."""

    # --- per-term weights (folded into ResidualBlock A and c) ---------------------------------
    w_pos: float = 1.0     # S-pos : style link position tracking
    w_rot: float = 0.5     # S-rot : style link orientation tracking
    w_cd: float = 2.0      # C-D   : robot contact distance (vs channel)
    w_cx: float = 1.0      # C-X   : robot contact geodesic (witness on the surface)
    w_cod: float = 1.0     # CO-D  : object self-contact distance (object vs ground/other objects)
    w_cox: float = 0.5     # CO-X  : object self-contact geodesic (DEFERRED in v1, see plan)
    w_obj: float = 1.0     # O     : object pose anchor to its observed pose
    w_reg: float = 1e-2    # reg   : step damping (well-conditioned QP)

    # --- contact activation ------------------------------------------------------------------
    contact_gate: bool = True        # rows only for pairs active in the demonstrated (reference) field
    contact_d_ref_scale: float = 0.05  # soft falloff: weight *= exp(-(max(d_ref,0)/scale)^2);
                                       # <= 0 disables the falloff (active pairs weight 1)

    # --- per-DOF box trust-region radii (TrustRegion.radius, per-DOF, norm=-1) ----------------
    tr_base_pos: float = 0.05   # free-flyer translation step (m)   -> v[0:3]
    tr_base_rot: float = 0.10   # free-flyer rotation step (rad)    -> v[3:6]
    tr_joints: float = 0.10     # actuated joint step (rad)         -> v[6:6+dof]
    tr_object_pos: float = 0.05  # object translation step (m)      -> δξ[0:3] per object
    tr_object_rot: float = 0.10  # object rotation step (rad)       -> δξ[3:6] per object

    # --- SQP loop ----------------------------------------------------------------------------
    n_iter_first: int = 10       # iterations for the cold-start frame (absorbs joint refinement)
    n_iter_per_frame: int = 4    # iterations for warm-started frames
    step_tol: float = 1e-4       # convergence: ‖dv‖ < step_tol
    backend: str = "cvxpy"       # solve backend (Plan A factory key)
    robot_name: str | None = None  # optional label forwarded by runner.solve (no validation)

    def __post_init__(self) -> None:
        for name in ("w_pos", "w_rot", "w_cd", "w_cx", "w_cod", "w_cox", "w_obj", "w_reg"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"SolveConfig.{name} must be >= 0, got {getattr(self, name)}")
        for name in ("tr_base_pos", "tr_base_rot", "tr_joints", "tr_object_pos", "tr_object_rot"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"SolveConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.step_tol <= 0.0:
            raise ValueError(f"SolveConfig.step_tol must be > 0, got {self.step_tol}")
        if self.n_iter_first < 1 or self.n_iter_per_frame < 1:
            raise ValueError("SolveConfig.n_iter_* must be >= 1")
        if self.backend not in ("cvxpy",):
            raise ValueError(f"SolveConfig.backend must be 'cvxpy', got {self.backend!r}")
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_config.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/config.py tests/test_solve_config.py
git commit -m "feat(holov2): solve/config — SolveConfig (poids, activation contact, trust-region par-DOF, boucle)"
```

---

### Task 2 : `solve/terms/_ops.py` — les ops réutilisables des résidus

**Files:**
- Create: `src/solve/terms/__init__.py`
- Create: `src/solve/terms/_ops.py`
- Test: `tests/test_solve_terms_ops.py`

**Interfaces:**
- Consumes: `prepare.contracts.GeodesicTable` (typing of `GeoField`, numpy-only).
- Produces:
  - `world_normal(R, n_local) -> n_world` — `R` (3,3) or (M,3,3), `n_local` (...,3) → (...,3).
  - `dist_jac(direction, jac) -> (M, K)` — `direction` (M,3), `jac` (M,3,K); `∂(n·p)/∂step`.
  - `geo_chain(grad, jac) -> (M, K)` — same contraction as `dist_jac` (geodesic gradient chain).
  - `so3_log(R_ref, R_cur) -> (L, 3)` — world-frame error `log(R_cur·R_refᵀ)`.
  - `se3_log_world(R_ref, p_ref, R_cur, p_cur) -> (N, 6)` — world-aligned `[p_cur−p_ref, so3_log(R_ref,R_cur)]`.
  - `quat_to_rot(wxyz) -> (..., 3, 3)`.
  - `scatter_obj(block, object_idx, n_obj) -> (m, n_obj*6)` — place a per-object `(m,6)` block.
  - `GeoField(tables, rot, pos, object_idx)` — per-channel geodesic tables + world frames (Assumption 3).

- [ ] **Step 1 : Écrire le test (unitaire + différences finies par op)**

```python
# tests/test_solve_terms_ops.py
"""Reusable residual ops: unit values + finite-difference (FD) of the linearised model each op
encodes. The builders' correctness reduces to these ops + a contraction, so they carry the FD load."""
import numpy as np

from src.solve.terms._ops import (GeoField, dist_jac, geo_chain, quat_to_rot, scatter_obj,
                                   se3_log_world, so3_log, world_normal)


def _rand_rot(rng):
    a = rng.standard_normal(3); a /= np.linalg.norm(a)
    th = rng.uniform(0.2, 2.5)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def test_world_normal_single_and_batched():
    rng = np.random.default_rng(0)
    R = _rand_rot(rng)
    n = rng.standard_normal((5, 3))
    got = world_normal(R, n)
    assert got.shape == (5, 3)
    assert np.allclose(got, n @ R.T)                       # n_world[m] = R @ n[m]
    Rb = np.stack([_rand_rot(rng) for _ in range(5)])       # per-row rotation
    gb = world_normal(Rb, n)
    assert np.allclose(gb, np.einsum("mij,mj->mi", Rb, n))
    assert np.allclose(world_normal(np.eye(3), n), n)       # identity (ground channel)


def test_dist_jac_matches_directional_derivative():
    # d(v) = nᵀ(p0 + J v - w) is linear in v -> dist_jac(n,J) @ v == d(v) - d(0) exactly.
    rng = np.random.default_rng(1)
    M, nv = 4, 9
    n = rng.standard_normal((M, 3)); J = rng.standard_normal((M, 3, nv))
    A = dist_jac(n, J)
    assert A.shape == (M, nv)
    v = rng.standard_normal(nv)
    d_lin = np.einsum("mi,mij,j->m", n, J, v)               # exact directional derivative
    assert np.allclose(A @ v, d_lin)


def test_geo_chain_is_the_same_contraction():
    rng = np.random.default_rng(2)
    g = rng.standard_normal((3, 3)); J = rng.standard_normal((3, 3, 6))
    assert np.allclose(geo_chain(g, J), dist_jac(g, J))


def test_so3_log_recovers_axis_angle_and_jacobian():
    rng = np.random.default_rng(3)
    R_ref = _rand_rot(rng)
    u = rng.standard_normal(3); u /= np.linalg.norm(u)
    for th in (1e-4, 0.3, 3.0, np.pi - 1e-3):
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        E = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)   # exp(th [u]x), world-left
        R_cur = E @ R_ref                                            # R_cur Rrefᵀ = E
        e = so3_log(R_ref[None], R_cur[None])[0]
        assert np.allclose(e, th * u, atol=1e-6)
    # Jacobian convention: d/dα so3_log(R_ref, exp(α[w]x) R_ref) |0 = w  (world angular vel = jac_rot v)
    w = rng.standard_normal(3)
    eps = 1e-6
    Kw = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    Rp = (np.eye(3) + eps * Kw) @ R_ref
    de = so3_log(R_ref[None], Rp[None])[0] / eps
    assert np.allclose(de, w, atol=1e-4)


def test_se3_log_world_blocks():
    rng = np.random.default_rng(4)
    R_ref = np.stack([_rand_rot(rng), _rand_rot(rng)])
    R_cur = np.stack([_rand_rot(rng), _rand_rot(rng)])
    p_ref = rng.standard_normal((2, 3)); p_cur = rng.standard_normal((2, 3))
    e = se3_log_world(R_ref, p_ref, R_cur, p_cur)
    assert e.shape == (2, 6)
    assert np.allclose(e[:, :3], p_cur - p_ref)
    assert np.allclose(e[:, 3:], so3_log(R_ref, R_cur))
    # identical poses -> zero residual
    z = se3_log_world(R_ref, p_ref, R_ref, p_ref)
    assert np.allclose(z, 0.0)


def test_quat_to_rot_known():
    R = quat_to_rot(np.array([[1.0, 0.0, 0.0, 0.0]]))       # identity wxyz
    assert np.allclose(R[0], np.eye(3))
    R90 = quat_to_rot(np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]]))  # +90° about z
    assert np.allclose(R90[0] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_scatter_obj_places_block():
    rng = np.random.default_rng(5)
    blk = rng.standard_normal((3, 6))
    A_obj = scatter_obj(blk, object_idx=1, n_obj=3)
    assert A_obj.shape == (3, 18)
    assert np.allclose(A_obj[:, 6:12], blk)
    assert np.allclose(A_obj[:, :6], 0.0) and np.allclose(A_obj[:, 12:], 0.0)


def test_geofield_shapes_and_ground_identity():
    pts = np.zeros((4, 3), np.float32)
    gf = GeoField(tables=(None, None), rot=np.stack([np.eye(3), np.eye(3)]),
                  pos=np.zeros((2, 3)), object_idx=(-1, 0))
    assert gf.rot.shape == (2, 3, 3) and gf.object_idx == (-1, 0)
    assert np.allclose(world_normal(gf.rot[0], pts.astype(float)), pts)  # ground frame = I
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_ops.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms._ops`).

- [ ] **Step 3 : Créer `src/solve/terms/__init__.py`**

```python
"""``solve/terms`` — residual builders (``style``/``contact``/``object``/``reg``/``constraints``) +
their complex residual-only ops (``_ops``). INTERNAL to ``solve`` (used by ``solve/assemble`` in
Plan C); deliberately NOT re-exported by ``solve/__init__`` so the public ``solve`` import stays
cvxpy/torch/pinocchio-free. Each builder folds the ``SolveConfig`` weights into the ``ResidualBlock``
``A``/``c`` (no separate weight) and returns Plan A contract objects."""
```

- [ ] **Step 4 : Écrire `src/solve/terms/_ops.py`**

```python
"""Complex ops AT THE EXCLUSIVE SERVICE OF THE RESIDUALS — the ``solve``-specific contractions / frame
maps / manifold logs that the spec keeps OUT of the (ref-free) ``targets`` evaluator. Pure numpy
(float64), no I/O, no mutation. Shared by C and CO (rule #8 homogeneity): one ``dist_jac`` contraction,
one ``world_normal`` frame map, one ``so3_log``.

Conventions (locked by ``targets``):
  * Robot point Jacobians (``point_jac``, ``jac_pos``, ``jac_rot``) are WORLD / LOCAL_WORLD_ALIGNED.
  * OBJECT-channel ``direction``/``witness``/geodesic gradients are OBJECT-LOCAL -> map to world with
    ``world_normal(R_i, …)`` before contracting with a world Jacobian; contract with the RAW local
    vector against the object tangent Jacobian ``probe_jac_obj`` (object-local).
  * Orientation residual is the WORLD-frame log ``log(R_cur·R_refᵀ)`` to pair with the world angular
    Jacobian ``jac_rot`` (``omega_world = jac_rot·v``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...prepare.contracts import GeodesicTable


def world_normal(R: np.ndarray, n_local: np.ndarray) -> np.ndarray:
    """Map an object-LOCAL direction/normal/gradient to WORLD: ``n_world = R · n_local``.
    ``R`` (3,3) [one frame] or (M,3,3) [per row]; ``n_local`` (...,3). Ground channel: ``R = I``."""
    R = np.asarray(R, np.float64)
    n = np.asarray(n_local, np.float64)
    return np.einsum("...ij,...j->...i", R, n)


def dist_jac(direction: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂(directionᵀ·point)/∂step`` = ``directionᵀ·jac`` row-wise. ``direction`` (M,3), ``jac``
    (M,3,K) -> (M,K). The signed-distance gradient w.r.t. the point is the contact unit normal, so
    this gives ``∂d/∂step`` for both the robot tangent (K=nv, ``point_jac``) and the object tangent
    (K=6, ``probe_jac_obj`` / ``cloud_jac_self``)."""
    direction = np.asarray(direction, np.float64)
    jac = np.asarray(jac, np.float64)
    return np.einsum("mi,mij->mj", direction, jac)


def geo_chain(grad: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂geo/∂step`` = ``gradᵀ·jac`` — the SAME contraction as ``dist_jac`` (the geodesic gradient is
    tangent to the surface; its normal component, if any, is annihilated by the tangent Jacobian).
    Kept as a named op for builder readability (rule #8)."""
    return dist_jac(grad, jac)


def quat_to_rot(wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion(s) ``wxyz`` (...,4) -> rotation matrix (...,3,3). Normalises defensively."""
    q = np.asarray(wxyz, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _log_one(E: np.ndarray) -> np.ndarray:
    """SO(3) log of a single rotation matrix -> rotation vector (3,). Robust near 0 and π."""
    cos = np.clip((np.trace(E) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-7:                                    # near identity: first-order
        return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if np.pi - theta < 1e-4:                            # near π: axis from the symmetric part
        Aerr = (E + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(Aerr), 0.0, None))
        # fix signs from the off-diagonal of (E - Eᵀ)
        s = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        axis = np.where(s < 0, -axis, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return theta * axis
    w = theta / (2.0 * np.sin(theta))
    return w * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def so3_log(R_ref: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """World-frame orientation error per row: ``log(R_cur·R_refᵀ)`` (L,3,3),(L,3,3) -> (L,3).
    Pairs with the world angular Jacobian ``jac_rot`` (Gauss-Newton: ``A=jac_rot``, ``c=so3_log``)."""
    R_ref = np.asarray(R_ref, np.float64); R_cur = np.asarray(R_cur, np.float64)
    E = np.einsum("lij,lkj->lik", R_cur, R_ref)         # R_cur · R_refᵀ
    return np.stack([_log_one(E[l]) for l in range(E.shape[0])])


def se3_log_world(R_ref: np.ndarray, p_ref: np.ndarray,
                  R_cur: np.ndarray, p_cur: np.ndarray) -> np.ndarray:
    """World-aligned SE(3) error per object: ``[p_cur − p_ref, log(R_cur·R_refᵀ)]`` (N,6). Matches the
    world-aligned object tangent ``δξ = (δt, δθ)`` (the O term anchors the object to its observed pose)."""
    p_ref = np.asarray(p_ref, np.float64); p_cur = np.asarray(p_cur, np.float64)
    out = np.empty((p_ref.shape[0], 6), np.float64)
    out[:, :3] = p_cur - p_ref
    out[:, 3:] = so3_log(R_ref, R_cur)
    return out


def scatter_obj(block: np.ndarray, object_idx: int, n_obj: int) -> np.ndarray:
    """Place a per-object ``(m,6)`` Jacobian block into the full ``(m, n_obj*6)`` object coupling
    matrix (sparse: zeros for the other objects). ``object_idx`` in ``[0, n_obj)``."""
    block = np.asarray(block, np.float64)
    m = block.shape[0]
    A_obj = np.zeros((m, n_obj * 6), np.float64)
    A_obj[:, object_idx * 6:(object_idx + 1) * 6] = block
    return A_obj


@dataclass(frozen=True)
class GeoField:
    """Per-channel geodesic tables + channel WORLD frames — the bundle ``build_contact`` reads as its
    ``geo`` argument. Assembled by Plan C from ``InteractionContext`` (the geodesic tables) +
    ``FrameTargets.object_rot/pos``. Lets ``build_contact`` (a) read the geodesic field per channel and
    (b) map object-LOCAL field directions/gradients to world via ``world_normal(rot[c], …)`` —
    uniformly across channels (ground frame = identity). See plan Assumption 3."""

    tables: tuple[GeodesicTable | None, ...]  # (C,) per-channel geodesic table; None -> no C-X row
    rot: np.ndarray                           # (C, 3, 3) per-channel world rotation (ground = I)
    pos: np.ndarray                           # (C, 3)    per-channel world translation (ground = 0)
    object_idx: tuple[int, ...]               # (C,) channel -> object index (-1 for ground)
```

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_ops.py -q`
Expected: PASS (8 tests).

- [ ] **Step 6 : Commit**

```bash
git add src/solve/terms/__init__.py src/solve/terms/_ops.py tests/test_solve_terms_ops.py
git commit -m "feat(holov2): solve/terms/_ops — world_normal/dist_jac/geo_chain/so3_log/se3_log_world/scatter + GeoField (FD-testés)"
```

---

### Task 3 : `solve/terms/style.py` — S-pos, S-rot

**Files:**
- Create: `src/solve/terms/style.py`
- Test: `tests/test_solve_terms_style.py`

**Interfaces:**
- Consumes: `StyleEval(position (L,3), rotation (L,3,3), jac_pos (L,3,nv), jac_rot (L,3,nv), link_names)`, `StyleTargets(link_names, position (L,3), orientation (L,4 wxyz)|None)`, `SolveConfig`; `_ops.quat_to_rot/so3_log`; Plan A `ResidualBlock`.
- Produces: `build_style(style_eval, style_targets, cfg) -> list[ResidualBlock]` — `[S-pos]` (+ `[S-rot]` if `orientation` set), `A_obj=None`, weights folded.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_terms_style.py
"""build_style : formes + linéarisation A·δv + c vs FD du résidu réel (pos linéaire ; rot = so3_log)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.style import build_style
from src.solve.terms._ops import quat_to_rot, so3_log
from src.targets.contracts import StyleEval, StyleTargets


def _rand_rot(rng):
    a = rng.standard_normal(3); a /= np.linalg.norm(a); th = rng.uniform(0.2, 2.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def _make(rng, L=3, nv=9, with_rot=True):
    pos = rng.standard_normal((L, 3)); rot = np.stack([_rand_rot(rng) for _ in range(L)])
    jp = rng.standard_normal((L, 3, nv)); jr = rng.standard_normal((L, 3, nv))
    names = tuple(f"link{i}" for i in range(L))
    ev = StyleEval(position=pos, rotation=rot, jac_pos=jp, jac_rot=jr, link_names=names)
    tgt_pos = rng.standard_normal((L, 3))
    ori = np.stack([_rot_to_quat(_rand_rot(rng)) for _ in range(L)]) if with_rot else None
    tgt = StyleTargets(link_names=names, position=tgt_pos, orientation=ori)
    return ev, tgt


def test_spos_shapes_and_linear():
    rng = np.random.default_rng(0); ev, tgt = _make(rng)
    cfg = SolveConfig()
    blocks = {b.name: b for b in build_style(ev, tgt, cfg)}
    sp = blocks["S-pos"]
    L, nv = 3, 9
    assert sp.A.shape == (L * 3, nv) and sp.c.shape == (L * 3,) and sp.A_obj is None
    # residual r(v) = w·((pos + jp·v) − tgt) is linear -> A·v + c == r(v) exactly.
    v = rng.standard_normal(nv)
    r = cfg.w_pos * ((ev.position + np.einsum("lij,j->li", ev.jac_pos, v)) - tgt.position).reshape(-1)
    assert np.allclose(sp.A @ v + sp.c, r)
    assert np.allclose(sp.c, (cfg.w_pos * (ev.position - tgt.position)).reshape(-1))


def test_srot_linearization_vs_fd():
    rng = np.random.default_rng(1); ev, tgt = _make(rng)
    cfg = SolveConfig()
    sr = {b.name: b for b in build_style(ev, tgt, cfg)}["S-rot"]
    L, nv = 3, 9
    assert sr.A.shape == (L * 3, nv) and sr.c.shape == (L * 3,)
    R_ref = quat_to_rot(tgt.orientation)
    assert np.allclose(sr.c, (cfg.w_rot * so3_log(R_ref, ev.rotation)).reshape(-1))
    # FD: perturb world orientation by jac_rot·v (R_cur(v) = exp([jac_rot·v]x) R_cur). Check A·v matches
    # the first-order change of so3_log -> validates A = w_rot·jac_rot (the world-frame convention).
    v = rng.standard_normal(nv) * 1e-5
    R_pert = np.empty_like(ev.rotation)
    for l in range(L):
        wl = ev.jac_rot[l] @ v
        K = np.array([[0, -wl[2], wl[1]], [wl[2], 0, -wl[0]], [-wl[1], wl[0], 0]])
        R_pert[l] = (np.eye(3) + K) @ ev.rotation[l]
    c_pert = (cfg.w_rot * so3_log(R_ref, R_pert)).reshape(-1)
    assert np.allclose(sr.A @ v, c_pert - sr.c, atol=1e-6)


def test_position_only_skips_srot():
    rng = np.random.default_rng(2); ev, tgt = _make(rng, with_rot=False)
    names = [b.name for b in build_style(ev, tgt, SolveConfig())]
    assert names == ["S-pos"]
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_style.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms.style`).

- [ ] **Step 3 : Écrire `src/solve/terms/style.py`**

```python
"""S-pos / S-rot — the STYLE residual builder (robot only, ``δv``). Linearises the per-link position
and orientation tracking error from ``StyleEval`` (current FK + Jacobians) vs ``StyleTargets`` (the
reference posture). Weights ``cfg.w_pos`` / ``cfg.w_rot`` are folded into ``A`` and ``c`` (rule:
``ResidualBlock`` carries no separate weight). ``A_obj = None`` (style does not touch the object)."""
from __future__ import annotations

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import quat_to_rot, so3_log
from ...targets.contracts import StyleEval, StyleTargets


def build_style(style_eval: StyleEval, style_targets: StyleTargets,
                cfg: SolveConfig) -> list[ResidualBlock]:
    """``[S-pos]`` (+ ``[S-rot]`` if the targets carry orientation). Stacks the L links into ``3L``
    rows. ``S-pos``: ``A = w_pos·jac_pos`` (3L,nv), ``c = w_pos·(pos_cur − pos_ref)``. ``S-rot``:
    ``A = w_rot·jac_rot``, ``c = w_rot·so3_log(R_ref, R_cur)`` (R_ref via quat→R, world-frame log)."""
    L, nv = style_eval.position.shape[0], style_eval.jac_pos.shape[2]
    blocks: list[ResidualBlock] = []

    A_pos = (cfg.w_pos * style_eval.jac_pos).reshape(L * 3, nv)
    c_pos = (cfg.w_pos * (style_eval.position - style_targets.position)).reshape(L * 3)
    blocks.append(ResidualBlock(A=A_pos, c=c_pos, A_obj=None, name="S-pos"))

    if style_targets.orientation is not None:
        R_ref = quat_to_rot(style_targets.orientation)                  # (L,3,3) from wxyz
        A_rot = (cfg.w_rot * style_eval.jac_rot).reshape(L * 3, nv)
        c_rot = (cfg.w_rot * so3_log(R_ref, style_eval.rotation)).reshape(L * 3)
        blocks.append(ResidualBlock(A=A_rot, c=c_rot, A_obj=None, name="S-rot"))

    return blocks
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_style.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/terms/style.py tests/test_solve_terms_style.py
git commit -m "feat(holov2): solve/terms/style — build_style S-pos/S-rot (poids repliés, linéarisation FD-testée)"
```

---

### Task 4 : `solve/terms/contact.py` — C-D, C-X (géodésique)

**Files:**
- Create: `src/solve/terms/contact.py`
- Test: `tests/test_solve_terms_contact.py`

**Interfaces:**
- Consumes: `ContactEval(field MultiChannelField (C,M), point_jac (M,3,nv), probe_jac_obj (C,M,3,6), env)`, `RobotInteractionTargets(field)`, `GeoField` (Task 2), `SolveConfig`; `_ops.world_normal/dist_jac/geo_chain/scatter_obj`; `targets.interaction.geodesic.geo_value_grad/nearest_index`; Plan A `ResidualBlock`.
- Produces: `build_contact(contact_eval, robot_field_ref, geo, cfg) -> list[ResidualBlock]`.
  - **C-D** (one block, all active pairs stacked): `c = w·(d_cur − d_ref)`, `A = w·dist_jac(n_world, point_jac)`, `A_obj = w·scatter_obj(dist_jac(n_local, probe_jac_obj[c]), object_idx)`.
  - **C-X** (one block, active pairs with a geodesic table): `c = w·value`, `A = w·geo_chain(world_normal(R_i, grad_local), point_jac)`, `A_obj = w·scatter_obj(dist_jac(grad_local, probe_jac_obj[c]), object_idx)`.
  - Per-row weight `w = w_term · alpha(d_ref)` (gating + falloff). Empty list if no active pairs.

- [ ] **Step 1 : Écrire le test (C-D FD vs distance ; C-X assemblage + FD sur champ linéaire)**

```python
# tests/test_solve_terms_contact.py
"""build_contact : C-D linéarisation vs FD de la distance ; C-X assemblage vs geo_value_grad + FD
(champ géodésique synthétique LINÉAIRE -> gradient exact). FLAG repris : C-X mappe le gradient
object-local -> monde via world_normal(R_i, .) ; même `value` pour c et le gradient (signe cohérent)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.contact import build_contact
from src.solve.terms._ops import GeoField, dist_jac, world_normal
from src.targets.contracts import (ContactEval, MultiChannelField, RobotInteractionTargets)
from src.prepare.contracts import GeodesicTable
from src.targets.interaction.geodesic import geo_value_grad, nearest_index


def _mcf(C, M, rng, active):
    return MultiChannelField(
        distance=rng.standard_normal((C, M)),
        direction=rng.standard_normal((C, M, 3)),
        witness=rng.standard_normal((C, M, 3)),
        active=active, channels=tuple(["ground"] + [f"obj{i}" for i in range(C - 1)]))


def _eval(C, M, nv, rng, active):
    return ContactEval(
        field=_mcf(C, M, rng, active),
        point_jac=rng.standard_normal((M, 3, nv)),
        probe_jac_obj=rng.standard_normal((C, M, 3, 6)),
        env=())


def test_cd_linearization_vs_distance_fd():
    rng = np.random.default_rng(0)
    C, M, nv = 1, 4, 9                              # ground channel only (R_i = I), n_obj=0
    active = np.ones((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    geo = GeoField(tables=(None,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    cfg = SolveConfig(contact_d_ref_scale=0.0)     # disable falloff -> weight = w_cd on active rows
    cd = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-D"]
    assert cd.A.shape == (M, nv) and cd.c.shape == (M,) and cd.A_obj is None
    # per-row: c = w·(d_cur − d_ref) ; A·v = w·(n_world·point_jac)·v  (linear distance model)
    n_world = world_normal(np.eye(3), ev.field.direction[0])      # ground: local == world
    assert np.allclose(cd.c, cfg.w_cd * (ev.field.distance[0] - ref.field.distance[0]))
    v = rng.standard_normal(nv)
    assert np.allclose(cd.A @ v, cfg.w_cd * (dist_jac(n_world, ev.point_jac) @ v))


def test_cd_object_channel_couples_dxi():
    rng = np.random.default_rng(1)
    C, M, nv = 2, 3, 9                              # ground + 1 object -> n_obj = 1
    active = np.zeros((C, M), bool); active[1, :] = True   # only object-channel pairs active
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])        # object world rotation (90° z)
    geo = GeoField(tables=(None, None), rot=np.stack([np.eye(3), R1]),
                   pos=np.zeros((2, 3)), object_idx=(-1, 0))
    cfg = SolveConfig(contact_d_ref_scale=0.0)
    cd = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-D"]
    assert cd.A_obj is not None and cd.A_obj.shape == (M, 6)
    # object side uses the LOCAL direction against probe_jac_obj of channel 1
    expect = cfg.w_cd * dist_jac(ev.field.direction[1], ev.probe_jac_obj[1])
    assert np.allclose(cd.A_obj, expect)
    # robot side uses the WORLD-mapped normal world_normal(R1, dir_local)
    expect_A = cfg.w_cd * dist_jac(world_normal(R1, ev.field.direction[1]), ev.point_jac)
    assert np.allclose(cd.A, expect_A)


def test_cx_assembly_and_fd_on_linear_field():
    rng = np.random.default_rng(2)
    C, M, nv = 1, 2, 9
    active = np.ones((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    # synthetic LINEAR geodesic field f(p) = a·p over a small point set -> geo_value_grad exact.
    P = 30
    pts = rng.standard_normal((P, 3)).astype(np.float32)
    a = np.array([0.7, -0.3, 0.4])
    geo_rows = (pts.astype(np.float64) @ a).astype(np.float32)
    table = GeodesicTable(points=pts, normals=np.tile([0, 0, 1.0], (P, 1)).astype(np.float32),
                          geo=np.tile(geo_rows, (P, 1)), name="ground")
    geo = GeoField(tables=(table,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    cfg = SolveConfig(contact_d_ref_scale=0.0)
    cx = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-X"]
    assert cx.A.shape == (M, nv) and cx.c.shape == (M,)
    # recompute the expected (value, grad) the builder must use
    src = nearest_index(table.points, ref.field.witness[0])       # source from the reference witness
    val, grad = geo_value_grad(table, src, ev.field.witness[0])    # query = current witness
    assert np.allclose(cx.c, cfg.w_cx * val, atol=1e-3)
    # ground channel: world_normal(I, grad) == grad ; A = w·geo_chain(grad, point_jac)
    v = rng.standard_normal(nv)
    A_expect = cfg.w_cx * np.einsum("mi,mij->mj", grad, ev.point_jac)
    assert np.allclose(cx.A @ v, A_expect @ v, atol=2e-2)         # loose: geodesic grad approx (Assumption 1)


def test_no_active_pairs_returns_empty():
    rng = np.random.default_rng(3)
    C, M, nv = 1, 3, 9
    active = np.zeros((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    geo = GeoField(tables=(None,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    assert build_contact(ev, ref, geo, SolveConfig()) == []
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_contact.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms.contact`).

- [ ] **Step 3 : Écrire `src/solve/terms/contact.py`**

```python
"""C-D / C-X — the robot CONTACT residual builder. Couples ``δv`` (robot) and ``δξ`` (object channel):
the robot's M correspondence points are driven onto the demonstrated contact geometry. C-D linearises
the signed-distance error (current vs the transported human reference); C-X linearises the geodesic
WITNESS residual (drive the current witness toward the demonstrated contact location, target 0).

Frame convention (see plan Assumptions 1 & 3): OBJECT-channel ``direction``/witness/geodesic gradient
are object-LOCAL; map to world with ``world_normal(geo.rot[c], …)`` for the world ``point_jac`` (δv)
contraction, and contract the RAW local vector with ``probe_jac_obj[c]`` for the object tangent (δξ).
Ground channel: ``geo.rot[c] = I`` (local == world), no object coupling. Only pairs ``active`` in the
REFERENCE field become rows. Weights ``cfg.w_cd``/``cfg.w_cx`` × the per-row activation ``alpha(d_ref)``
are folded into ``A``/``c``."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import GeoField, dist_jac, geo_chain, scatter_obj, world_normal
from ...targets.contracts import ContactEval, RobotInteractionTargets
from ...targets.interaction.geodesic import geo_value_grad, nearest_index


def _alpha(d_ref: np.ndarray, cfg: SolveConfig) -> np.ndarray:
    """Per-row contact activation weight from the demonstrated distance: soft falloff
    ``exp(−(max(d_ref,0)/scale)²)`` (closer demonstrated contact -> ~1, far -> ~0). ``scale <= 0``
    disables the falloff (every active row weight 1). Gating to active rows is done by the caller."""
    if cfg.contact_d_ref_scale <= 0.0:
        return np.ones_like(d_ref)
    return np.exp(-(np.clip(d_ref, 0.0, None) / cfg.contact_d_ref_scale) ** 2)


def build_contact(contact_eval: ContactEval, robot_field_ref: RobotInteractionTargets,
                  geo: GeoField, cfg: SolveConfig) -> list[ResidualBlock]:
    """``[C-D]`` (+ ``[C-X]`` if any active pair has a geodesic table). Stacks all active (channel,
    point) pairs into rows; folds ``w · alpha(d_ref)`` into ``A``/``c``. ``n_obj`` from the field's
    object channels. Returns ``[]`` if no active pair."""
    field_cur, field_ref = contact_eval.field, robot_field_ref.field
    C, M = field_cur.n_channels, field_cur.n_points
    nv = contact_eval.point_jac.shape[2]
    n_obj = sum(1 for j in geo.object_idx if j >= 0)
    active = field_ref.active                                       # demonstrated contacts (Assumption 5)

    cd_A, cd_c, cd_Aobj = [], [], []
    cx_A, cx_c, cx_Aobj = [], [], []
    for cidx in range(C):
        rows = np.nonzero(active[cidx])[0]
        if rows.size == 0:
            continue
        R_i = geo.rot[cidx]
        obj = geo.object_idx[cidx]
        dir_local = field_cur.direction[cidx, rows]                 # (k,3) object-local (world if ground)
        n_world = world_normal(R_i, dir_local)                      # (k,3) -> world for point_jac
        w = (cfg.w_cd * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]   # (k,1)

        # --- C-D : signed-distance error ---
        cd_A.append(w * dist_jac(n_world, contact_eval.point_jac[rows]))        # (k,nv)
        cd_c.append((w[:, 0]) * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
        if obj >= 0:
            blk = w * dist_jac(dir_local, contact_eval.probe_jac_obj[cidx, rows])  # (k,6) object-local
            cd_Aobj.append(scatter_obj(blk, obj, n_obj))
        else:
            cd_Aobj.append(np.zeros((rows.size, n_obj * 6)) if n_obj else None)

        # --- C-X : geodesic witness residual (only channels with a table) ---
        table = geo.tables[cidx]
        if table is not None:
            wx = (cfg.w_cx * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]
            src = nearest_index(table.points, field_ref.witness[cidx, rows])    # source = ref witness
            val, grad_local = geo_value_grad(table, src, field_cur.witness[cidx, rows])  # query = cur
            grad_world = world_normal(R_i, grad_local)              # object-local grad -> world
            cx_A.append(wx * geo_chain(grad_world, contact_eval.point_jac[rows]))
            cx_c.append((wx[:, 0]) * val)                           # target 0 (no ref to subtract)
            if obj >= 0:
                gblk = wx * dist_jac(grad_local, contact_eval.probe_jac_obj[cidx, rows])
                cx_Aobj.append(scatter_obj(gblk, obj, n_obj))
            elif n_obj:
                cx_Aobj.append(np.zeros((rows.size, n_obj * 6)))

    blocks: list[ResidualBlock] = []
    if cd_c:
        A_obj = np.vstack(cd_Aobj) if (n_obj and all(b is not None for b in cd_Aobj)) else None
        blocks.append(ResidualBlock(A=np.vstack(cd_A), c=np.concatenate(cd_c),
                                    A_obj=A_obj, name="C-D"))
    if cx_c:
        A_obj = np.vstack(cx_Aobj) if (n_obj and len(cx_Aobj) == len(cx_c)) else None
        blocks.append(ResidualBlock(A=np.vstack(cx_A), c=np.concatenate(cx_c),
                                    A_obj=A_obj, name="C-X"))
    return blocks
```

> **Note assemblage `A_obj`** : pour C-D, chaque canal pousse un bloc `(k, n_obj*6)` (zéros pour les canaux non-objet) ⇒ empilables sans trou. Si `n_obj == 0`, `A_obj = None` partout (cohérent avec `Problem.__post_init__`). Pour C-X, seuls les canaux à table contribuent ; quand des canaux mixtes (objet + sol) coexistent, les canaux sol poussent un bloc de zéros pour garder l'empilement homogène (la branche `elif n_obj`).

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_contact.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/terms/contact.py tests/test_solve_terms_contact.py
git commit -m "feat(holov2): solve/terms/contact — build_contact C-D/C-X (couplage δv↔δξ, géodésique world_normal, FD)"
```

---

### Task 5 : `solve/terms/object.py` — CO-D, O (CO-X différé)

**Files:**
- Create: `src/solve/terms/object.py`
- Test: `tests/test_solve_terms_object.py`

**Interfaces:**
- Consumes: `ContactEval.env: tuple[ContactEnvEval]` (`field (C,P_i)`, `cloud_jac_self (P_i,3,6)`, `probe_jac_obj (C,P_i,3,6)`), `EnvironmentInteractionTargets(per_object)`, `object_rot (N,3,3)`, `object_pos (N,3)`, `SolveConfig`; `_ops.world_normal/dist_jac/scatter_obj/se3_log_world`; Plan A `ResidualBlock`.
- Produces: `build_object(contact_eval, env_refs, object_rot, object_pos, cfg) -> list[ResidualBlock]`.
  - **CO-D** (object self-contact): `A = 0 (m,nv)`, `A_obj` = self term `dist_jac(n_world, cloud_jac_self)` into the object's own slot **+** (object↔object) `dist_jac(n_local, probe_jac_obj[c])` into the other object's slot; `c = w_cod·(d_cur − d_ref)`.
  - **O** (pose anchor): `A = 0 (6N,nv)`, `A_obj = w_obj·I (6N)`, `c = w_obj·se3_log_world(ref, cur)` (= 0 at the linearisation point in v1; Assumption 6 — general formula implemented & unit-tested via `se3_log_world`).
  - **CO-X DEFERRED** (Assumption 6: no `geo` in the signature).

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_terms_object.py
"""build_object : CO-D (A=0, A_obj via cloud_jac_self/probe_jac_obj) + O (ancre I, c=se3_log_world).
nv passé via point_jac (M,3,nv). CO-X différée (pas de `geo` dans la signature) — non émise."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.object import build_object
from src.solve.terms._ops import dist_jac, world_normal, se3_log_world
from src.targets.contracts import (ContactEval, ContactEnvEval, MultiChannelField,
                                    EnvironmentInteractionTargets, RobotInteractionTargets)


def _mcf(C, P, rng, active, names):
    return MultiChannelField(distance=rng.standard_normal((C, P)),
                             direction=rng.standard_normal((C, P, 3)),
                             witness=rng.standard_normal((C, P, 3)), active=active, channels=names)


def test_co_d_self_and_O(monkeypatch=None):
    rng = np.random.default_rng(0)
    N, C, P, nv, M = 1, 2, 3, 9, 1            # 1 object => channels (ground, obj0); diagonal inactive
    names = ("ground", "obj0")
    act = np.zeros((C, P), bool); act[0, :] = True       # object cloud vs GROUND active (self-channel off)
    env_field = _mcf(C, P, rng, act, names)
    env = ContactEnvEval(field=env_field, cloud_jac_self=rng.standard_normal((P, 3, 6)),
                         probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    ev = ContactEval(field=_mcf(C, M, rng, np.zeros((C, M), bool), names),
                     point_jac=rng.standard_normal((M, 3, nv)),
                     probe_jac_obj=rng.standard_normal((C, M, 3, 6)), env=(env,))
    refs = EnvironmentInteractionTargets(per_object=(_mcf(C, P, rng, act, names),))
    R0 = np.eye(3)[None]; p0 = np.zeros((1, 3))
    cfg = SolveConfig()
    blocks = {b.name: b for b in build_object(ev, refs, R0, p0, cfg)}
    cod = blocks["CO-D"]
    assert cod.A.shape == (P, nv) and np.allclose(cod.A, 0.0)
    assert cod.A_obj.shape == (P, N * 6)
    # ground channel: world frame = I, self term = w·dist_jac(dir, cloud_jac_self) in object 0 slot
    dir0 = env_field.direction[0]                         # ground -> world == local
    expect = cfg.w_cod * dist_jac(world_normal(np.eye(3), dir0), env.cloud_jac_self)
    assert np.allclose(cod.A_obj, expect)
    assert np.allclose(cod.c, cfg.w_cod * (env_field.distance[0] - refs.per_object[0].field.distance[0]))
    # O term: anchor block = w_obj·I, c = w_obj·se3_log_world(ref,cur) = 0 at the linearisation point
    o = blocks["O"]
    assert o.A.shape == (N * 6, nv) and np.allclose(o.A, 0.0)
    assert np.allclose(o.A_obj, cfg.w_obj * np.eye(N * 6))
    assert np.allclose(o.c, 0.0)


def test_object_object_coupling_two_objects():
    rng = np.random.default_rng(1)
    N, C, P, nv, M = 2, 3, 2, 9, 1           # channels (ground, obj0, obj1)
    names = ("ground", "obj0", "obj1")
    # object 0's cloud vs channel 2 (object 1) active -> couples δξ0 (self) AND δξ1 (probe)
    act = np.zeros((C, P), bool); act[2, :] = True
    env0_field = _mcf(C, P, rng, act, names)
    env0 = ContactEnvEval(field=env0_field, cloud_jac_self=rng.standard_normal((P, 3, 6)),
                          probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    env1 = ContactEnvEval(field=_mcf(C, P, rng, np.zeros((C, P), bool), names),
                          cloud_jac_self=rng.standard_normal((P, 3, 6)),
                          probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    ev = ContactEval(field=_mcf(C, M, rng, np.zeros((C, M), bool), names),
                     point_jac=rng.standard_normal((M, 3, nv)),
                     probe_jac_obj=rng.standard_normal((C, M, 3, 6)), env=(env0, env1))
    refs = EnvironmentInteractionTargets(per_object=(env0_field, env1.field))
    R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    object_rot = np.stack([np.eye(3), R1]); object_pos = np.zeros((2, 3))
    cfg = SolveConfig()
    cod = {b.name: b for b in build_object(ev, refs, object_rot, object_pos, cfg)}["CO-D"]
    assert cod.A_obj.shape == (P, N * 6)
    dir_local = env0_field.direction[2]
    # self contribution (object 0, world frame = R1 of channel 2) -> slot 0
    self_blk = cfg.w_cod * dist_jac(world_normal(R1, dir_local), env0.cloud_jac_self)
    assert np.allclose(cod.A_obj[:, 0:6], self_blk)
    # cross contribution (object 1 of channel 2, local dir) -> slot 1
    cross_blk = cfg.w_cod * dist_jac(dir_local, env0.probe_jac_obj[2])
    assert np.allclose(cod.A_obj[:, 6:12], cross_blk)


def test_O_nonzero_when_pose_drifts():
    # se3_log_world is the engine of c; check it is non-trivial when cur != ref (the general formula).
    rng = np.random.default_rng(2)
    R_ref = np.eye(3)[None]; p_ref = np.zeros((1, 3))
    R_cur = np.array([[[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]]); p_cur = np.array([[0.1, 0.0, 0.0]])
    e = se3_log_world(R_ref, p_ref, R_cur, p_cur)
    assert e.shape == (1, 6) and not np.allclose(e, 0.0)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_object.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms.object`).

- [ ] **Step 3 : Écrire `src/solve/terms/object.py`**

```python
"""CO-D / O — the OBJECT-as-variable residual builder (object only, ``δξ``; ``A = 0``). CO-D drives the
object's OWN contacts to stay consistent (object vs ground / vs other objects): object ``i``'s cloud
points sit on the scene channels; their motion couples through ``cloud_jac_self`` (object ``i``) and,
for an object↔object channel, ``probe_jac_obj`` (the other object). O anchors each object to its
OBSERVED pose (``se3_log_world``). The object is pulled by C (Task 4), retained by CO-D + O.

Frame convention (plan Assumption 4): channel 0 = ground (world frame I); channel ``c>=1`` = object
``c−1`` (its world pose ``object_rot[c−1]``). Self term uses the WORLD-mapped normal (``cloud_jac_self``
is world); the object↔object cross term uses the RAW LOCAL normal against ``probe_jac_obj``. The self
diagonal (object ``i`` vs its own channel) is already ``active=False`` upstream, so it emits no row.

NOTE — CO-X (geodesic) is DEFERRED (plan Assumption 6): it needs the per-channel geodesic tables, which
the canonical 5-arg ``build_object`` signature does not carry (unlike ``build_contact``'s ``geo``). It
is identical to C-X once a ``geo`` bundle is threaded; ``cfg.w_cox`` is reserved for it. O's ``c`` is
zero at the linearisation point in v1 (current pose == observed at frame start); the general
``se3_log_world(ref, cur)`` formula is implemented so a Plan-C live current pose yields a non-zero
anchor."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import dist_jac, scatter_obj, se3_log_world, world_normal
from ...targets.contracts import ContactEval, EnvironmentInteractionTargets


def build_object(contact_eval: ContactEval, env_refs: EnvironmentInteractionTargets,
                 object_rot: np.ndarray, object_pos: np.ndarray,
                 cfg: SolveConfig) -> list[ResidualBlock]:
    """``[CO-D]`` (if any active object self-contact) + ``[O]`` (always, one 6-row block per object).
    ``A = 0`` for both (object terms touch only ``δξ``). ``nv`` from ``point_jac``; ``N`` objects."""
    N = object_rot.shape[0]
    nv = contact_eval.point_jac.shape[2]
    blocks: list[ResidualBlock] = []

    # --- CO-D : object self-consistency, per object cloud i, stacked over active (channel, point) ---
    cod_A, cod_c, cod_Aobj = [], [], []
    for i, env in enumerate(contact_eval.env):
        field_cur = env.field
        field_ref = env_refs.per_object[i].field
        C = field_cur.n_channels
        for cidx in range(C):
            rows = np.nonzero(field_ref.active[cidx])[0]
            if rows.size == 0:
                continue
            jc = cidx - 1                                     # channel -> object index (-1 = ground)
            R_c = np.eye(3) if jc < 0 else object_rot[jc]     # channel world frame
            dir_local = field_cur.direction[cidx, rows]       # (k,3) channel-local (world if ground)
            n_world = world_normal(R_c, dir_local)            # for cloud_jac_self (world)
            w = cfg.w_cod

            # self term: object i moves its own cloud point -> object i slot
            self_blk = w * dist_jac(n_world, env.cloud_jac_self[rows])     # (k,6)
            A_obj = scatter_obj(self_blk, i, N)
            # object<->object term: channel jc != i moves the probe -> object jc slot (raw local normal)
            if jc >= 0 and jc != i:
                cross_blk = w * dist_jac(dir_local, env.probe_jac_obj[cidx, rows])
                A_obj = A_obj + scatter_obj(cross_blk, jc, N)

            cod_A.append(np.zeros((rows.size, nv)))
            cod_c.append(w * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
            cod_Aobj.append(A_obj)

    if cod_c:
        blocks.append(ResidualBlock(A=np.vstack(cod_A), c=np.concatenate(cod_c),
                                    A_obj=np.vstack(cod_Aobj), name="CO-D"))

    # --- O : anchor each object to its observed pose (current == observed at the linearisation pt) ---
    e = se3_log_world(object_rot, object_pos, object_rot, object_pos)      # (N,6) -> 0 in v1 (Assumption 6)
    blocks.append(ResidualBlock(A=np.zeros((N * 6, nv)), c=cfg.w_obj * e.reshape(N * 6),
                                A_obj=cfg.w_obj * np.eye(N * 6), name="O"))
    return blocks
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_object.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/terms/object.py tests/test_solve_terms_object.py
git commit -m "feat(holov2): solve/terms/object — build_object CO-D + O (δξ seul, couplage objet↔objet ; CO-X différée)"
```

---

### Task 6 : `solve/terms/reg.py` — reg

**Files:**
- Create: `src/solve/terms/reg.py`
- Test: `tests/test_solve_terms_reg.py`

**Interfaces:**
- Consumes: `nv: int`, `SolveConfig`; Plan A `ResidualBlock`.
- Produces: `build_reg(nv, cfg) -> list[ResidualBlock]` — `[reg]` with `A = w_reg·I(nv)`, `c = 0(nv)`, `A_obj=None`.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_terms_reg.py
"""build_reg : damping de pas A = w_reg·I, c = 0."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.reg import build_reg


def test_reg_block():
    nv = 9
    cfg = SolveConfig(w_reg=0.05)
    blocks = build_reg(nv, cfg)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.name == "reg" and b.A_obj is None
    assert b.A.shape == (nv, nv) and b.c.shape == (nv,)
    assert np.allclose(b.A, 0.05 * np.eye(nv)) and np.allclose(b.c, 0.0)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_reg.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms.reg`).

- [ ] **Step 3 : Écrire `src/solve/terms/reg.py`**

```python
"""reg — step damping ``‖w_reg·δv‖²`` (a well-conditioned QP, bounded step). ``A = w_reg·I(nv)``,
``c = 0``, ``A_obj = None`` (the object has its own anchor, the O term). Posture regularisation toward
a nominal pose (Holosoma ``q_nominal``) is a noted future variant — v1 is plain damping."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig


def build_reg(nv: int, cfg: SolveConfig) -> list[ResidualBlock]:
    """Single ``reg`` block: ``A = w_reg·I(nv)``, ``c = 0(nv)``."""
    return [ResidualBlock(A=cfg.w_reg * np.eye(nv), c=np.zeros(nv), A_obj=None, name="reg")]
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_reg.py -q`
Expected: PASS (1 test).

- [ ] **Step 5 : Commit**

```bash
git add src/solve/terms/reg.py tests/test_solve_terms_reg.py
git commit -m "feat(holov2): solve/terms/reg — build_reg (damping de pas w_reg·I)"
```

---

### Task 7 : `solve/terms/constraints.py` — limites articulaires + trust-region box

**Files:**
- Modify: `src/prepare/contracts.py` (add `joint_pos_limits` to the `RobotModel` Protocol)
- Modify: `src/prepare/load/robot.py` (implement `joint_pos_limits` on `PinRobot`)
- Create: `src/solve/terms/constraints.py`
- Test: `tests/test_solve_terms_constraints.py`

**Interfaces:**
- Consumes: a `RobotModel` exposing `nv`, `dof`, `neutral() -> (nq,)`, `joint_pos_limits() -> (lower (dof,), upper (dof,))`; `SolveConfig`; Plan A `LinearConstraint`, `TrustRegion`.
- Produces: `build_constraints(robot, cfg) -> tuple[list[LinearConstraint], list[TrustRegion]]`.
  - Joint-limit `LinearConstraint`: `A` = selection (dof, nv) picking `δv[6:6+dof]`, `lb = lower − q0`, `ub = upper − q0`, `q0 = neutral()[7:]` (Assumption 2), `A_obj=None`, name `"joint_limits"`.
  - `TrustRegion(var="dv", radius = [tr_base_pos]*3 + [tr_base_rot]*3 + [tr_joints]*dof, norm=-1)`.

- [ ] **Step 1 : Étendre le Protocol `RobotModel` (`src/prepare/contracts.py`)** — ajouter, dans le `class RobotModel(Protocol)`, après `link_jacobians` :

```python
    def joint_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuated joint position limits ``(lower (dof,), upper (dof,))`` (rad), aligned with the
        joint DOFs ``v[6:6+dof]`` / ``q[7:7+dof]``. Used by ``solve`` to box the joint step."""
```

- [ ] **Step 2 : Implémenter sur `PinRobot` (`src/prepare/load/robot.py`)** — ajouter la méthode (pinocchio expose les limites sur le modèle ; les 7 premières entrées `q` = free-flyer, les `dof` suivantes = joints) :

```python
    def joint_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuated joint position limits from the URDF (rad), the joint slice of the free-flyer
        config (q[:7] is the base): ``(lower (dof,), upper (dof,))``."""
        lo = np.asarray(self.model.lowerPositionLimit, np.float64)[7:7 + self.dof]
        hi = np.asarray(self.model.upperPositionLimit, np.float64)[7:7 + self.dof]
        return lo, hi
```

- [ ] **Step 3 : Vérifier l'import prepare (pas de régression de contrat)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.prepare.contracts; print('prepare contracts import ok')"`
Expected: `prepare contracts import ok`

- [ ] **Step 4 : Écrire le test (robot STUB, pinocchio-free)**

```python
# tests/test_solve_terms_constraints.py
"""build_constraints : box limites articulaires sur δv[6:] (linéarisée au neutre) + TrustRegion box
par-DOF depuis la config. Robot STUB (pas de pinocchio)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.constraints import build_constraints


class _StubRobot:
    nv = 9          # 6 (free-flyer) + 3 joints
    dof = 3
    def neutral(self):
        # q = [pos(3), quat xyzw(4), joints(3)] -> joints at indices 7:10
        return np.array([0, 0, 0, 0, 0, 0, 1, 0.1, 0.0, -0.1], float)
    def joint_pos_limits(self):
        return np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])


def test_joint_limits_constraint():
    robot = _StubRobot()
    cfg = SolveConfig()
    cons, _ = build_constraints(robot, cfg)
    jl = [c for c in cons if c.name == "joint_limits"][0]
    assert jl.A.shape == (3, 9) and jl.A_obj is None
    # selection picks v[6:9]
    S = np.zeros((3, 9)); S[0, 6] = S[1, 7] = S[2, 8] = 1.0
    assert np.allclose(jl.A, S)
    q0 = robot.neutral()[7:]
    assert np.allclose(jl.lb, np.array([-1.0, -2.0, -3.0]) - q0)
    assert np.allclose(jl.ub, np.array([1.0, 2.0, 3.0]) - q0)


def test_trust_region_box():
    robot = _StubRobot()
    cfg = SolveConfig(tr_base_pos=0.05, tr_base_rot=0.1, tr_joints=0.2)
    _, trs = build_constraints(robot, cfg)
    assert len(trs) == 1
    tr = trs[0]
    assert tr.var == "dv" and tr.norm == -1 and tr.radius.shape == (9,)
    assert np.allclose(tr.radius[:3], 0.05)
    assert np.allclose(tr.radius[3:6], 0.1)
    assert np.allclose(tr.radius[6:9], 0.2)
```

- [ ] **Step 5 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_constraints.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.terms.constraints`).

- [ ] **Step 6 : Écrire `src/solve/terms/constraints.py`**

```python
"""Joint limits + box trust region — the LINEAR/box side of the subproblem (the residuals are the
quadratic side). Joint limits: ``lower ≤ q_joint + δv_joint ≤ upper`` -> a box on the joint DOFs of
``δv`` (the selection ``v[6:6+dof]``). ``build_constraints(robot, cfg)`` has no live ``q``, so v1
linearises at the NEUTRAL joints ``q0 = robot.neutral()[7:]`` (EXACT at the cold-start frame; an
approximate static box afterwards — Plan C should re-base on the live ``q`` each SQP iterate; see plan
Assumption 2). Trust region: per-DOF box radii from the config (heterogeneous units handled per-DOF)."""
from __future__ import annotations

import numpy as np

from ..contracts import LinearConstraint, TrustRegion
from ..config import SolveConfig


def build_constraints(robot, cfg: SolveConfig) -> tuple[list[LinearConstraint], list[TrustRegion]]:
    """Box joint-limit ``LinearConstraint`` on ``δv[6:6+dof]`` (linearised at neutral) + the per-DOF box
    ``TrustRegion`` for ``δv``. The object trust region (``δξ``) is added by Plan C's assemble when
    ``n_obj > 0`` (it needs ``n_obj``, not available here)."""
    nv, dof = robot.nv, robot.dof
    lower, upper = robot.joint_pos_limits()
    q0 = np.asarray(robot.neutral(), np.float64)[7:7 + dof]      # neutral joint angles (Assumption 2)

    S = np.zeros((dof, nv), np.float64)                          # select δv joint DOFs (v[6:6+dof])
    S[np.arange(dof), 6 + np.arange(dof)] = 1.0
    joint_limits = LinearConstraint(A=S, lb=np.asarray(lower, np.float64) - q0,
                                    ub=np.asarray(upper, np.float64) - q0,
                                    A_obj=None, name="joint_limits")

    radius = np.concatenate([np.full(3, cfg.tr_base_pos), np.full(3, cfg.tr_base_rot),
                             np.full(dof, cfg.tr_joints)])       # (nv,) per-DOF box radius
    trust = TrustRegion(var="dv", radius=radius, norm=-1)
    return [joint_limits], [trust]
```

> **Note (Assumption 2)** : `build_constraints` ne fabrique PAS le trust-region objet (`var="dxi"`) car il dépend de `n_obj`, inconnu ici ; Plan C l'ajoute dans `assemble` à partir de `cfg.tr_object_pos`/`cfg.tr_object_rot` (radius `[tr_object_pos]*3 + [tr_object_rot]*3` répété N fois).

- [ ] **Step 7 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_terms_constraints.py -q`
Expected: PASS (2 tests).

- [ ] **Step 8 : Commit**

```bash
git add src/prepare/contracts.py src/prepare/load/robot.py src/solve/terms/constraints.py tests/test_solve_terms_constraints.py
git commit -m "feat(holov2): solve/terms/constraints — limites articulaires (box sur δv joints) + trust-region box ; RobotModel.joint_pos_limits"
```

---

### Task 8 : garde-fou import — `solve` reste léger, `terms` interne

**Files:**
- Verify: `src/solve/__init__.py` (NE PAS exporter `terms`)
- Test: `tests/test_solve_import_light.py`

**Interfaces:**
- Consumes: the existing `solve/__init__` (Plan A) re-exporting only `contracts` + `backend`.
- Produces: a guard test asserting `import src.solve` pulls no cvxpy/torch/pinocchio, and that `terms` is reachable as a submodule but NOT a `solve` package attribute.

- [ ] **Step 1 : Vérifier `src/solve/__init__.py`** — confirmer qu'il n'importe PAS `terms` (Plan A le laisse à `contracts` + `backend`). Si une ligne `from . import terms` ou `from .terms import …` existe, la SUPPRIMER. Aucun autre changement.

Run: `grep -n "terms" src/solve/__init__.py || echo "OK: solve/__init__ does not import terms"`
Expected: `OK: solve/__init__ does not import terms`

- [ ] **Step 2 : Écrire le test**

```python
# tests/test_solve_import_light.py
"""Garde-fou : import src.solve reste cvxpy/torch/pinocchio-free ; terms est interne (chargé à la
demande par assemble en Plan C), pas un attribut du package solve. Les builders restent importables
explicitement et n'amènent pas torch (geo_value_grad est numpy-only)."""
import sys


def test_solve_import_is_light():
    for m in ("cvxpy", "torch", "pinocchio"):
        sys.modules.pop(m, None)
    import src.solve  # noqa: F401
    assert "cvxpy" not in sys.modules, "cvxpy leaked at import src.solve"
    assert "torch" not in sys.modules, "torch leaked at import src.solve"
    assert "pinocchio" not in sys.modules, "pinocchio leaked at import src.solve"
    assert not hasattr(src.solve, "terms"), "terms must stay internal (not exported by solve/__init__)"


def test_terms_importable_and_torch_free():
    import src.solve.terms.style       # noqa: F401
    import src.solve.terms.contact     # noqa: F401  (pulls geo_value_grad — numpy-only)
    import src.solve.terms.object      # noqa: F401
    import src.solve.terms.reg         # noqa: F401
    import src.solve.terms.constraints  # noqa: F401
    assert "torch" not in sys.modules and "cvxpy" not in sys.modules
```

- [ ] **Step 3 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_import_light.py -q`
Expected: PASS (2 tests).

> Si `test_terms_importable_and_torch_free` échoue sur `torch`/`cvxpy`, la fuite vient d'un import de package (`from ...targets import …` au lieu du sous-module `from ...targets.interaction.geodesic import geo_value_grad`). Corriger l'import fautif vers le sous-module précis (cf. règle d'or #1 : sortie publique, jamais un interne lourd).

- [ ] **Step 4 : Lancer toute la suite Plan B**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_config.py tests/test_solve_terms_ops.py tests/test_solve_terms_style.py tests/test_solve_terms_contact.py tests/test_solve_terms_object.py tests/test_solve_terms_reg.py tests/test_solve_terms_constraints.py tests/test_solve_import_light.py -q`
Expected: PASS (all).

- [ ] **Step 5 : Commit**

```bash
git add tests/test_solve_import_light.py
git commit -m "test(holov2): solve — garde-fou import léger (cvxpy/torch/pinocchio-free, terms interne)"
```

---

## Self-Review

**1. Spec coverage** (vs `2026-06-30-solve-stage-design.md`, Plan B perimeter):

| Spec item | Task |
|---|---|
| `config.py` — weights `w_pos…w_reg`, contact `α` (gate + d_ref), trust radii par-DOF, `n_iter_first/per_frame`, `step_tol`, `backend` | Task 1 ✅ |
| `_ops.world_normal(R_i, n_local)` | Task 2 ✅ |
| `_ops.dist_jac(n_world, point_jac) = ∂d/∂δv` | Task 2 ✅ |
| `_ops.geo_chain(∇geo, point_jac) = ∂geo/∂δv` | Task 2 ✅ |
| `_ops.so3_log(R_ref, R_cur)` | Task 2 ✅ |
| object-tangent helpers (`se3_log_world`, `scatter_obj`, `GeoField`, `quat_to_rot`) | Task 2 ✅ |
| `style.build_style` — S-pos, S-rot | Task 3 ✅ |
| `contact.build_contact` — C-D, C-X (géodésique, `dist_jac`/`geo_chain`/`world_normal`, active only, α folded) | Task 4 ✅ |
| `object.build_object` — CO-D, O (`cloud_jac_self`/`probe_jac_obj`, A=0, A_obj only, O via `se3_log_world`) | Task 5 ✅ (**CO-X DEFERRED** — Assumption 6: no `geo` in signature) ⚠️ |
| `reg.build_reg` — `A = w_reg·I`, `c=0` | Task 6 ✅ |
| `constraints.build_constraints` — joint limits (box on δv joints) + box `TrustRegion` per-DOF | Task 7 ✅ |
| `solve` import cvxpy/torch/pinocchio-free, `terms` internal | Task 8 ✅ |
| Weights folded into `A`/`c` (no separate weight) | All builders ✅ |
| FD tests per op + linearisation-vs-FD per builder | Tasks 2–7 ✅ |

Gap flagged: **CO-X** geodesic object self-contact is deferred (the canonical `build_object(contact_eval, env_refs, object_rot, object_pos, cfg)` has no `geo` argument to carry the per-channel geodesic tables; `cfg.w_cox` is reserved). It is structurally identical to C-X once Plan C threads a `geo` bundle into `build_object`.

**2. Placeholder scan:** No `TBD`/`TODO`/"add error handling"/"similar to". Every code step carries complete numpy `A`/`c`/`A_obj` assembly and full FD/unit tests. ✅

**3. Type consistency:**
- Builder signatures match the canonical block verbatim: `build_style(style_eval, style_targets, cfg)`, `build_contact(contact_eval, robot_field_ref, geo, cfg)`, `build_object(contact_eval, env_refs, object_rot, object_pos, cfg)`, `build_reg(nv, cfg)`, `build_constraints(robot, cfg)`. ✅
- All return Plan A contracts: `ResidualBlock(A (m,nv), c (m,), A_obj (m,n_obj*6)|None, name)`, `LinearConstraint(A, lb, ub, A_obj, name)`, `TrustRegion(var, radius, norm)` — consistent with `solve/contracts.py` (Plan A, not re-planned here). ✅
- `_ops` names used by builders match definitions: `world_normal`, `dist_jac`, `geo_chain`, `so3_log`, `se3_log_world`, `quat_to_rot`, `scatter_obj`, `GeoField`. ✅
- `targets` types consumed match `src/targets/contracts.py` field names/shapes exactly (`StyleEval.jac_pos/jac_rot`, `ContactEval.point_jac/probe_jac_obj/env`, `ContactEnvEval.cloud_jac_self/probe_jac_obj`, `MultiChannelField.distance/direction/witness/active/channels`, `RobotInteractionTargets.field`, `EnvironmentInteractionTargets.per_object`); `geo_value_grad`/`nearest_index` signatures match `src/targets/interaction/geodesic.py`. ✅
- `SolveConfig` field names used identically across builders (`w_pos`, `w_rot`, `w_cd`, `w_cx`, `w_cod`, `w_obj`, `w_reg`, `contact_d_ref_scale`, `tr_base_pos/rot`, `tr_joints`). `w_cox` reserved (CO-X deferred). ✅
- `RobotModel.joint_pos_limits()` added in both the Protocol (Task 7 Step 1) and `PinRobot` (Step 2); consumed by `build_constraints` (Step 6) and the test stub (Step 4) with the same `(lower (dof,), upper (dof,))` shape. ✅

**Open assumptions surfaced (for Plan C / review):** (1) geodesic gradient sign/frame — same `value` feeds `c` and gradient; object-local→world via `world_normal(R_i, ·)`; loose FD tol ~2e-2; (2) joint-limit DOFs `v[6+j] ↔ q[7+j]`, v1 linearised at neutral q0 (live-q rebasing DEFERRED to a future increment; the per-DOF joint trust region bounds the step), needs `joint_pos_limits()`; (3) object-channel `R_i` threaded via the `GeoField` `geo` bundle in `build_contact`; (4) channel 0 = ground / `c≥1` → object `c−1`; (5) active mask = reference field; (6) CO-X deferred + O `c=0` at the linearisation point (general `se3_log_world` ready for a Plan-C live current pose).
