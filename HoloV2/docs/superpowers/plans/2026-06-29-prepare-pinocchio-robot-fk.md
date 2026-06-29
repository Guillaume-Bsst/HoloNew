# prepare — Modèle robot pinocchio (free-flyer) + jacobiennes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le moteur FK robot (yourdfpy, base fixe, sans jacobienne) par un modèle **pinocchio free-flyer** qui fournit les transforms monde ET les jacobiennes géométriques analytiques, comme fondation cinématique de l'évaluateur `targets` (Plan 2).

**Architecture:** `prepare/load/robot.py` expose `PinRobot` (impl du protocol `RobotModel`) bâti sur `pin.buildModelFromUrdf(..., JointModelFreeFlyer())`. Configuration `q = [pelvis(7: pos + quat xyzw), angles_joints]` (convention pinocchio), tangente `v` de dim `nv = 6 + n_joints`. Les jacobiennes sont lues en `LOCAL_WORLD_ALIGNED` (axes monde). Porté de V1 `HoloNew/src/test_socp/pin_model.py`. La sampling de surface pour la correspondance (`point_cloud/correspondence/robot_surface.py`) est indépendante de `RobotModel` (échantillonnage mesh propre) et n'est PAS touchée.

**Tech Stack:** Python, numpy, pinocchio 4.0.0 (déjà dans l'env `holonew`), pytest.

## Global Constraints

- `prepare/contracts.py` reste **numpy-only à l'import** : `RobotModel` est un `Protocol` (pas d'import pinocchio dans `contracts.py`). pinocchio est importé **uniquement** dans `prepare/load/robot.py`.
- Compute en **float64**.
- Imports **relatifs** dans `src/`, **absolus** (`from src.…`) dans `tests/`.
- Quaternions **wxyz** dans les contrats du projet ; **mais** la config pinocchio `q` utilise le quaternion **xyzw** (convention pinocchio) — documenté, interne à `PinRobot`.
- Invariants de contrat → `raise ValueError` explicite.
- Tests dans `HoloV2/tests/`, lancés depuis `HoloV2/` avec le python de l'env :
  `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<f> -q`.
- Commits **conventionnels**, en français. **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic). Auteur : `Guillaume-Bsst`.
- Le type/knob ne se duplique jamais ; `RobotModel` reste l'unique surface cinématique.

---

### Task 1 : Étendre le protocol `RobotModel`

**Files:**
- Modify: `src/prepare/contracts.py` (le bloc `class RobotModel(Protocol)`, ~lignes 57-70)
- Test: `tests/test_robot_protocol.py` (create)

**Interfaces:**
- Produces: protocol `RobotModel` avec, en plus de l'existant (`link_names`, `dof`, `link_transforms`, `rest_transforms`) : attributs `nq: int`, `nv: int` ; méthodes `neutral() -> np.ndarray` (`(nq,)`), `integrate(q, v) -> np.ndarray` (`(nq,)`), `link_jacobians(q) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` = `(rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))` en repère monde.

- [ ] **Step 1: Écrire le test (structurel, runtime_checkable)**

```python
# tests/test_robot_protocol.py
"""Le protocol RobotModel expose la surface cinématique étendue (free-flyer + jacobiennes)."""
import numpy as np

from src.prepare.contracts import RobotModel


class _Dummy:
    link_names = ("a",)
    dof = 1
    nq = 8
    nv = 7

    def link_transforms(self, q): return np.zeros((1, 3, 3)), np.zeros((1, 3))
    def rest_transforms(self): return np.zeros((1, 3, 3)), np.zeros((1, 3))
    def neutral(self): return np.zeros(self.nq)
    def integrate(self, q, v): return np.zeros(self.nq)
    def link_jacobians(self, q):
        return (np.zeros((1, 3, 3)), np.zeros((1, 3)),
                np.zeros((1, 3, self.nv)), np.zeros((1, 3, self.nv)))


def test_dummy_satisfies_protocol():
    assert isinstance(_Dummy(), RobotModel)   # runtime_checkable structural check
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_protocol.py -q`
Expected: FAIL (le protocol n'a pas encore `nq`/`nv`/`neutral`/`integrate`/`link_jacobians` ; `isinstance` échoue sur les membres manquants).

- [ ] **Step 3: Étendre le protocol**

Dans `src/prepare/contracts.py`, remplacer le corps de `class RobotModel(Protocol)` par :

```python
@runtime_checkable
class RobotModel(Protocol):
    """Robot kinematics. Rest transforms (q-independent) are used by ``prepare`` to sample the G1
    surface / build the correspondence; full FK + Jacobians (q-dependent) are used by ``solve`` via the
    ``targets`` evaluator. Configuration ``q`` is a pinocchio FREE-FLYER vector
    ``[pelvis(7: pos + quat xyzw), joints]`` of length ``nq``; the tangent ``v`` has dim
    ``nv = 6 + n_joints``. Concrete impl in ``prepare/load/robot.py`` (pinocchio)."""

    link_names: tuple[str, ...]
    dof: int          # actuated joints (= nv - 6)
    nq: int           # configuration dim (free-flyer)
    nv: int           # tangent dim (= 6 + dof)

    def link_transforms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L,3,3) rotations, (L,3) positions: WORLD transform of each link for ``q`` (free-flyer)."""

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Link transforms at the neutral configuration (base identity, joints 0)."""

    def neutral(self) -> np.ndarray:
        """Neutral configuration ``(nq,)`` (base identity with unit quaternion, joints 0)."""

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Manifold step ``q ⊕ v`` ``(nq,)`` (keeps the base quaternion unit)."""

    def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For ``q``: (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv)) in the WORLD frame,
        aligned with ``link_names``. ``jac_lin``/``jac_ang`` are the LOCAL_WORLD_ALIGNED translational
        / angular frame Jacobians: ``dp_world = jac_lin @ v``, ``omega_world = jac_ang @ v``."""
```

Vérifier que `@runtime_checkable` et `Protocol` sont déjà importés en tête de `contracts.py` (ils le sont — `RobotModel` est déjà `@runtime_checkable`).

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_protocol.py -q`
Expected: PASS

- [ ] **Step 5: Vérifier l'import numpy-only (pas de pinocchio dans contracts)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.prepare.contracts; assert 'pinocchio' not in sys.modules; print('ok numpy-only')"`
Expected: `ok numpy-only`

- [ ] **Step 6: Commit**

```bash
git add src/prepare/contracts.py tests/test_robot_protocol.py
git commit -m "feat(holov2): RobotModel protocol — free-flyer (nq/nv/neutral/integrate) + link_jacobians"
```

---

### Task 2 : `PinRobot` — modèle + `link_transforms` + `neutral` + `rest_transforms`

**Files:**
- Modify: `src/prepare/load/robot.py` (remplace `UrdfRobot` par `PinRobot` ; `build_robot_model` renvoie `PinRobot` ; garde `CORRESPONDENCE_REST_POSE` / `correspondence_rest_angles`)
- Test: `tests/test_robot_fk.py` (modify — porte `test_urdf_robot_fk` en free-flyer)

**Interfaces:**
- Consumes: `RobotSpec` (`urdf_path`, `name`), protocol `RobotModel` (Task 1).
- Produces: `PinRobot(spec)` avec `link_names`, `dof`, `nq`, `nv`, `link_transforms(q)`, `rest_transforms()`, `neutral()`, `integrate(q, v)` ; `build_robot_model(spec) -> PinRobot`.

- [ ] **Step 1: Écrire le test (parité pinocchio vs yourdfpy, base à l'identité)**

Remplacer `tests/test_robot_fk.py::test_urdf_robot_fk` par les deux tests ci-dessous (garder `test_correspondence_rest_angles_is_robot_keyed` inchangé) :

```python
@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_pin_robot_fk_shapes_and_neutral():
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    assert robot.dof == 29
    assert robot.nv == 6 + 29 and robot.nq == 7 + 29
    assert "pelvis" in robot.link_names
    n = len(robot.link_names)

    q0 = robot.neutral()
    assert q0.shape == (robot.nq,)
    assert np.isclose(np.linalg.norm(q0[3:7]), 1.0)                 # unit base quaternion

    rot, pos = robot.rest_transforms()
    assert rot.shape == (n, 3, 3) and pos.shape == (n, 3)
    assert np.allclose(rot[0] @ rot[0].T, np.eye(3), atol=1e-6)     # orthonormal

    # bend a joint -> some link relocates
    q = q0.copy()
    q[7] += 0.8
    _, pos1 = robot.link_transforms(q)
    assert not np.allclose(pos, pos1)


@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_pin_fk_parity_vs_yourdfpy_base_relative():
    # At the neutral free-flyer base (identity), pinocchio WORLD transforms == yourdfpy base-relative
    # transforms (same URDF kinematics). Compare a few links at a random actuated config.
    import yourdfpy
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    urdf = yourdfpy.URDF.load(str(_URDF), load_meshes=False, build_scene_graph=True)

    rng = np.random.default_rng(0)
    angles = rng.uniform(-0.3, 0.3, size=29)
    cfg = {name: float(a) for name, a in zip(urdf.actuated_joint_names, angles)}
    urdf.update_cfg(np.array([cfg[n] for n in urdf.actuated_joint_names]))

    q = robot.neutral()
    # set actuated joints in pinocchio order via the public mapping (Task 2 helper)
    q = robot.config_from_angles(cfg)
    rot, pos = robot.link_transforms(q)

    for name in ("left_elbow_link", "right_wrist_yaw_link", "left_knee_link"):
        if name not in robot.link_names:
            continue
        i = robot.link_names.index(name)
        T = np.asarray(urdf.get_transform(name))                    # base-relative (base at origin)
        assert np.allclose(pos[i], T[:3, 3], atol=1e-5), name
        assert np.allclose(rot[i], T[:3, :3], atol=1e-5), name
```

(Le test utilise `robot.config_from_angles`, ajouté ci-dessous : il fixe les angles articulaires nommés dans un `q` neutre.)

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_fk.py -q`
Expected: FAIL (`PinRobot`/`config_from_angles`/`neutral` absents ; `build_robot_model` renvoie encore yourdfpy fixed-base).

- [ ] **Step 3: Implémenter `PinRobot` (remplacer `UrdfRobot`)**

Remplacer la classe `UrdfRobot` ET `build_robot_model` dans `src/prepare/load/robot.py` par (garder le haut du fichier : docstring, `CORRESPONDENCE_REST_POSE`, `correspondence_rest_angles`) :

```python
class PinRobot:
    """``RobotModel`` backed by pinocchio (free-flyer). World link transforms + analytic frame
    Jacobians (LOCAL_WORLD_ALIGNED). Config ``q = [pelvis(7: pos + quat xyzw), joints]`` (pinocchio
    order); tangent ``v`` dim ``nv = 6 + n_joints``. Ported from HoloNew ``test_socp/pin_model.py``."""

    def __init__(self, spec: RobotSpec) -> None:
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(spec.urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.nq: int = int(self.model.nq)
        self.nv: int = int(self.model.nv)
        self.dof: int = self.nv - 6
        # BODY frames = URDF links; keep their names + ids (transport/remap index by NAME).
        self.link_names: tuple[str, ...] = tuple(
            f.name for f in self.model.frames if f.type == pin.FrameType.BODY)
        self._fids = {name: self.model.getFrameId(name) for name in self.link_names}
        # actuated joint name -> idx_q / idx_v (joints 2..njoints; joint 1 is the free-flyer)
        self._joint_qadr = {self.model.names[j]: self.model.joints[j].idx_q
                            for j in range(2, self.model.njoints)}

    def neutral(self) -> np.ndarray:
        return np.asarray(self._pin.neutral(self.model), np.float64)

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(self._pin.integrate(self.model, np.asarray(q, np.float64),
                                              np.asarray(v, np.float64)), np.float64)

    def config_from_angles(self, angles: dict) -> np.ndarray:
        """Neutral base + named actuated joint angles -> q (nq,). Joints absent default to 0."""
        q = self.neutral()
        for name, a in angles.items():
            if name in self._joint_qadr:
                q[self._joint_qadr[name]] = float(a)
        return q

    def _fk(self, q: np.ndarray) -> None:
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, pin.normalize(self.model, np.asarray(q, np.float64)))
        pin.updateFramePlacements(self.model, self.data)

    def link_transforms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._fk(q)
        n = len(self.link_names)
        rot = np.empty((n, 3, 3)); pos = np.empty((n, 3))
        for i, name in enumerate(self.link_names):
            oMf = self.data.oMf[self._fids[name]]
            rot[i] = np.asarray(oMf.rotation); pos[i] = np.asarray(oMf.translation)
        return rot, pos

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        return self.link_transforms(self.neutral())


def build_robot_model(spec: RobotSpec) -> PinRobot:
    """Build the pinocchio ``RobotModel`` for ``spec`` (FK + Jacobians, no meshes)."""
    return PinRobot(spec)
```

Supprimer l'import `yourdfpy` du module et toute référence résiduelle à `UrdfRobot` (`grep -n "UrdfRobot\|yourdfpy" src/prepare/load/robot.py` doit être vide).

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_fk.py -q`
Expected: PASS (3 tests : rest-angles, shapes/neutral, parité).

- [ ] **Step 5: Vérifier qu'aucun `UrdfRobot` n'est importé ailleurs**

Run: `grep -rn "UrdfRobot" src/ tests/`
Expected: aucune sortie (sinon, mettre à jour l'import vers `PinRobot`/`build_robot_model`).

- [ ] **Step 6: Commit**

```bash
git add src/prepare/load/robot.py tests/test_robot_fk.py
git commit -m "feat(holov2): PinRobot — FK free-flyer pinocchio (remplace yourdfpy), parité vs yourdfpy testée"
```

---

### Task 3 : `PinRobot.link_jacobians` — jacobiennes géométriques (différences finies)

**Files:**
- Modify: `src/prepare/load/robot.py` (ajoute `link_jacobians`)
- Test: `tests/test_robot_fk.py` (ajoute un test FD)

**Interfaces:**
- Consumes: `PinRobot` (Task 2), `pin.getFrameJacobian(..., LOCAL_WORLD_ALIGNED)`.
- Produces: `PinRobot.link_jacobians(q) -> (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))`.

- [ ] **Step 1: Écrire le test FD**

Ajouter à `tests/test_robot_fk.py` :

```python
@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_link_jacobians_match_finite_differences():
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))   # random valid config

    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)
    n = len(robot.link_names)
    assert jac_lin.shape == (n, 3, robot.nv) and jac_ang.shape == (n, 3, robot.nv)
    assert np.allclose(pos, robot.link_transforms(q)[1], atol=1e-9)   # transforms agree with FK

    # finite-difference the WORLD position of a few links along each tangent direction.
    eps = 1e-6
    test_links = [robot.link_names.index(n) for n in ("left_elbow_link", "pelvis")
                  if n in robot.link_names]
    for k in range(robot.nv):
        v = np.zeros(robot.nv); v[k] = eps
        pos_p = robot.link_transforms(robot.integrate(q, v))[1]
        pos_m = robot.link_transforms(robot.integrate(q, -v))[1]
        fd = (pos_p - pos_m) / (2 * eps)                             # (L, 3) d pos / d v_k
        for i in test_links:
            assert np.allclose(jac_lin[i, :, k], fd[i], atol=1e-4), (robot.link_names[i], k)
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_fk.py::test_link_jacobians_match_finite_differences -q`
Expected: FAIL (`link_jacobians` n'existe pas).

- [ ] **Step 3: Implémenter `link_jacobians`**

Ajouter à `PinRobot` (dans `src/prepare/load/robot.py`) :

```python
    def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World transforms + LOCAL_WORLD_ALIGNED translational/angular frame Jacobians per link.
        ``dp_world = jac_lin @ v``, ``omega_world = jac_ang @ v`` (v in pinocchio tangent order)."""
        pin = self._pin
        qn = pin.normalize(self.model, np.asarray(q, np.float64))
        pin.computeJointJacobians(self.model, self.data, qn)
        pin.updateFramePlacements(self.model, self.data)
        n = len(self.link_names); nv = self.nv
        rot = np.empty((n, 3, 3)); pos = np.empty((n, 3))
        jac_lin = np.empty((n, 3, nv)); jac_ang = np.empty((n, 3, nv))
        for i, name in enumerate(self.link_names):
            fid = self._fids[name]
            oMf = self.data.oMf[fid]
            rot[i] = np.asarray(oMf.rotation); pos[i] = np.asarray(oMf.translation)
            J6 = np.asarray(pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED))
            jac_lin[i] = J6[0:3, :]; jac_ang[i] = J6[3:6, :]
        return rot, pos, jac_lin, jac_ang
```

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_fk.py -q`
Expected: PASS (tous les tests robot, dont le FD).

- [ ] **Step 5: Vérifier que `PinRobot` satisfait le protocol**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "from pathlib import Path; from src.prepare.contracts import RobotModel, RobotSpec; from src.prepare.load.robot import build_robot_model; u=Path('models/g1/g1_29dof.urdf'); r=build_robot_model(RobotSpec(name='g1',urdf_path=u,link_names=(),dof=29,height=1.3)); print('isinstance', isinstance(r, RobotModel))"`
Expected: `isinstance True`

- [ ] **Step 6: Commit**

```bash
git add src/prepare/load/robot.py tests/test_robot_fk.py
git commit -m "feat(holov2): PinRobot.link_jacobians — jacobiennes géométriques LWA, validées par différences finies"
```

---

### Task 4 : Intégration — câbler le swap et verdir les consommateurs existants

**Files:**
- Modify: `tests/test_solve_field_eval.py:54` (la config `q` passe en free-flyer)
- Verify: `src/prepare/runner.py` (construit `build_robot_model` → `PinRobot` ; rien à changer si l'API protocol tient)
- Verify: `tests/test_robot_cloud.py` (transforms synthétiques — doit rester vert sans changement)

**Interfaces:**
- Consumes: `PinRobot` (Tasks 2-3) via `build_robot_model` dans `runner.py`.
- Produces: arbre de tests vert avec le moteur pinocchio.

- [ ] **Step 1: Mettre à jour la config `q` du test du package value-seule**

Dans `tests/test_solve_field_eval.py`, remplacer :

```python
    q = np.zeros(ctx.robot.dof)
```

par :

```python
    q = ctx.robot.neutral()                                # free-flyer neutral config (nq,)
```

(le reste du test — `pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))` puis `eval_fields` — est inchangé : il valide toujours le chemin value-seule conservé.)

- [ ] **Step 2: Vérifier `test_robot_cloud` (aucun changement attendu)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_cloud.py -q`
Expected: PASS (3 tests ; il utilise des transforms synthétiques, indépendant du moteur FK).

- [ ] **Step 3: Vérifier que le runner construit bien `PinRobot`**

Run: `grep -n "build_robot_model\|robot.link_names" src/prepare/runner.py`
Expected: `build_robot_model(spec.robot)` présent (ligne ~211) ; `robot_point_cloud(corr_table, robot.link_names)` (ligne ~212) — le remap par nom absorbe l'ordre des frames pinocchio. Aucun changement de code nécessaire.

- [ ] **Step 4: Lancer la suite robot/cloud + le test value-seule (skip si données absentes)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_robot_protocol.py tests/test_robot_fk.py tests/test_robot_cloud.py tests/test_solve_field_eval.py -q`
Expected: PASS ou SKIP (les tests gardés par données HODome/SMPL-X absentes peuvent SKIP ; aucun FAIL).

- [ ] **Step 5: Smoke d'import global de `prepare` (pas de régression d'import)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.prepare.runner, src.prepare.load.robot, src.prepare.contracts; print('prepare imports ok')"`
Expected: `prepare imports ok`

- [ ] **Step 6: Commit**

```bash
git add tests/test_solve_field_eval.py
git commit -m "test(holov2): config q free-flyer (neutral) pour le swap pinocchio ; suite robot verte"
```

---

## Self-Review

**1. Spec coverage** (vs `2026-06-29-targets-evaluator-seam-design.md`, section « Changements prepare ») :
- `RobotModel.link_jacobians(q)` free-flyer → Task 1 (protocol) + Task 3 (impl). ✅
- Impl pinocchio dans `prepare/load/robot.py`, portée de V1 `pin_model.py` → Task 2-3. ✅
- `link_transforms(q)` conservé (posage déjà câblé) → Task 2. ✅
- Parité `link_transforms` pinocchio vs yourdfpy → Task 2 Step 1. ✅
- Jacobienne par différences finies (le test-clé) → Task 3 Step 1. ✅
- Robot instancié uniquement dans `prepare/` ; `contracts` numpy-only → Task 1 Step 5 (assert), Task 2 (pinocchio importé dans la classe). ✅
- (Hors périmètre Plan 1, → Plan 2 : `StyleEval`/`ContactEval`, `style/eval`, `interaction/eval`, `evaluator.py`, réorg `targets/`, surface publique.)

**2. Placeholder scan** : aucun `TBD`/`TODO` ; chaque step porte le code/commande réel. ✅

**3. Type consistency** : `link_jacobians` renvoie partout le 4-uple `(rot, pos, jac_lin, jac_ang)` ; `neutral()`→`(nq,)`, `integrate(q,v)`→`(nq,)`, `config_from_angles(dict)`→`(nq,)` cohérents entre protocol (Task 1), impl (Task 2-3) et tests. `nv = 6 + dof`, `nq = 7 + dof`. ✅

> **Note** : `config_from_angles` est utilisé par le test de parité (Task 2) mais n'est PAS dans le protocol (helper concret de `PinRobot`). C'est volontaire : la construction de `q` depuis des angles nommés est un confort de test/prepare, pas une obligation de la surface `RobotModel`. Si Plan 2 en a besoin sur le protocol, l'y ajouter à ce moment.
