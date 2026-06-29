# targets — réorg + types d'éval + Evaluator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réorganiser `targets/` autour de la symétrie référence/évaluation, ajouter les types d'évaluation q-dépendants (`StyleEval`, `ContactEval`, `ContactEnvEval`) et l'objet `Evaluator(ctx, robot_name)` qui transforme une config `(q, object_poses)` en (état géométrique courant + jacobiennes analytiques) — le seam `targets → solve`.

**Architecture:** `targets` fait UNE opération conceptuelle (poser une config, lire son état style + contact) appliquée à DEUX configs : la RÉFÉRENCE (humain/démo, q-indép., déjà en place — `pipeline.py`) et l'ÉVALUATION (robot @ `q` + objets @ `SE(3)`, q-dép., NOUVEAU — `evaluator.py`). Les mêmes noyaux purs (`pose_cloud`, `eval_fields`) servent les deux ; seul le côté cible porte une jacobienne. L'évaluateur est construit UNE fois à partir des assets statiques de `InteractionContext` (`robot_cloud`, `channels`, `object_clouds`, `margin`, `robot`) et appelle `ctx.robot.link_jacobians(q)` (Plan 1, LOCAL_WORLD_ALIGNED, repère monde) à chaque itération.

**Tech Stack:** Python, numpy (float64), scipy (rotations dans les tests), pinocchio (caché derrière `RobotModel`, jamais importé par `targets`), pytest.

## Global Constraints

- **Dépendances à sens unique, zéro cycle** : `targets` importe la sortie publique de `prepare` (`from ..prepare.contracts import ...`), jamais `solve`. `solve` n'importera que la surface publique de `targets` (`targets/__init__`).
- **Pas de noyau partagé** : les sous-modules `style/`, `interaction/` restent internes ; seul `evaluator.py` les réunit ; `solve` n'importe que le package.
- **`contracts.py` / `config.py` numpy-only à l'import** : torch/pinocchio JAMAIS dans `targets` (la cinématique lourde est derrière le protocol `RobotModel`).
- **Cœur pur** : les ops de calcul ne font ni I/O, ni log, ni mutation de leurs inputs ; elles rendent un artefact `frozen`. `prof.span` dans les orchestrateurs uniquement.
- **Plan 1 est ACQUIS (LOCKED)** : `RobotModel` expose `link_names`, `dof`, `nq`, `nv`, `neutral()`, `integrate(q, v)`, `link_transforms(q) -> (rot (L,3,3), pos (L,3))`, et `link_jacobians(q) -> (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))` en repère MONDE (LOCAL_WORLD_ALIGNED), aligné sur `link_names`. `nv = 6 + dof`. NE PAS replanifier Plan 1.
- **Compute en float64** ; arrays annotés `np.ndarray` + **forme en commentaire** (`# (L, 3, nv)`).
- **Erreurs** : valider les invariants de contrat par un `raise ValueError` explicite au `__post_init__` (cf. `MultiChannelField`).
- **Quaternions wxyz** ; poses `(x, y, z, qw, qx, qy, qz)` ; tangente objet **world-aligned** `(δt, δθ)` (cohérente LOCAL_WORLD_ALIGNED).
- **Imports** : relatifs DANS `src/` ; absolus (`from src.…`) dans `tests/`. Tests dans **`HoloV2/tests/`** (PAS le `tests/` racine).
- Tests lancés depuis `HoloV2/` avec : `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<f> -q`. `max_frames` bas ; la plupart des tests d'éval sont gardés par la présence du G1 URDF (`models/g1/g1_29dof.urdf`) — rapides (clouds synthétiques) ou par les données HODome/SMPL-X/corr (intégration).
- Commits **conventionnels, en français**. **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic). Auteur : `Guillaume-Bsst`.

---

### Task 1 : RÉORG — renommer `eval.py`→`fields.py` (noyau) et `targets.py`→`refs.py` (réfs)

Renommage mécanique, **préservant le comportement** : libère le nom `interaction/eval.py` (réutilisé en Task 4 pour `contact_eval`) et nomme les références `refs.py` (symétrie avec l'éval). Aucun changement de logique ; seuls les imports bougent. Les imports niveau-package (`from .interaction import ...`) dans `pipeline.py` / `targets/__init__.py` ne bougent PAS (ils passent par `interaction/__init__.py`, re-mis-à-jour ici).

**Files:**
- Rename: `src/targets/interaction/eval.py` -> `src/targets/interaction/fields.py` (kernel `eval_fields`, contenu inchangé)
- Rename: `src/targets/interaction/targets.py` -> `src/targets/interaction/refs.py` (`robot_interaction_targets` / `environment_interaction_targets`, contenu inchangé)
- Modify: `src/targets/interaction/__init__.py` (2 lignes d'import)
- Modify: `tests/test_targets_public_api.py:8` (le SEUL import de sous-module direct `interaction.eval`)

**Interfaces:**
- Consumes: rien de nouveau.
- Produces: `targets.interaction.fields.eval_fields` (ex-`eval.eval_fields`), `targets.interaction.refs.robot_interaction_targets` / `environment_interaction_targets` (ex-`targets.*`). Le package `targets.interaction` re-exporte les MÊMES noms qu'avant.

- [ ] **Step 1: Renommer les deux modules par `git mv` (préserve l'historique)**

```bash
git mv src/targets/interaction/eval.py src/targets/interaction/fields.py
git mv src/targets/interaction/targets.py src/targets/interaction/refs.py
```

- [ ] **Step 2: Mettre à jour les imports internes du package `interaction`**

Dans `src/targets/interaction/__init__.py`, remplacer le corps par :

```python
"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot, assemble. Pure ops in the submodules; the flow is one-way
(pose -> eval -> transport -> assemble). ``pose_cloud`` is shared by every cloud kind."""
from .pointclouds import pose_cloud
from .fields import eval_fields
from .transport import transport
from .refs import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
```

- [ ] **Step 3: Mettre à jour le seul test important `interaction.eval` directement**

Dans `tests/test_targets_public_api.py`, remplacer la ligne 8 :

```python
    from src.targets.interaction.eval import eval_fields as _ef
```

par :

```python
    from src.targets.interaction.fields import eval_fields as _ef
```

- [ ] **Step 4: Vérifier qu'aucun autre importeur de `interaction.eval` / `interaction.targets` ne subsiste**

Run: `grep -rn "interaction\.eval\b\|interaction\.targets\b\|from \.eval import\|from \.targets import" src/ tests/`
Expected: AUCUNE sortie sauf le label de span `prof.span("interaction.eval", ...)` dans `src/targets/pipeline.py` (c'est une chaîne d'étiquette, PAS un import — on la laisse telle quelle).

- [ ] **Step 5: Lancer les tests d'interaction (rapides) — doivent rester VERTS**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_eval_fields.py tests/test_transport.py tests/test_pose_cloud.py tests/test_robot_cloud.py tests/test_targets_public_api.py -q`
Expected: PASS (rename behavior-preserving ; les mêmes noms sont re-exportés).

- [ ] **Step 6: Smoke d'import du package `targets`**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import src.targets; from src.targets import eval_fields, pose_cloud; from src.targets.interaction import robot_interaction_targets, environment_interaction_targets; print('rename ok')"`
Expected: `rename ok`

- [ ] **Step 7: Commit**

```bash
git add src/targets/interaction/ tests/test_targets_public_api.py
git commit -m "refactor(holov2): targets/interaction — eval.py->fields.py (noyau) + targets.py->refs.py (réfs)"
```

---

### Task 2 : Types d'éval q-dépendants dans `targets/contracts.py`

`StyleEval`, `ContactEval`, `ContactEnvEval` — frozen dataclasses numpy-only, avec validation de forme au `__post_init__` (cohérent avec `MultiChannelField`). Formes EXACTES du spec ; `nv = 6 + n_joints`.

**Files:**
- Modify: `src/targets/contracts.py` (ajout d'une section ÉVAL à la fin)
- Test: `tests/test_eval_contracts.py` (create)

**Interfaces:**
- Consumes: `MultiChannelField` (déjà dans `contracts.py`).
- Produces:
  - `StyleEval(position (L,3), rotation (L,3,3), jac_pos (L,3,nv), jac_rot (L,3,nv), link_names (L,))`
  - `ContactEval(field: MultiChannelField (C,M), point_jac (M,3,nv), probe_jac_obj (C,M,3,6), env: tuple[ContactEnvEval,...])`
  - `ContactEnvEval(field: MultiChannelField (C,P_i), cloud_jac_self (P_i,3,6), probe_jac_obj (C,P_i,3,6))`

- [ ] **Step 1: Écrire les tests (construction + validation de forme)**

Créer `tests/test_eval_contracts.py` :

```python
"""Les types d'éval q-dépendants (StyleEval / ContactEval / ContactEnvEval) : construction valide +
rejet des formes incohérentes au __post_init__ (cohérent avec MultiChannelField)."""
import numpy as np
import pytest

from src.targets.contracts import (ContactEnvEval, ContactEval, MultiChannelField, StyleEval)


def _mcf(c: int, p: int) -> MultiChannelField:
    return MultiChannelField(
        distance=np.zeros((c, p)), direction=np.zeros((c, p, 3)),
        witness=np.zeros((c, p, 3)), active=np.zeros((c, p), bool),
        channels=tuple(f"ch{i}" for i in range(c)))


def test_style_eval_construction_and_validation():
    L, nv = 4, 35
    se = StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                   jac_pos=np.zeros((L, 3, nv)), jac_rot=np.zeros((L, 3, nv)),
                   link_names=tuple(f"l{i}" for i in range(L)))
    assert se.position.shape == (L, 3) and se.jac_rot.shape == (L, 3, nv)

    with pytest.raises(ValueError):                                # L mismatch on jac_pos
        StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                  jac_pos=np.zeros((L + 1, 3, nv)), jac_rot=np.zeros((L, 3, nv)),
                  link_names=tuple(f"l{i}" for i in range(L)))
    with pytest.raises(ValueError):                               # nv mismatch jac_rot vs jac_pos
        StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                  jac_pos=np.zeros((L, 3, nv)), jac_rot=np.zeros((L, 3, nv + 1)),
                  link_names=tuple(f"l{i}" for i in range(L)))


def test_contact_eval_construction_and_validation():
    C, M, nv = 2, 5, 35
    env = (ContactEnvEval(field=_mcf(C, 3), cloud_jac_self=np.zeros((3, 3, 6)),
                          probe_jac_obj=np.zeros((C, 3, 3, 6))),)
    ce = ContactEval(field=_mcf(C, M), point_jac=np.zeros((M, 3, nv)),
                     probe_jac_obj=np.zeros((C, M, 3, 6)), env=env)
    assert ce.point_jac.shape == (M, 3, nv) and ce.probe_jac_obj.shape == (C, M, 3, 6)

    with pytest.raises(ValueError):                               # M mismatch on point_jac
        ContactEval(field=_mcf(C, M), point_jac=np.zeros((M + 1, 3, nv)),
                    probe_jac_obj=np.zeros((C, M, 3, 6)), env=env)
    with pytest.raises(ValueError):                               # probe_jac_obj last dim != 6
        ContactEval(field=_mcf(C, M), point_jac=np.zeros((M, 3, nv)),
                    probe_jac_obj=np.zeros((C, M, 3, 5)), env=env)


def test_contact_env_eval_validation():
    C, P = 2, 4
    ok = ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P, 3, 6)),
                        probe_jac_obj=np.zeros((C, P, 3, 6)))
    assert ok.cloud_jac_self.shape == (P, 3, 6)

    with pytest.raises(ValueError):                               # P mismatch on cloud_jac_self
        ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P + 1, 3, 6)),
                       probe_jac_obj=np.zeros((C, P, 3, 6)))
    with pytest.raises(ValueError):                               # probe_jac_obj channels mismatch
        ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P, 3, 6)),
                       probe_jac_obj=np.zeros((C + 1, P, 3, 6)))
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_eval_contracts.py -q`
Expected: FAIL (`ImportError: cannot import name 'StyleEval'` — les types n'existent pas encore).

- [ ] **Step 3: Ajouter les types d'éval à `contracts.py`**

Dans `src/targets/contracts.py`, repérer la fin de la classe `FrameTrace` (sa dernière ligne) :

```python
    targets: FrameTargets                          # final outputs (style + robot + env)
```

et l'étendre en y APPENDANT, juste après, la nouvelle section :

```python
    targets: FrameTargets                          # final outputs (style + robot + env)


# =============================================================================
# EVAL (q-dependent) — current geometric state + analytic Jacobians (targets.Evaluator)
# =============================================================================
# Mirror of the references above for the SAME conceptual op (pose a config, read style + contact),
# applied to the OPTIMISED config (robot @ q + objects @ SE(3)). Reference-free, cost-free: the
# residual (cur - ref) and the cost live in ``solve``. Tangent convention: pinocchio v
# (nv = 6 + n_joints) for q; world-aligned (δt, δθ) for each object (LOCAL_WORLD_ALIGNED).
@dataclass(frozen=True)
class StyleEval:
    """État courant des links suivis à ``q`` (FK), + jacobiennes géométriques. Reference-free,
    cost-free. Ordre = ``StyleTargets.link_names`` (mêmes links que la référence de style)."""

    position: np.ndarray         # (L, 3)      position monde courante du link
    rotation: np.ndarray         # (L, 3, 3)   rotation monde courante du link
    jac_pos: np.ndarray          # (L, 3, nv)  ∂position/∂v   (monde)
    jac_rot: np.ndarray          # (L, 3, nv)  ∂ω/∂v          (jac angulaire géométrique, monde)
    link_names: tuple[str, ...]  # (L,)

    def __post_init__(self) -> None:
        L = len(self.link_names)
        if self.position.shape != (L, 3):
            raise ValueError(f"position shape {self.position.shape} != ({L}, 3)")
        if self.rotation.shape != (L, 3, 3):
            raise ValueError(f"rotation shape {self.rotation.shape} != ({L}, 3, 3)")
        if self.jac_pos.ndim != 3 or self.jac_pos.shape[:2] != (L, 3):
            raise ValueError(f"jac_pos shape {self.jac_pos.shape} != ({L}, 3, nv)")
        nv = self.jac_pos.shape[2]
        if self.jac_rot.shape != (L, 3, nv):
            raise ValueError(f"jac_rot shape {self.jac_rot.shape} != jac_pos ({L}, 3, {nv})")


@dataclass(frozen=True)
class ContactEnvEval:
    """Côté env : nuage objet ``i`` vs canaux. Dépend des poses objets seules (pas de ``q``).
    Diagonale self-contact déjà neutralisée par ``eval_fields`` (``self_idx``) côté ``field`` ;
    ``probe_jac_obj`` y est rempli par la formule générique (inoffensif, la diagonale est ignorée
    par ``solve``). Tangente objet world-aligned ``(δt, δθ)``."""

    field: MultiChannelField   # (C, P_i)
    cloud_jac_self: np.ndarray  # (P_i, 3, 6)    ∂(point du nuage objet i, monde)/∂(tangente objet i)
    probe_jac_obj: np.ndarray  # (C, P_i, 3, 6) ∂(probe dans le frame canal)/∂(tangente SE(3) objet du canal)

    def __post_init__(self) -> None:
        C, P = self.field.n_channels, self.field.n_points
        if self.cloud_jac_self.shape != (P, 3, 6):
            raise ValueError(f"cloud_jac_self shape {self.cloud_jac_self.shape} != ({P}, 3, 6)")
        if self.probe_jac_obj.shape != (C, P, 3, 6):
            raise ValueError(f"probe_jac_obj shape {self.probe_jac_obj.shape} != ({C}, {P}, 3, 6)")


@dataclass(frozen=True)
class ContactEval:
    """Géométrie de contact courante (robot) + jacobiennes géométriques pour ``(q, object_poses)``.
    Reference-free, cost-free. Canal-first ``(C, M)`` sur les M points de contrôle robot. ``field``
    suit la convention ``MultiChannelField`` (sol en monde, canal objet en objet-local) ; ``point_jac``
    est en MONDE. ``probe_jac_obj`` : lignes du canal sol = 0 ; canal ``c`` -> objet
    ``channels[c].object_idx`` (creux). Tangente objet world-aligned ``(δt, δθ)``."""

    field: MultiChannelField   # (C, M)
    point_jac: np.ndarray      # (M, 3, nv)     ∂(point robot monde)/∂v
    probe_jac_obj: np.ndarray  # (C, M, 3, 6)   ∂(probe dans le frame canal)/∂(tangente SE(3) objet du canal)
    env: tuple[ContactEnvEval, ...]  # côté environnement, un par nuage objet

    def __post_init__(self) -> None:
        C, M = self.field.n_channels, self.field.n_points
        if self.point_jac.ndim != 3 or self.point_jac.shape[:2] != (M, 3):
            raise ValueError(f"point_jac shape {self.point_jac.shape} != ({M}, 3, nv)")
        if self.probe_jac_obj.shape != (C, M, 3, 6):
            raise ValueError(f"probe_jac_obj shape {self.probe_jac_obj.shape} != ({C}, {M}, 3, 6)")
```

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_eval_contracts.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Vérifier que `contracts.py` reste numpy-only-importable (compile + import sans torch)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m py_compile src/targets/contracts.py && ~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.targets.contracts; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules; print('contracts numpy-only ok')"`
Expected: `contracts numpy-only ok`

- [ ] **Step 6: Commit**

```bash
git add src/targets/contracts.py tests/test_eval_contracts.py
git commit -m "feat(holov2): targets/contracts — StyleEval/ContactEval/ContactEnvEval (état courant + jacobiennes)"
```

---

### Task 3 : `style/eval.py` — `style_eval(robot, q, link_names) -> StyleEval` (config-free)

FK + jacobiennes des links suivis. CONFIG-FREE : le SCALE/OFFSET de `style/config` ne sert qu'à la RÉFÉRENCE (`build.py`), pas à l'éval du robot — l'éval lit la cinématique brute via `robot.link_jacobians(q)` et gather les links suivis.

**Files:**
- Create: `src/targets/style/eval.py`
- Modify: `src/targets/style/__init__.py` (exporter `style_eval`)
- Test: `tests/test_style_eval.py` (create)

**Interfaces:**
- Consumes: `RobotModel.link_jacobians(q) -> (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))` et `RobotModel.link_names`/`integrate` (Plan 1) ; `StyleEval` (Task 2).
- Produces: `style_eval(robot: RobotModel, q: np.ndarray, link_names: tuple[str, ...]) -> StyleEval`.

- [ ] **Step 1: Écrire les tests (FK direct + jacobiennes par différences finies)**

Créer `tests/test_style_eval.py` :

```python
"""style_eval : FK + jacobiennes géométriques des links suivis (config-free). Gardé par le G1 URDF
(le moteur pinocchio de Plan 1). position/rotation == FK direct ; jac_pos/jac_rot vs différences
finies via robot.integrate (jac_rot = vitesse angulaire monde, FD par rotvec de la rotation relative)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import RobotSpec
from src.prepare.load.robot import build_robot_model
from src.targets.config import style_table
from src.targets.contracts import StyleEval
from src.targets.style.eval import style_eval

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
_SKIP = pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF absent")


def _robot():
    return build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))


@_SKIP
def test_style_eval_matches_fk_and_shapes():
    robot = _robot()
    link_names = tuple(style_table("g1").keys())               # tracked links = style recipe order
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(0).standard_normal(robot.nv))

    se = style_eval(robot, q, link_names)
    L, nv = len(link_names), robot.nv
    assert isinstance(se, StyleEval)
    assert se.position.shape == (L, 3) and se.rotation.shape == (L, 3, 3)
    assert se.jac_pos.shape == (L, 3, nv) and se.jac_rot.shape == (L, 3, nv)
    assert se.link_names == link_names

    rot_all, pos_all = robot.link_transforms(q)                # FK direct
    idx = [robot.link_names.index(n) for n in link_names]
    assert np.allclose(se.position, pos_all[idx], atol=1e-9)
    assert np.allclose(se.rotation, rot_all[idx], atol=1e-9)


@_SKIP
def test_style_eval_jacobians_match_finite_differences():
    robot = _robot()
    link_names = tuple(style_table("g1").keys())
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))

    se = style_eval(robot, q, link_names)
    L, nv, eps = len(link_names), robot.nv, 1e-6
    for k in range(nv):
        v = np.zeros(nv); v[k] = eps
        se_p = style_eval(robot, robot.integrate(q, v), link_names)
        se_m = style_eval(robot, robot.integrate(q, -v), link_names)
        fd_pos = (se_p.position - se_m.position) / (2 * eps)   # (L, 3) ∂pos/∂v_k
        assert np.allclose(se.jac_pos[:, :, k], fd_pos, atol=1e-4)
        for i in range(L):
            dR = se_p.rotation[i] @ se_m.rotation[i].T          # relative rotation in WORLD frame
            omega = R.from_matrix(dR).as_rotvec() / (2 * eps)   # (3,) ∂ω/∂v_k
            assert np.allclose(se.jac_rot[i, :, k], omega, atol=1e-4), (link_names[i], k)
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_style_eval.py -q`
Expected: FAIL (`ModuleNotFoundError: src.targets.style.eval`) — ou SKIP si le G1 URDF est absent (alors implémenter quand même puis Step 4).

- [ ] **Step 3: Implémenter `style_eval`**

Créer `src/targets/style/eval.py` :

```python
"""style.eval — FK courant + jacobiennes géométriques des links suivis -> ``StyleEval`` (q-dépendant).

Le pendant ÉVALUATION de ``style.build`` (réf, q-indép.) : ``build`` pose la démo humaine en cible de
style ; ``style_eval`` lit l'état COURANT du robot à ``q``. CONFIG-FREE : le SCALE/OFFSET de la recette
(``targets/config``) ne sert qu'à la référence (où placer la cible), pas à l'éval du robot — ici on
lit la cinématique brute (``RobotModel.link_jacobians``) et on gather les links suivis dans l'ordre de
la recette (= ``StyleTargets.link_names``), pour que ``solve`` aligne cur vs réf canal-par-canal.

``link_jacobians`` rend les transforms monde ET les jacobiennes de frame LOCAL_WORLD_ALIGNED (axes
monde) pour TOUS les links ; on en extrait les links suivis par leur NOM. Pur, float64, torch-free
(la cinématique lourde est cachée dans l'instance ``RobotModel``). Reference-free, cost-free.
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import RobotModel
from ..contracts import StyleEval


def style_eval(robot: RobotModel, q: np.ndarray, link_names: tuple[str, ...]) -> StyleEval:
    """État courant (FK) + jacobiennes des links ``link_names`` à ``q`` -> ``StyleEval``.

    ``robot.link_jacobians(q)`` rend ``(rot (L_all,3,3), pos (L_all,3), jac_lin (L_all,3,nv),
    jac_ang (L_all,3,nv))`` en repère monde, aligné sur ``robot.link_names`` ; on gather les links
    suivis (par nom). ``jac_pos`` = jac translationnelle, ``jac_rot`` = jac angulaire géométrique."""
    missing = [n for n in link_names if n not in robot.link_names]
    if missing:
        raise ValueError(f"style links absent from robot.link_names: {missing}")

    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)        # (L_all, ...) repère monde
    gather = np.array([robot.link_names.index(n) for n in link_names], np.int64)  # (L,) into FK order
    return StyleEval(
        position=np.ascontiguousarray(pos[gather]),            # (L, 3)
        rotation=np.ascontiguousarray(rot[gather]),            # (L, 3, 3)
        jac_pos=np.ascontiguousarray(jac_lin[gather]),         # (L, 3, nv)
        jac_rot=np.ascontiguousarray(jac_ang[gather]),         # (L, 3, nv)
        link_names=tuple(link_names),
    )
```

- [ ] **Step 4: Exporter `style_eval` sur la surface du sous-package `style`**

Dans `src/targets/style/__init__.py`, remplacer le corps par :

```python
"""Style treatment: reference (``build``: demo joints -> StyleTargets) + evaluation
(``style_eval``: robot FK @ q -> StyleEval). See ``build.py`` / ``eval.py``."""
from .build import build
from .eval import style_eval

__all__ = ["build", "style_eval"]
```

- [ ] **Step 5: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_style_eval.py -q`
Expected: PASS (2 tests) — ou SKIP si le G1 URDF est absent.

- [ ] **Step 6: Vérifier l'import torch-free du sous-package style**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; from src.targets.style import style_eval, build; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules; print('style import ok')"`
Expected: `style import ok`

- [ ] **Step 7: Commit**

```bash
git add src/targets/style/eval.py src/targets/style/__init__.py tests/test_style_eval.py
git commit -m "feat(holov2): style.eval — FK courant + jacobiennes des links suivis (config-free), validées par FD"
```

---

### Task 4 : `interaction/eval.py` (NOUVEAU) — `contact_eval(ctx, q, object_rot, object_pos) -> ContactEval`

Le pendant ÉVALUATION du flux interaction : pose le `robot_cloud` @FK(q), évalue via `fields.eval_fields`, et construit les jacobiennes analytiques (robot `point_jac`, objet `probe_jac_obj`, env `cloud_jac_self`). Formules figées du spec (tangente world-aligned).

**Files:**
- Create: `src/targets/interaction/eval.py` (note : `fields.py` est désormais le noyau ; `eval.py` est ré-créé pour l'évaluation q-dép.)
- Modify: `src/targets/interaction/__init__.py` (re-exporter `contact_eval`)
- Test: `tests/test_contact_eval.py` (create)

**Interfaces:**
- Consumes: `pose_cloud` (`.pointclouds`), `eval_fields` (`.fields`), `ContactEval`/`ContactEnvEval` (`..contracts`), `InteractionContext` (`...prepare.contracts`, attributs `robot_cloud`, `channels`, `object_clouds`, `margin`, `robot`), `RobotModel.link_jacobians`/`link_transforms`/`integrate`.
- Produces: `contact_eval(ctx: InteractionContext, q: np.ndarray, object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEval`.

**Dérivation `point_jac` (point de contrôle robot, vectorisée — clarification d'implémentation) :** `robot_cloud` est un `PointCloud` K=1 (`parts (M,1)` = index de link en ordre FK ; `offsets (M,1,3)` = offset rest-local sur ce link ; `weights (M,1)=1`). Pour un point de contrôle d'offset local `o` sur le link `(R, t)`, l'offset monde est `r = R @ o` et la jacobienne du point monde est `J_lin[link] - [r]_× J_ang[link]` (la vitesse du point = vitesse de l'origine du frame + `ω × r`, avec `ω = J_ang v`, soit `(J_lin - [r]_× J_ang) v`). On l'écrit en forme générale skinning (somme pondérée sur K), qui se réduit à K=1 pour le `robot_cloud` et reste exacte si un nuage K>1 était posé un jour.

- [ ] **Step 1: Écrire les tests (champ == eval_fields direct ; point_jac / probe_jac_obj / env vs FD)**

Créer `tests/test_contact_eval.py` (ctx SYNTHÉTIQUE : `contact_eval` ne touche que `robot_cloud`/`channels`/`object_clouds`/`margin`/`robot` — pas besoin des données SMPL/HODome, seulement du G1 URDF pour des jacobiennes cohérentes) :

```python
"""contact_eval : champ courant + jacobiennes analytiques pour (q, poses objets). Gardé par le G1
URDF (jacobiennes cohérentes via le moteur pinocchio de Plan 1). ctx synthétique : contact_eval ne
lit que robot_cloud / channels / object_clouds / margin / robot. Les jacobiennes sont validées par
différences finies ; tangente objet world-aligned (δt, δθ): rotation = expm([δθ]_x) R (gauche, monde)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import (Channel, CorrespondenceTable, InteractionContext, PointCloud,
                                    RobotSpec)
from src.prepare.load.robot import build_robot_model
from src.prepare.sdf.build import build_plane_sdf
from src.targets.interaction import contact_eval, eval_fields, pose_cloud

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
_SKIP = pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF absent")


def _robot():
    return build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))


def _ctx(robot):
    """Synthetic InteractionContext: a few robot control points on REAL links, a ground + one object
    plane channel, one object cloud. human_cloud / correspondence are unused dummies."""
    names = [n for n in ("left_elbow_link", "pelvis", "right_knee_link") if n in robot.link_names]
    rng = np.random.default_rng(3)
    parts = np.array([[robot.link_names.index(n)] for n in names], np.int64)        # (M, 1) FK order
    offsets = (0.05 * rng.standard_normal((len(names), 1, 3)))                       # (M, 1, 3) local
    robot_cloud = PointCloud(parts=parts, weights=np.ones((len(names), 1)), offsets=offsets)

    ground = Channel("ground", None, build_plane_sdf([-2, -2], [2, 2], spacing=0.1, margin=0.5, name="ground"))
    obj = Channel("obj0", 0, build_plane_sdf([-2, -2], [2, 2], spacing=0.1, margin=0.5, name="obj0"))
    obj_cloud = PointCloud(parts=np.zeros((2, 1), np.int64), weights=np.ones((2, 1)),
                           offsets=np.array([[[0.1, 0.0, 0.0]], [[0.0, 0.1, 0.0]]]))
    dummy = PointCloud(parts=np.zeros((1, 1), np.int64), weights=np.ones((1, 1)), offsets=np.zeros((1, 1, 3)))
    corr = CorrespondenceTable(smpl_idx=np.zeros(1, np.int64), link_idx=np.zeros(1, np.int64),
                               offset_local=np.zeros((1, 3)), link_names=robot.link_names)
    return InteractionContext(channels=(ground, obj), human_cloud=dummy, object_clouds=(obj_cloud,),
                              correspondence=corr, margin=0.5, robot_cloud=robot_cloud, robot=robot)


def _obj_pose():
    rot = R.from_rotvec([0.2, -0.5, 0.3]).as_matrix()[None]    # (1, 3, 3)
    pos = np.array([[0.4, -0.2, 0.6]])                         # (1, 3)
    return rot, pos


@_SKIP
def test_contact_eval_field_matches_eval_fields_direct():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(0).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))    # (M, 3) world
    ref = eval_fields(pts, ctx.channels, object_rot, object_pos, ctx.margin)
    assert np.allclose(ce.field.distance, ref.distance)
    assert np.allclose(ce.field.witness, ref.witness)
    assert np.allclose(ce.field.direction, ref.direction)
    M, nv = ctx.robot_cloud.n_points, robot.nv
    assert ce.point_jac.shape == (M, 3, nv)
    assert ce.probe_jac_obj.shape == (len(ctx.channels), M, 3, 6)


@_SKIP
def test_point_jac_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    nv, eps = robot.nv, 1e-6
    for k in range(nv):
        v = np.zeros(nv); v[k] = eps
        p_plus = pose_cloud(ctx.robot_cloud, *robot.link_transforms(robot.integrate(q, v)))
        p_minus = pose_cloud(ctx.robot_cloud, *robot.link_transforms(robot.integrate(q, -v)))
        fd = (p_plus - p_minus) / (2 * eps)                    # (M, 3) ∂(point monde)/∂v_k
        assert np.allclose(ce.point_jac[:, :, k], fd, atol=1e-4), k


@_SKIP
def test_probe_jac_obj_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(2).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    points = pose_cloud(ctx.robot_cloud, *robot.link_transforms(q))   # (M, 3) world, held FIXED
    c = 1                                                            # channels[1] is the object channel
    j = ctx.channels[c].object_idx                                   # object index of that channel

    def probe_x(rot_j, pos_j):
        return (points - pos_j) @ rot_j                            # (M, 3) = R_jᵀ (p - t_j)

    eps = 1e-6
    for a in range(3):                                             # δt columns 0..2
        dt = np.zeros(3); dt[a] = eps
        fd = (probe_x(object_rot[j], object_pos[j] + dt) - probe_x(object_rot[j], object_pos[j] - dt)) / (2 * eps)
        assert np.allclose(ce.probe_jac_obj[c, :, :, a], fd, atol=1e-6), ("δt", a)
    for a in range(3):                                             # δθ columns 3..5 (world-aligned)
        w = np.zeros(3); w[a] = eps
        rp = R.from_rotvec(w).as_matrix() @ object_rot[j]
        rm = R.from_rotvec(-w).as_matrix() @ object_rot[j]
        fd = (probe_x(rp, object_pos[j]) - probe_x(rm, object_pos[j])) / (2 * eps)
        assert np.allclose(ce.probe_jac_obj[c, :, :, 3 + a], fd, atol=1e-6), ("δθ", a)
    assert np.allclose(ce.probe_jac_obj[0], 0.0)                   # ground channel rows = 0


@_SKIP
def test_env_cloud_jac_self_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(4).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    assert len(ce.env) == len(ctx.object_clouds)
    env0 = ce.env[0]

    # field == eval_fields direct on the posed object cloud (self_idx=0)
    obj_world = pose_cloud(ctx.object_clouds[0], object_rot[0][None], object_pos[0][None])
    ref = eval_fields(obj_world, ctx.channels, object_rot, object_pos, ctx.margin, self_idx=0)
    assert np.allclose(env0.field.distance, ref.distance)
    assert np.allclose(env0.field.witness, ref.witness)

    eps = 1e-6
    for a in range(3):                                            # δt: ∂p/∂δt = I
        dt = np.zeros(3); dt[a] = eps
        pp = pose_cloud(ctx.object_clouds[0], object_rot[0][None], (object_pos[0] + dt)[None])
        pm = pose_cloud(ctx.object_clouds[0], object_rot[0][None], (object_pos[0] - dt)[None])
        fd = (pp - pm) / (2 * eps)
        assert np.allclose(env0.cloud_jac_self[:, :, a], fd, atol=1e-6), ("δt", a)
    for a in range(3):                                            # δθ: ∂p/∂δθ = -[p - t_i]_x
        w = np.zeros(3); w[a] = eps
        rp = R.from_rotvec(w).as_matrix() @ object_rot[0]
        rm = R.from_rotvec(-w).as_matrix() @ object_rot[0]
        pp = pose_cloud(ctx.object_clouds[0], rp[None], object_pos[0][None])
        pm = pose_cloud(ctx.object_clouds[0], rm[None], object_pos[0][None])
        fd = (pp - pm) / (2 * eps)
        assert np.allclose(env0.cloud_jac_self[:, :, 3 + a], fd, atol=1e-6), ("δθ", a)
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_contact_eval.py -q`
Expected: FAIL (`ImportError: cannot import name 'contact_eval'`) — ou SKIP si le G1 URDF est absent.

- [ ] **Step 3: Implémenter `contact_eval` (+ helpers `_skew`, `_probe_jac`, `_env_eval`)**

Créer `src/targets/interaction/eval.py` :

```python
"""contact_eval — pendant ÉVALUATION du flux interaction (q-dépendant) : pose le ``robot_cloud``
@FK(q), évalue contre tous les canaux (``fields.eval_fields``) et construit les jacobiennes
géométriques ANALYTIQUES -> ``ContactEval``. Reference-free, cost-free : ``solve`` compose les
résidus/coûts. Tangente world-aligned (LOCAL_WORLD_ALIGNED), cohérente ``RobotModel.link_jacobians``.

Formules figées (spec) :
- Robot (point de contrôle, offset local ``o`` sur link ``(R, t)``, offset monde ``r = R @ o``) :
  ``point_jac = J_lin - [r]_× J_ang``  (somme pondérée sur K ; ``robot_cloud`` est K=1).
- Objet, probe ``x = R_iᵀ (p - t_i)`` vs tangente ``(δt, δθ)`` monde :
  ``∂x/∂δt = -R_iᵀ`` , ``∂x/∂δθ = R_iᵀ [p - t_i]_×``.
- Objet, nuage propre ``p = t_i + R_i o`` : ``∂p/∂δt = I₃`` , ``∂p/∂δθ = -[p - t_i]_×``.

Pur, array-oriented, torch-free (la cinématique lourde est cachée dans ``ctx.robot``). Ported de la
logique V1 ``contact/*`` côté valeur ; les jacobiennes sont neuves (analytiques).
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import Channel, InteractionContext
from ..contracts import ContactEnvEval, ContactEval
from .fields import eval_fields
from .pointclouds import pose_cloud


def _skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) matrice antisymétrique ``[v]_×`` (``[v]_× a = v × a``)."""
    v = np.asarray(v, np.float64)
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        np.stack([z, -w, y], axis=-1),
        np.stack([w, z, -x], axis=-1),
        np.stack([-y, x, z], axis=-1),
    ], axis=-2)


def _probe_jac(channels: tuple[Channel, ...], points: np.ndarray,
               object_rot: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
    """``(C, P, 3, 6)`` ∂(probe dans le frame canal)/∂(tangente SE(3) de l'objet du canal). Canal sol
    (``object_idx is None``) -> lignes 0. Canal objet ``i`` : ``∂x/∂δt = -R_iᵀ``, ``∂x/∂δθ =
    R_iᵀ [p - t_i]_×`` (probe ``x = R_iᵀ(p - t_i)``, ``p`` = point monde tenu fixe)."""
    p_count = points.shape[0]
    out = np.zeros((len(channels), p_count, 3, 6))                  # (C, P, 3, 6)
    for c, ch in enumerate(channels):
        if ch.object_idx is None:
            continue                                               # ground rows stay 0
        i = ch.object_idx
        rit = np.asarray(object_rot[i], np.float64).T              # (3, 3) = R_iᵀ
        ti = np.asarray(object_pos[i], np.float64)                 # (3,)
        out[c, :, :, 0:3] = -rit[None, :, :]                       # ∂x/∂δt = -R_iᵀ (broadcast sur P)
        out[c, :, :, 3:6] = rit[None] @ _skew(points - ti)        # R_iᵀ [p - t_i]_×   (P, 3, 3)
    return out


def _env_eval(ctx: InteractionContext, i: int,
              object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEnvEval:
    """Côté env pour le nuage objet ``i`` : champ (self_idx=i) + ``cloud_jac_self`` + ``probe_jac_obj``."""
    obj_world = pose_cloud(ctx.object_clouds[i], object_rot[i][None], object_pos[i][None])  # (P_i, 3)
    field = eval_fields(obj_world, ctx.channels, object_rot, object_pos, ctx.margin, self_idx=i)

    ti = np.asarray(object_pos[i], np.float64)                     # (3,)
    p_count = obj_world.shape[0]
    cloud_jac_self = np.zeros((p_count, 3, 6))                     # (P_i, 3, 6)
    cloud_jac_self[:, :, 0:3] = np.eye(3)[None]                    # ∂p/∂δt = I₃
    cloud_jac_self[:, :, 3:6] = -_skew(obj_world - ti)            # ∂p/∂δθ = -[p - t_i]_×

    probe_jac_obj = _probe_jac(ctx.channels, obj_world, object_rot, object_pos)  # (C, P_i, 3, 6)
    return ContactEnvEval(field=field, cloud_jac_self=cloud_jac_self, probe_jac_obj=probe_jac_obj)


def contact_eval(ctx: InteractionContext, q: np.ndarray,
                 object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEval:
    """État de contact courant (robot) + jacobiennes pour ``(q, object_poses)`` -> ``ContactEval``.

    ``object_rot (N, 3, 3)`` / ``object_pos (N, 3)`` sont les poses objets monde courantes (mêmes que
    la réf). Pose le ``robot_cloud`` @FK(q) via ``ctx.robot.link_jacobians(q)`` (transforms + jac de
    frame monde), évalue contre ``ctx.channels``, et assemble ``point_jac`` (monde), ``probe_jac_obj``
    (frame canal) et le côté ``env`` (un par nuage objet)."""
    robot = ctx.robot
    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)           # (L,3,3),(L,3),(L,3,nv),(L,3,nv) monde
    cloud = ctx.robot_cloud
    parts = np.asarray(cloud.parts)                               # (M, K) into FK link order
    weights = np.asarray(cloud.weights, np.float64)              # (M, K)
    offsets = np.asarray(cloud.offsets, np.float64)             # (M, K, 3) link-local

    points = pose_cloud(cloud, rot, pos)                          # (M, 3) world robot control points

    r = np.einsum("mkij,mkj->mki", rot[parts], offsets)          # (M, K, 3) world offset link->point
    contrib = jac_lin[parts] - np.einsum("mkij,mkjn->mkin", _skew(r), jac_ang[parts])  # (M,K,3,nv)
    point_jac = np.einsum("mk,mkin->min", weights, contrib)      # (M, 3, nv) = J_lin - [r]_× J_ang

    field = eval_fields(points, ctx.channels, object_rot, object_pos, ctx.margin)       # (C, M)
    probe_jac_obj = _probe_jac(ctx.channels, points, object_rot, object_pos)            # (C, M, 3, 6)
    env = tuple(_env_eval(ctx, i, object_rot, object_pos) for i in range(len(ctx.object_clouds)))
    return ContactEval(field=field, point_jac=point_jac, probe_jac_obj=probe_jac_obj, env=env)
```

- [ ] **Step 4: Re-exporter `contact_eval` sur le package `interaction`**

Dans `src/targets/interaction/__init__.py`, remplacer le corps par :

```python
"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot, assemble (references); evaluate the robot @ q (``contact_eval``). Pure ops
in the submodules; the flow is one-way (pose -> eval -> transport -> assemble)."""
from .pointclouds import pose_cloud
from .fields import eval_fields
from .eval import contact_eval
from .transport import transport
from .refs import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "contact_eval", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
```

- [ ] **Step 5: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_contact_eval.py -q`
Expected: PASS (4 tests) — ou SKIP si le G1 URDF est absent.

- [ ] **Step 6: Non-régression du flux référence + import torch-free**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_eval_fields.py tests/test_transport.py tests/test_targets_public_api.py -q && ~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; from src.targets.interaction import contact_eval, eval_fields; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules; print('interaction torch-free ok')"`
Expected: PASS puis `interaction torch-free ok`.

- [ ] **Step 7: Commit**

```bash
git add src/targets/interaction/eval.py src/targets/interaction/__init__.py tests/test_contact_eval.py
git commit -m "feat(holov2): interaction.contact_eval — champ courant + jacobiennes analytiques (point/probe/cloud), FD"
```

---

### Task 5 : `targets/evaluator.py` — `class Evaluator` (orchestrateur q-dép., construit 1×)

Réunit `style_eval` et `contact_eval` derrière l'API solveur. Construit UNE fois à partir des assets statiques de `ctx` ; les links suivis sont dérivés de la recette de style (comme `style/build.py` : `tuple(style_table(robot_name).keys())`).

**Files:**
- Create: `src/targets/evaluator.py`
- Test: `tests/test_evaluator.py` (create — intégration gardée par les données, comme `test_solve_field_eval`)

**Interfaces:**
- Consumes: `style_eval` (`.style`), `contact_eval` (`.interaction`), `style_table` (`.config`), `InteractionContext` (`..prepare.contracts`), `StyleEval`/`ContactEval` (`.contracts`).
- Produces: `Evaluator(ctx: InteractionContext, robot_name: str)` avec `.style(q) -> StyleEval` et `.contacts(q, object_rot, object_pos) -> ContactEval`.

> **Note (dérivation des links suivis & signature)** : `InteractionContext.robot` est un `RobotModel` (pas de `.name`), or la recette de style est robot-keyée par NOM (`config.style_table(robot_name)`, exactement la clé qu'utilise `style/build.py` via `robot.name`). L'évaluateur prend donc `robot_name` en 2ᵉ argument de construction (le spec écrit `Evaluator(ctx)` de façon abrégée). `.contacts` prend `(q, object_rot, object_pos)` — la même décomposition `(object_rot, object_pos)` que `eval_fields` / `FrameTargets`, plutôt qu'un tuple `object_poses`.

- [ ] **Step 1: Écrire le test d'intégration (gardé par les données HODome/SMPL-X/corr/URDF)**

Créer `tests/test_evaluator.py` :

```python
"""Evaluator (intégration) sur des assets réellement préparés (data-gated, comme test_solve_field_eval).
Construit Evaluator(ctx, "g1") une fois ; à q neutre, StyleEval.position est fini et de bonnes formes,
et ContactEval.field retombe sur eval_fields direct sur le robot_cloud posé @q — sanity du seam."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets import Evaluator, ContactEval, StyleEval, eval_fields, pose_cloud
from src.targets.config import style_table
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
def test_evaluator_style_and_contacts_on_real_assets(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    ev = Evaluator(ctx, spec.robot.name)
    q = ctx.robot.neutral()                                    # valid free-flyer config (nq,)

    se = ev.style(q)
    assert isinstance(se, StyleEval)
    L = len(style_table("g1"))
    assert se.position.shape == (L, 3) and se.jac_pos.shape == (L, 3, ctx.robot.nv)
    assert np.isfinite(se.position).all() and np.isfinite(se.jac_pos).all()

    ft = process_frame(g, ctx, spec.robot, f=0)
    ce = ev.contacts(q, ft.object_rot, ft.object_pos)
    assert isinstance(ce, ContactEval)
    M, C = ctx.correspondence.n_points, len(ctx.channels)
    assert ce.point_jac.shape == (M, 3, ctx.robot.nv)
    assert ce.probe_jac_obj.shape == (C, M, 3, 6)
    assert len(ce.env) == len(ctx.object_clouds)

    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))   # (M, 3) world
    ref = eval_fields(pts, ctx.channels, ft.object_rot, ft.object_pos, ctx.margin)
    assert np.allclose(ce.field.distance, ref.distance)       # ContactEval.field == eval_fields direct
    assert np.allclose(ce.field.witness, ref.witness)
```

- [ ] **Step 2: Lancer, vérifier l'échec (ou SKIP si données absentes)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_evaluator.py -q`
Expected: FAIL (`ImportError: cannot import name 'Evaluator'`) — ou SKIP si les données HODome/SMPL-X/corr/URDF sont absentes (alors implémenter puis valider via Step 5 import).

- [ ] **Step 3: Implémenter `Evaluator`**

Créer `src/targets/evaluator.py` :

```python
"""evaluator — orchestrateur q-DÉPENDANT du seam ``targets -> solve`` (objet ``Evaluator``).

Réunit les deux concerns (qui ne se connaissent pas) derrière l'API solveur : ``style_eval`` (FK des
links suivis) et ``contact_eval`` (champ + jacobiennes de contact). Construit UNE fois (séquence-wide)
à partir des assets STATIQUES de l'``InteractionContext`` (``robot_cloud``, ``channels``,
``object_clouds``, ``margin``, ``robot``) ; les seules entrées par itération sont les variables de
décision ``(q, object_poses)``. Les références par frame (``list[FrameTargets]``) sont une sortie
PARALLÈLE (``pipeline``), pas une entrée de l'évaluateur.

Les links suivis sont dérivés de la recette de style robot-keyée — exactement la clé qu'utilise
``style/build.py`` (``config.style_table(robot.name).keys()``) ; ``ctx.robot`` étant un ``RobotModel``
sans nom, le nom du robot est passé à la construction. Pur, torch-free (cinématique cachée dans
``ctx.robot``).
"""
from __future__ import annotations

import numpy as np

from ..prepare.contracts import InteractionContext
from .config import style_table
from .contracts import ContactEval, StyleEval
from .interaction import contact_eval
from .style import style_eval


class Evaluator:
    """Évaluateur q-dépendant construit une fois à partir de ``ctx``. ``robot_name`` clé la recette de
    style (links suivis). Expose ``.style(q)`` et ``.contacts(q, object_rot, object_pos)``."""

    def __init__(self, ctx: InteractionContext, robot_name: str) -> None:
        self._ctx = ctx
        self._robot = ctx.robot
        self._style_links: tuple[str, ...] = tuple(style_table(robot_name).keys())  # = StyleTargets order

    def style(self, q: np.ndarray) -> StyleEval:
        """État courant (FK) + jacobiennes des links suivis à ``q``."""
        return style_eval(self._robot, q, self._style_links)

    def contacts(self, q: np.ndarray, object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEval:
        """Champ de contact courant (robot) + jacobiennes pour ``(q, object_poses)``."""
        return contact_eval(self._ctx, q, object_rot, object_pos)
```

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_evaluator.py -q`
Expected: PASS (1 test) — ou SKIP si les données sont absentes.

- [ ] **Step 5: Smoke d'import + torch-free (indépendant des données)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; from src.targets.evaluator import Evaluator; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules; print('evaluator import ok')"`
Expected: `evaluator import ok`

- [ ] **Step 6: Commit**

```bash
git add src/targets/evaluator.py tests/test_evaluator.py
git commit -m "feat(holov2): targets.Evaluator — orchestrateur q-dép (style/contacts), construit 1x depuis ctx"
```

---

### Task 6 : Surface publique `targets/__init__.py` + doc `docs/TARGETS.md`

Exposer `Evaluator`, `StyleEval`, `ContactEval`, `ContactEnvEval` (en plus de l'existant) sur le package, et documenter la scission référence/éval + l'évaluateur.

**Files:**
- Modify: `src/targets/__init__.py`
- Modify: `docs/TARGETS.md`
- Test: `tests/test_targets_public_api.py` (ajout d'un test de surface)

**Interfaces:**
- Consumes: `Evaluator` (`.evaluator`), `StyleEval`/`ContactEval`/`ContactEnvEval` (`.contracts`).
- Produces: `from src.targets import Evaluator, StyleEval, ContactEval, ContactEnvEval` (+ existant).

- [ ] **Step 1: Écrire le test de surface (ajout à `test_targets_public_api.py`)**

Ajouter à la fin de `tests/test_targets_public_api.py` :

```python
def test_public_surface_exports_evaluator_and_eval_types():
    from src.targets import Evaluator, StyleEval, ContactEval, ContactEnvEval
    from src.targets.evaluator import Evaluator as _Ev
    from src.targets.contracts import (StyleEval as _SE, ContactEval as _CE,
                                       ContactEnvEval as _CEE)
    assert Evaluator is _Ev                # same object, re-exported at the package surface
    assert StyleEval is _SE and ContactEval is _CE and ContactEnvEval is _CEE
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_targets_public_api.py::test_public_surface_exports_evaluator_and_eval_types -q`
Expected: FAIL (`ImportError: cannot import name 'Evaluator' from 'src.targets'`).

- [ ] **Step 3: Étendre `targets/__init__.py`**

Remplacer le corps de `src/targets/__init__.py` par :

```python
"""``targets`` stage — online: per-frame style + interaction REFERENCES (q-independent) AND the
q-dependent EVALUATOR (the ``targets -> solve`` seam).

Public surface (what downstream stages import): ``targets.contracts`` (the data types it produces —
references ``FrameTargets``/``FrameTrace`` + eval ``StyleEval``/``ContactEval``/``ContactEnvEval``),
``targets.config`` (its knobs — ``TargetsConfig`` / ``StyleConfig``), the pure interaction kernel
reused by ``solve`` (``pose_cloud`` / ``eval_fields`` -> ``MultiChannelField``, plus the geodesic
readers), AND the ``Evaluator`` (built once from ``InteractionContext``, evaluates ``(q, object_poses)``
-> current geometry + Jacobians). Import these from the PACKAGE (``from ..targets import ...``), never
from the internal ``targets.interaction`` submodule. It consumes the upstream ``prepare`` contracts;
``solve`` and ``viz`` import their inputs from ``targets.contracts``.
"""
from .interaction import eval_fields, pose_cloud, geo_value_grad, nearest_index
from .contracts import MultiChannelField, StyleEval, ContactEval, ContactEnvEval
from .evaluator import Evaluator

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField", "geo_value_grad", "nearest_index",
           "Evaluator", "StyleEval", "ContactEval", "ContactEnvEval"]
```

- [ ] **Step 4: Mettre à jour `docs/TARGETS.md` (scission réf/éval + évaluateur)**

Dans `docs/TARGETS.md`, remplacer le bloc « Modules (1 responsabilité chacun) » (de la ligne ` ```` ` ouvrante après le titre jusqu'à la ` ```` ` fermante) par :

````markdown
```
targets/
  config.py            knobs de targets (StyleConfig échelle morpho + hauteurs / TargetsConfig) ET la
                       recette de style robot-keyed (DATA : SMPL_BODY_INDEX, ARM_BODIES, _STYLE_TABLE
                       lien->corps/poids/offsets, style_table()) — tout le style hors math en un fichier.
  contracts.py         RÉFS  : ContactField, MultiChannelField, StyleTargets, Robot/EnvironmentInteractionTargets,
                       FrameTargets, FramePose, FrameTrace.  ÉVAL : StyleEval, ContactEval, ContactEnvEval
                       (état géométrique courant + jacobiennes analytiques ; reference-free, cost-free).
  evaluator.py         ÉVALUATEUR q-dépendant (orchestrateur) : Evaluator(ctx, robot_name), construit 1×.
                         .style(q)                       -> StyleEval   (FK links suivis + J_pos/J_rot)
                         .contacts(q, object_rot, object_pos) -> ContactEval (field courant + point_jac
                                                                + probe_jac_obj + env)
  style/               OBJECTIF DE STYLE (ex-"body") — posture, ignore l'objet.
    build.py           (réf) FramePose démo -> StyleTargets (recette GMR : SCALE puis OFFSET).
    eval.py            (éval) style_eval : robot FK @ q -> StyleEval (config-free ; gather des links suivis).
  interaction/
    pointclouds.py     pose_cloud — pose tout nuage (humain K~4 / objet·robot K=1) [noyau partagé]
    fields.py          eval_fields (sample chaque Channel : SDF trilinéaire — chemin unique) [noyau, ex-eval.py]
    eval.py            (éval) contact_eval : robot_cloud @FK(q) + objets @SE(3) -> ContactEval
                       (field courant + jacobiennes : point_jac monde, probe_jac_obj/cloud_jac_self objet)
    transport.py       transport (gather via correspondence) [réf]
    refs.py            robot_interaction_targets / environment_interaction_targets [réf, ex-targets.py]
    geodesic.py        geo_value_grad / nearest_index — lecture différentiable du champ géodésique. [noyau]
  pipeline.py          RÉFÉRENCES : process_frame / trace_frame -> FrameTargets ; run_sequence.
```

## Symétrie référence / évaluation
`targets` fait UNE opération (poser une config, lire son état style + contact) sur DEUX configs :
- **Référence** (`pipeline.py`, q-indép., 1×/frame, bakée) : humain/démo -> `FrameTargets` (style +
  contact refs transportés). Pas de jacobienne.
- **Évaluation** (`evaluator.py`, q-dép., par itération) : robot @ `q` + objets @ `SE(3)` ->
  `StyleEval` / `ContactEval` (état courant + jacobiennes analytiques). Les MÊMES noyaux purs
  (`pose_cloud`, `fields.eval_fields`) servent les deux ; seul le côté cible porte une jacobienne.
Le résidu (`cur - ref`) et le coût se font **dans `solve`** à partir de ces deux sorties.
`Evaluator` est construit UNE fois (assets statiques de `ctx`) ; entrées par itération = `(q, object_poses)`.
````

- [ ] **Step 5: Lancer la suite de surface + compile/import**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_targets_public_api.py -q && ~/.holonew_deps/miniconda3/envs/holonew/bin/python -m py_compile src/targets/__init__.py src/targets/evaluator.py && ~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.targets; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules; print('targets public surface torch-free ok')"`
Expected: PASS puis `targets public surface torch-free ok`.

- [ ] **Step 6: Suite targets complète (verte ou SKIP, aucun FAIL)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_eval_contracts.py tests/test_style_eval.py tests/test_contact_eval.py tests/test_evaluator.py tests/test_eval_fields.py tests/test_transport.py tests/test_pose_cloud.py tests/test_robot_cloud.py tests/test_targets_public_api.py -q`
Expected: PASS ou SKIP (tests gardés par G1 URDF / données HODome) ; aucun FAIL.

- [ ] **Step 7: Commit**

```bash
git add src/targets/__init__.py docs/TARGETS.md tests/test_targets_public_api.py
git commit -m "feat(holov2): targets — surface publique (Evaluator + types d'éval) + doc scission réf/éval"
```

---

## Self-Review

**1. Spec coverage** (vs `2026-06-29-targets-evaluator-seam-design.md`) :

| Exigence du spec | Tâche |
|---|---|
| Réorg `interaction/eval.py` -> `fields.py` (noyau) | Task 1 ✅ |
| Réorg `interaction/targets.py` -> `refs.py` (réfs) | Task 1 ✅ |
| `StyleEval` (position/rotation/jac_pos/jac_rot/link_names) | Task 2 ✅ |
| `ContactEval` (field/point_jac/probe_jac_obj/env) | Task 2 ✅ |
| `ContactEnvEval` (field/cloud_jac_self/probe_jac_obj) | Task 2 ✅ |
| `__post_init__` valide les formes (L/M/C/nv) | Task 2 ✅ |
| `style/eval.py` `style_eval(robot, q, link_names)` config-free | Task 3 ✅ |
| `interaction/eval.py` `contact_eval` (NOUVEAU) | Task 4 ✅ |
| Formule robot `point_jac = J_lin - [R·o]_× J_ang` | Task 4 (Step 3 + FD Step 1) ✅ |
| Formule objet probe `∂x/∂δt = -R_iᵀ`, `∂x/∂δθ = R_iᵀ[p-t_i]_×` | Task 4 (`_probe_jac` + FD) ✅ |
| Formule env `cloud_jac_self` `∂p/∂δt = I`, `∂p/∂δθ = -[p-t_i]_×` | Task 4 (`_env_eval` + FD) ✅ |
| `field` (canal frame) == `eval_fields` direct ; jac monde | Task 4 (tests field==direct, convention de repère documentée) ✅ |
| `Evaluator(ctx)` construit 1× ; `.style` / `.contacts` | Task 5 ✅ |
| Links suivis = `style_table(robot_name).keys()` (comme build) | Task 5 (`_style_links`) ✅ |
| Intégration : `StyleEval.position` finie, `ContactEval.field` == direct | Task 5 ✅ |
| Surface publique : `Evaluator`/`StyleEval`/`ContactEval`/`ContactEnvEval` | Task 6 ✅ |
| Doc `TARGETS.md` scission réf/éval | Task 6 ✅ |
| Compile/import numpy-only (contracts) | Task 2 Step 5, Task 6 Step 5 ✅ |
| Imports mis à jour après rename (tests verts) | Task 1 (Steps 2-5) ✅ |
| Conformité #1/#2 (acyclique, pas de noyau central) | Global Constraints + imports relatifs/package ✅ |
| Hors périmètre (résidus/coûts/cônes -> `solve`) | non planifié (correct) ✅ |
| Changements `prepare` (`RobotModel.link_jacobians`, pinocchio) | Plan 1 (acquis, non replanifié) ✅ |

**2. Placeholder scan** : aucun `TBD`/`TODO`/« implement later » ; chaque step de code porte le code complet et chaque test de différences finies est écrit en entier (Tasks 3, 4). Toutes les commandes ont une sortie attendue.

**3. Type consistency** :
- `StyleEval` : `position (L,3)`, `rotation (L,3,3)`, `jac_pos (L,3,nv)`, `jac_rot (L,3,nv)`, `link_names (L,)` — identiques entre la déclaration (Task 2), le constructeur de `style_eval` (Task 3), et les tests (Tasks 3, 5). ✅
- `ContactEval` : `field (C,M)`, `point_jac (M,3,nv)`, `probe_jac_obj (C,M,3,6)`, `env tuple[ContactEnvEval,...]` — cohérent entre Task 2, `contact_eval` (Task 4), tests (Tasks 4, 5). ✅
- `ContactEnvEval` : `field (C,P_i)`, `cloud_jac_self (P_i,3,6)`, `probe_jac_obj (C,P_i,3,6)` — cohérent entre Task 2, `_env_eval` (Task 4), test (Task 4). ✅
- Signatures : `style_eval(robot, q, link_names)` (Task 3) == appel dans `Evaluator.style` (Task 5). `contact_eval(ctx, q, object_rot, object_pos)` (Task 4) == appel dans `Evaluator.contacts` (Task 5). `Evaluator(ctx, robot_name)` (Task 5) == construction dans le test (Task 5) et la doc (Task 6). ✅
- `_skew` réutilisé identiquement par `point_jac`, `_probe_jac`, `_env_eval` (un seul helper, DRY). ✅
- Convention de signe `point_jac` (`J_lin - [r]_× J_ang`) validée par le test FD `test_point_jac_matches_finite_differences` (Task 4), cohérente avec `pose_cloud` (les transforms viennent du MÊME `link_jacobians(q)`). ✅
- Re-exports : `interaction/__init__` exporte `eval_fields` (depuis `fields`), `contact_eval` (depuis `eval`), `refs.*` — `pipeline.py` et `targets/__init__` consomment via le package, jamais le sous-module (sauf le test public qui pointe `interaction.fields`, mis à jour Task 1). ✅
