# viz — Phase C : couches d'interaction roadmap (#3 contacts · #4 correspondance · #6 sdf_iso · #7 geodesic) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter au viewer **PROD** unifié (`viz/app.py`) les **4 couches composables** de la roadmap viz-solve qui ne dépendent que du **view-model** déjà posé par les phases A (socle `core/` + `Layer`/`UiState`) et B (`SolvedFrame` + couche `robot`) :

| # roadmap | Couche | Donnée (view-model) | Solve-gated ? |
|---|---|---|---|
| **#3** | `layers/contacts.py` (`ContactsLayer`) | `frame.targets.robot_interaction.field` (CIBLE) vs `frame.solved.contact_achieved.field` (ATTEINT), tracées sur `frame.solved.robot_points_world` | **oui** (no-op si `solved is None`) |
| **#4** | `layers/correspondence.py` (`CorrespondenceLayer`) | lignes SMPL↔G1 : `frame.human_cloud_world[smpl_idx]` → `frame.solved.robot_points_world`, appariées par `ctx.correspondence` | **oui** (no-op si `solved is None`) |
| **#6** | `layers/sdf_iso.py` (`SdfIsoLayer`) | bande iso (≈ surface) des SDF de canaux `ctx.channels[c].sdf`, posées par `frame.pose` | non (pré-solve OK) |
| **#7** | `layers/geodesic.py` (`GeodesicLayer`) | `ctx.channels[c].geodesic` (points/normales + heat géodésique mono-source), posées par `frame.pose` | non (pré-solve OK) |

**HORS PÉRIMÈTRE — roadmap #5 « activité des contraintes » : BLOQUÉ.** Afficher l'activité par-contrainte (slack) exige que `solve/` exporte le slack par-contrainte dans `Step`/`FrameInfo` — un **changement de contrat de `solve`**, pas une tâche viz. `SolvedFrame` ne porte aujourd'hui aucun slack par-contrainte ; tant que ce contrat n'a pas bougé, #5 ne peut pas être une couche. On le NOTE ici et on n'y touche pas.

**Architecture :** chaque couche = **un fichier** `viz/layers/x.py` exposant (1) une/des **fonction(s) pure(s) numpy** (géométrie d'affichage : segments d'appariement, extraction de bande iso, normalisation géodésique — *testées en unitaire*) et (2) une **classe `Layer`** mince (handles viser persistants au `setup`, simple affectation `.points/.colors/.visible` en `update`, lecture du SEUL view-model). `app.py` les **enregistre** dans sa liste de couches (1 ligne chacune) ; chaque couche crée son propre dossier GUI + checkbox dans son `setup`. Phase C **dépend de A et B mergées** : on CONSOMME leurs types par nom canonique (`VizContext`, `VizFrame`, `SolvedFrame`, `Layer`, `UiState`, `core.colors`) sans les redéfinir.

**Pré-requis view-model (Task 0) :** les 4 couches ont besoin d'assets statiques `prepare` qui ne sont PAS encore sur `VizContext` (Phase A n'y met que `channel_names`). Task 0 **ajoute** à `VizContext` les deux champs `channels: tuple[Channel, ...]` et `correspondence: CorrespondenceTable`, et fait que `BakeSource` les **peuple** depuis l'`InteractionContext` de `prepare` (`ic.channels`, `ic.correspondence`). C'est une modification petite et explicite de `model.py` + `sources.py` ; tout le reste de Phase C consomme ces deux champs. (`ctx.channels[c]` porte `sdf` *et* `geodesic` *et* `object_idx` → couvre #6, #7 et le binding de pose des canaux d'un seul coup ; `ctx.correspondence` porte `smpl_idx`/`link_idx` pour #4.)

**Tech Stack :** Python, numpy (compute float64, stockage/affichage float32), viser (confiné aux `layers/`, `core/viser_ops`, `app.py`), pytest. Env python :
`~/.holonew_deps/miniconda3/envs/holonew/bin/python`. Tests lancés **depuis `HoloV2/`**.

## Global Constraints (verbatim)

- code/comments/docstrings **ENGLISH** ... *correction maison* : conformément à `HoloV2/CLAUDE.md`, les **docstrings + commentaires inline du code sous `src/` sont rédigés en FRANÇAIS** (seuls les fichiers `docs/` — ce plan — sont hors de cette règle). Les **noms de symboles** restent anglais. (La consigne « code/comments/docstrings ENGLISH » du cadrage est ici subordonnée à CLAUDE.md, source de vérité du dépôt.)
- viz = **consommateur pur** : viser confiné à `layers/` + `core/viser_ops` (jamais dans `model.py`/`sources.py`, qui restent numpy-only/testables sans écran).
- les couches lisent **UNIQUEMENT** le view-model (`VizContext` au `setup`, `VizFrame`+`UiState` à l'`update`) ; jamais un contrat pipeline direct, jamais une autre couche, **aucune** logique de calcul du pipeline (`targets`/`solve` **inchangés**).
- tests dans **`HoloV2/tests/`**, via l'**env python `holonew`**, `max_frames` **bas** (cf. mémoire `run-tests-low-max-frames`).
- commits **conventionnels, français**. **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic). Auteur : `Guillaume-Bsst`.
- imports **relatifs** dans `src/` (`from ..core import colors`) ; **absolus** dans `tests/` (`from src.viz...`).
- une couche **solve-gated** est **no-op** (handles `.visible = False`) quand `frame.solved is None` ; jamais de crash.

---

## Canonical types CONSUMED (ne pas redéfinir)

Posés par les phases A/B (mergées) et `prepare`/`targets` :

- `src/viz/model.py` : `VizContext` (statique), `VizFrame` (par-frame, `frozen`), `SolvedFrame` (post-solve).
  - `VizFrame.pose: FramePose` · `VizFrame.human_cloud_world: (N,3)` · `VizFrame.human_field: MultiChannelField` · `VizFrame.targets: FrameTargets` · `VizFrame.solved: SolvedFrame | None`.
  - `SolvedFrame.robot_points_world: (M,3)` · `SolvedFrame.contact_achieved: ContactEval` · `SolvedFrame.object_poses: (N,7)` · `SolvedFrame.link_transforms: (L,4,4)`.
  - `VizContext` (Phase A) : `channel_names: tuple[str,...]` · `margin: float` · `style_link_names` · `smpl_faces` · `smpl_parents` · `n_objects: int` · `robot_urdf_path` · `has_solve: bool`. **Task 0 ajoute** `channels: tuple[Channel,...]` · `correspondence: CorrespondenceTable`.
- `src/viz/core/layer.py` : `Layer` (attribut `folder: str` ; `setup(self, server, gui, ctx: VizContext) -> None` ; `update(self, frame: VizFrame, ui: UiState) -> None`) ; `UiState` (lecture seule : `channel: str`, `color_mode: str`, `point_size: float`).
- `src/viz/core/colors.py` : `heat_distance(dist, margin) -> (P,3) uint8` · `diverging(dist, vmax) -> (P,3) uint8` · `active_mask(active) -> (P,3) uint8`.
- `src/prepare/contracts.py` : `Channel(name, object_idx, sdf, geodesic)` · `SDF(grid (Nx,Ny,Nz), witness (...,3), origin (3,), spacing, name)` · `GeodesicTable(points (P,3), normals (P,3), geo (P,P), name, sampling_id)` · `CorrespondenceTable(smpl_idx (M,), link_idx (M,), offset_local (M,3), link_names, smpl_sampling_id)` · `InteractionContext(channels, correspondence, ...)`.
- `src/targets/contracts.py` : `MultiChannelField(distance (C,P), direction (C,P,3), witness (C,P,3), active (C,P), channels)` · `ContactEval(field: MultiChannelField, ...)` · `RobotInteractionTargets(field: MultiChannelField)` · `FramePose(object_rot (N,3,3), object_pos (N,3), ...)`.

---

## File Structure

```
HoloV2/
  src/viz/
    model.py            # MODIFY (Task 0) : VizContext += channels, correspondence
    sources.py          # MODIFY (Task 0) : BakeSource peuple channels/correspondence
    app.py              # MODIFY (Task 5) : enregistre les 4 couches
    layers/
      contacts.py       # CREATE (Task 1) : contact_colors() + ContactsLayer        (#3)
      correspondence.py # CREATE (Task 2) : correspondence_segments() + CorrespondenceLayer  (#4)
      sdf_iso.py        # CREATE (Task 3) : iso_band_points() + SdfIsoLayer          (#6)
      geodesic.py       # CREATE (Task 4) : geo_normalized()/geo_heat_colors() + GeodesicLayer  (#7)
  tests/
    test_viz_context_interaction.py        # CREATE (Task 0)
    test_viz_contacts.py                    # CREATE (Task 1)
    test_viz_correspondence.py              # CREATE (Task 2)
    test_viz_sdf_iso.py                     # CREATE (Task 3)
    test_viz_geodesic.py                    # CREATE (Task 4)
    test_viz_interaction_layers_import.py   # CREATE (Task 5)
  docs/VIZ.md           # MODIFY (Task 5) : section « couches d'interaction (roadmap #3/#4/#6/#7) »
```

**Convention de test :** la **géométrie d'affichage pure** (segments, masque de bande, normalisation) est testée en unitaire **screen-free, torch-free** (numpy seul) — entrées connues → sorties connues. Les **classes `Layer`** (effets viser) ne sont PAS testées au rendu ; elles reçoivent un test **structurel** (importable, instanciable, `folder` est un `str`, conformité au protocole) + une **vérification viser manuelle** dont l'attendu est écrit explicitement.

---

### Task 0 : `VizContext` += `channels` + `correspondence` ; `BakeSource` les peuple

**Files:**
- Modify: `src/viz/model.py`
- Modify: `src/viz/sources.py`
- Test: `tests/test_viz_context_interaction.py`

**Interfaces:**
- Consumes : `Channel`, `CorrespondenceTable`, `InteractionContext` (`prepare.contracts`).
- Produces : `VizContext.channels: tuple[Channel, ...]`, `VizContext.correspondence: CorrespondenceTable` ; `BakeSource` qui les remplit depuis `ic.channels` / `ic.correspondence`.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_viz_context_interaction.py
"""VizContext porte désormais les assets statiques d'interaction (channels + correspondance) que les
couches contacts/correspondence/sdf_iso/geodesic consomment. Deux niveaux :
  - introspection (toujours exécutée) : les deux champs EXISTENT sur la dataclass ;
  - déterminisme/forme (gated data) : BakeSource les peuple identiquement build-vs-build, alignés sur
    channel_names / correspondence.n_points."""
import dataclasses

import numpy as np
import pytest

from src.viz.model import VizContext


def test_vizcontext_declares_interaction_fields():
    names = {f.name for f in dataclasses.fields(VizContext)}
    assert "channels" in names, "VizContext must carry the prepare Channels (sdf+geodesic+object_idx)"
    assert "correspondence" in names, "VizContext must carry the SMPL<->robot CorrespondenceTable"


# --- déterminisme/forme via BakeSource sur la scène démo (gated data, max_frames bas) ---
_DATA = pytest.importorskip  # lisibilité du gate ci-dessous


def _demo_spec():
    """SceneSpec démo minimal (HODome) ; skip propre si les données/le modèle sont absents."""
    from datapaths import HODOME, SMPLX_MODELS
    from src.viz._scene_args import demo_scene_spec  # helper de cadrage A/B (déjà câblé par app.py)
    if not HODOME.exists() or not SMPLX_MODELS.exists():
        pytest.skip("HODome/SMPL-X absent -> bake gated")
    return demo_scene_spec()


def test_bakesource_populates_interaction_context_deterministic():
    from src.viz.sources import BakeSource
    spec = _demo_spec()
    s1 = BakeSource(spec, solve=False, frame_step=8, max_frames=2)
    s2 = BakeSource(spec, solve=False, frame_step=8, max_frames=2)
    c1, c2 = s1.context, s2.context
    # alignement : channels <-> channel_names ; correspondence M-points cohérent
    assert tuple(ch.name for ch in c1.channels) == tuple(c1.channel_names)
    assert c1.correspondence.n_points == c1.correspondence.smpl_idx.shape[0]
    # déterminisme : mêmes noms de canaux, même appariement
    assert tuple(ch.name for ch in c1.channels) == tuple(ch.name for ch in c2.channels)
    assert np.array_equal(c1.correspondence.smpl_idx, c2.correspondence.smpl_idx)
    assert np.array_equal(c1.correspondence.link_idx, c2.correspondence.link_idx)
```

> NOTE d'intégration A/B : `demo_scene_spec()` / `BakeSource(spec, solve=..., frame_step=..., max_frames=...)` / `BakeSource.context` sont les surfaces posées par les phases A/B. Si leurs noms diffèrent au merge (ex. `BakeSource(spec).context` sans kwargs), ajuster CE test seul — la logique (deux champs présents + peuplés déterministes) est inchangée. Le 1ᵉʳ test (introspection) ne dépend d'aucun de ces noms et reste le garde-fou dur.

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_context_interaction.py -q`
Expected: FAIL (`test_vizcontext_declares_interaction_fields` : `channels`/`correspondence` absents de `VizContext`).

- [ ] **Step 3 : Modifier `src/viz/model.py` — ajouter les deux champs à `VizContext`**

Ajouter l'import (en tête des imports `prepare`) :

```python
from ..prepare.contracts import Channel, CorrespondenceTable
```

Dans la dataclass `VizContext`, ajouter les deux champs APRÈS `n_objects` (assets statiques d'interaction) :

```python
    # --- assets statiques d'interaction (consommés par les couches contacts/correspondence/sdf_iso/
    # geodesic) : on porte les Channel COMPLETS (chaque Channel tient son SDF, sa GeodesicTable et son
    # object_idx -> binding de pose), plus l'appariement SMPL<->robot. channel_names en reste dérivable.
    channels: tuple[Channel, ...]           # ground (object_idx=None) + un par objet, ordre prepare
    correspondence: CorrespondenceTable     # appariement statique SMPL<->robot (smpl_idx / link_idx)
```

> Si `VizContext` a des champs à valeur par défaut, insérer `channels`/`correspondence` AVANT eux (les champs sans défaut ne peuvent pas suivre un champ à défaut). `has_solve`/`robot_urdf_path` sont typiquement les derniers ; placer ces deux-là juste avant.

- [ ] **Step 4 : Modifier `src/viz/sources.py` — `BakeSource` peuple les deux champs**

Là où `BakeSource` construit son `VizContext` (il détient déjà l'`InteractionContext` de `prepare`, nommé `ic` / `self.ctx` selon A), passer les deux champs depuis l'`InteractionContext` :

```python
        context = VizContext(
            # ... champs Phase A inchangés (channel_names=ic.channel_names, margin=ic.margin, ...) ...
            channels=ic.channels,                 # Channel complets (sdf + geodesic + object_idx)
            correspondence=ic.correspondence,     # appariement SMPL<->robot statique
        )
```

> `InteractionContext.channels` et `InteractionContext.correspondence` existent déjà (cf. `prepare/contracts.py`) — aucune recomputation, simple ré-exposition sur le view-model (golden rule : consommateur pur). Aucune dépendance torch ajoutée (`Channel`/`CorrespondenceTable` sont numpy-only).

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_context_interaction.py -q`
Expected: PASS (introspection toujours ; déterminisme PASS si données présentes, SKIP sinon).

Run (numpy-only au chargement) : `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz.model; assert 'viser' not in sys.modules and 'torch' not in sys.modules; print('model numpy-only ok')"`
Expected: `model numpy-only ok`

- [ ] **Step 6 : Commit**

```bash
git add src/viz/model.py src/viz/sources.py tests/test_viz_context_interaction.py
git commit -m "feat(holov2): viz/model — VizContext porte channels + correspondence (assets statiques d'interaction), peuplés par BakeSource"
```

---

### Task 1 : `layers/contacts.py` — contact CIBLE vs ATTEINT (roadmap #3)

**Files:**
- Create: `src/viz/layers/contacts.py`
- Test: `tests/test_viz_contacts.py`

**Interfaces:**
- Consumes : `VizFrame` (`targets.robot_interaction.field`, `solved.contact_achieved.field`, `solved.robot_points_world`), `UiState` (`channel`, `color_mode`, `point_size`), `VizContext` (`channel_names`, `margin`), `core.colors`.
- Produces : `contact_colors(field, channel_idx, mode, margin) -> (M,3) uint8` (pure) ; `ContactsLayer(Layer)`.

Logique : les **M** points de contrôle robot ont (a) un champ de contact CIBLE transporté (`targets.robot_interaction.field`, `(C,M)`) et (b) un champ ATTEINT à `q` résolu (`solved.contact_achieved.field`, `(C,M)`), alignés sur les MÊMES M points `solved.robot_points_world`. On colore ces points par le canal sélectionné (distance heatmap / masque actif / uniforme). **Pas de witness/normale** (les points sont déjà en monde → aucune élévation local→monde, couche mince). **No-op si `solved is None`** (sans `q` résolu les M points n'ont pas de position monde).

- [ ] **Step 1 : Écrire le test (logique pure `contact_colors`)**

```python
# tests/test_viz_contacts.py
"""contact_colors : un MultiChannelField (C,M) + un canal + un mode -> (M,3) uint8. Screen-free,
torch-free. On vérifie forme/dtype + que 'distance' suit le champ (deux distances differentes ->
deux couleurs differentes) et que 'active' separe actifs/inactifs."""
import numpy as np

from src.targets.contracts import MultiChannelField
from src.viz.layers.contacts import contact_colors


def _field(C=2, M=4):
    dist = np.zeros((C, M)); dist[0] = np.array([0.0, 0.02, 0.05, 0.09])   # canal 0 varie
    direction = np.zeros((C, M, 3)); direction[..., 2] = 1.0
    witness = np.zeros((C, M, 3))
    active = np.zeros((C, M), bool); active[0] = np.array([True, True, False, False])
    return MultiChannelField(distance=dist, direction=direction, witness=witness,
                             active=active, channels=tuple(f"c{i}" for i in range(C)))


def test_distance_mode_shape_dtype_and_varies():
    f = _field()
    cols = contact_colors(f, channel_idx=0, mode="distance", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    # distances differentes -> au moins deux couleurs distinctes
    assert len({tuple(c) for c in cols}) >= 2


def test_active_mode_splits_active_inactive():
    f = _field()
    cols = contact_colors(f, channel_idx=0, mode="active", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    # les 2 actifs partagent une couleur, distincte de celle des 2 inactifs
    assert tuple(cols[0]) == tuple(cols[1])
    assert tuple(cols[2]) == tuple(cols[3])
    assert tuple(cols[0]) != tuple(cols[2])


def test_uniform_mode_single_color():
    f = _field()
    cols = contact_colors(f, channel_idx=1, mode="uniform", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    assert len({tuple(c) for c in cols}) == 1
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_contacts.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.contacts`).

- [ ] **Step 3 : Écrire `src/viz/layers/contacts.py`**

```python
"""Couche contacts (roadmap #3) — contact CIBLE vs ATTEINT sur les M points de contrôle robot.

Les M points de correspondance robot portent deux champs de contact alignés (canal-first ``(C, M)``)
sur les MÊMES positions monde ``solved.robot_points_world`` :
  - CIBLE   : ``frame.targets.robot_interaction.field`` (le champ humain transporté sur le robot) ;
  - ATTEINT : ``frame.solved.contact_achieved.field``   (réévalué à la config ``q`` résolue).
On colore les points par le canal sélectionné (heatmap distance / masque actif / uniforme), un nuage
pour chaque champ (toggle indépendant). Couche SOLVE-GATED : sans ``solved`` les M points n'ont pas de
position monde -> masquée. Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.layer import UiState
from ..model import VizContext, VizFrame
from ...targets.contracts import MultiChannelField

# teintes uniformes (uint8 RGB) quand mode == "uniform"
_TARGET_RGB = np.array([255, 170, 0], np.uint8)     # cible : orange
_ACHIEVED_RGB = np.array([0, 200, 120], np.uint8)   # atteint : vert


def contact_colors(field: MultiChannelField, channel_idx: int, mode: str, margin: float,
                   *, uniform_rgb: np.ndarray = _TARGET_RGB) -> np.ndarray:
    """(M, 3) uint8 : colore les M points par le canal ``channel_idx`` de ``field``.
    ``mode`` : 'distance' (heatmap signée), 'active' (masque booleen), sinon 'uniform'."""
    if mode == "distance":
        return colors.heat_distance(field.distance[channel_idx], margin)
    if mode == "active":
        return colors.active_mask(field.active[channel_idx])
    M = field.distance.shape[1]
    return np.tile(np.asarray(uniform_rgb, np.uint8), (M, 1))


class ContactsLayer:
    """``Layer`` : deux nuages (cible / atteint) sur ``solved.robot_points_world``, colores par le
    canal+mode globaux. No-op si ``frame.solved is None``."""

    folder = "Contacts (robot)"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._ctx = ctx
        with gui.add_folder(self.folder):
            self._cb_target = gui.add_checkbox("contact cible", True)
            self._cb_achieved = gui.add_checkbox("contact atteint", True)
        zero = np.zeros((1, 3), np.float32)
        self._h_target = server.scene.add_point_cloud(
            "/contacts/target", zero, np.zeros((1, 3), np.uint8), point_size=0.014)
        self._h_achieved = server.scene.add_point_cloud(
            "/contacts/achieved", zero, np.zeros((1, 3), np.uint8), point_size=0.014)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        if frame.solved is None:                       # solve-gated -> masquer
            self._h_target.visible = False
            self._h_achieved.visible = False
            return
        c = self._ctx.channel_names.index(ui.channel)
        pts = np.asarray(frame.solved.robot_points_world, np.float32)   # (M, 3) monde
        margin = float(self._ctx.margin)
        sz = float(ui.point_size)

        tgt = frame.targets.robot_interaction.field                    # cible (C, M)
        self._h_target.points = pts
        self._h_target.colors = contact_colors(tgt, c, ui.color_mode, margin, uniform_rgb=_TARGET_RGB)
        self._h_target.point_size = sz
        self._h_target.visible = bool(self._cb_target.value)

        ach = frame.solved.contact_achieved.field                      # atteint (C, M)
        self._h_achieved.points = pts
        self._h_achieved.colors = contact_colors(ach, c, ui.color_mode, margin,
                                                  uniform_rgb=_ACHIEVED_RGB)
        self._h_achieved.point_size = sz
        self._h_achieved.visible = bool(self._cb_achieved.value)
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_contacts.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5 : Vérification viser MANUELLE** (après Task 5 — couche câblée dans `app.py`)

À faire au moment où `app.py` enregistre la couche (Task 5). Libérer le port d'abord (`fuser -k 8080/tcp`), lancer le viewer prod **avec solve activé** sur la scène démo, ouvrir `http://localhost:8080`, dossier **« Contacts (robot) »**. ATTENDU :
  - cocher **contact atteint** : un nuage de M points sur le robot résolu, coloré par le canal sélectionné en mode **distance** (bleu = proche/pénétrant, rouge = loin dans la marge) ;
  - basculer le sélecteur **canal** sur `ground` → les points en contact pied/sol passent au bleu ; sur `obj0` → les points main/objet passent au bleu ;
  - cocher **contact cible** (orange en uniforme) : se superpose aux mêmes M points (la cible que le solve VISE) ; l'écart cible↔atteint = la qualité du contact résolu ;
  - **désactiver le solve** (relancer sans solve) → le dossier est présent mais les deux nuages restent **masqués** (no-op `solved is None`), aucun crash.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/contacts.py tests/test_viz_contacts.py
git commit -m "feat(holov2): viz/layers/contacts — contact cible vs atteint sur les points robot (roadmap #3, solve-gated)"
```

---

### Task 2 : `layers/correspondence.py` — lignes SMPL↔G1 (roadmap #4)

**Files:**
- Create: `src/viz/layers/correspondence.py`
- Test: `tests/test_viz_correspondence.py`

**Interfaces:**
- Consumes : `VizFrame` (`human_cloud_world`, `solved.robot_points_world`), `VizContext` (`correspondence.smpl_idx`), `UiState`.
- Produces : `correspondence_segments(human_cloud_world, robot_points_world, smpl_idx) -> (M,2,3)` (pure) ; `CorrespondenceLayer(Layer)`.

Logique : pour chaque point apparié `m`, tracer le segment `human_cloud_world[smpl_idx[m]] -> robot_points_world[m]`. C'est l'appariement OT figé (`CorrespondenceTable`) rendu visible à la config résolue : une bonne carte donne des lignes courtes et non croisées entre la surface humaine et le robot. **No-op si `solved is None`** (pas de côté robot en monde).

- [ ] **Step 1 : Écrire le test (logique pure `correspondence_segments`)**

```python
# tests/test_viz_correspondence.py
"""correspondence_segments : (nuage humain, points robot, smpl_idx) -> (M,2,3). Screen-free, numpy.
Indices connus -> segments connus (extremite humaine = human[smpl_idx[m]], extremite robot = robot[m])."""
import numpy as np
import pytest

from src.viz.layers.correspondence import correspondence_segments


def test_segments_pick_paired_endpoints():
    human = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])   # N=4
    robot = np.array([[0., 1., 0.], [0., 2., 0.]])                                # M=2
    smpl_idx = np.array([2, 0])           # robot0 <-> human2 ; robot1 <-> human0
    seg = correspondence_segments(human, robot, smpl_idx)
    assert seg.shape == (2, 2, 3) and seg.dtype == np.float32
    assert np.allclose(seg[0, 0], [2., 0., 0.]) and np.allclose(seg[0, 1], [0., 1., 0.])
    assert np.allclose(seg[1, 0], [0., 0., 0.]) and np.allclose(seg[1, 1], [0., 2., 0.])


def test_empty_when_no_points():
    seg = correspondence_segments(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), np.int64))
    assert seg.shape == (0, 2, 3)


def test_out_of_range_index_raises():
    human = np.zeros((2, 3)); robot = np.zeros((1, 3)); smpl_idx = np.array([5])   # 5 >= N=2
    with pytest.raises((IndexError, ValueError)):
        correspondence_segments(human, robot, smpl_idx)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_correspondence.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.correspondence`).

- [ ] **Step 3 : Écrire `src/viz/layers/correspondence.py`**

```python
"""Couche correspondance (roadmap #4) — lignes SMPL↔G1 à la config résolue.

L'appariement OT figé (``ctx.correspondence``) associe à chaque point de contrôle robot ``m`` un point
de la surface SMPL (``smpl_idx[m]``). On trace le segment ``human_cloud_world[smpl_idx[m]] ->
solved.robot_points_world[m]`` : la carte humain→robot rendue visible (lignes courtes, non croisees =
bonne carte). Couche SOLVE-GATED : sans ``solved`` le côté robot n'a pas de position monde -> masquee.
Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame

_LINE_RGB = np.array([120, 220, 230], np.uint8)     # cyan doux


def correspondence_segments(human_cloud_world: np.ndarray, robot_points_world: np.ndarray,
                            smpl_idx: np.ndarray) -> np.ndarray:
    """(M, 2, 3) f32 : segment[m] = (human_cloud_world[smpl_idx[m]], robot_points_world[m]).
    Lève si un ``smpl_idx`` sort du nuage humain (appariement incoherent)."""
    human = np.asarray(human_cloud_world, np.float64)
    robot = np.asarray(robot_points_world, np.float64)
    idx = np.asarray(smpl_idx, np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= human.shape[0]):
        raise ValueError(f"smpl_idx out of range [0, {human.shape[0]})")
    a = human[idx]                                   # (M, 3) extremite humaine
    b = robot                                        # (M, 3) extremite robot
    return np.stack([a, b], axis=1).astype(np.float32)


class CorrespondenceLayer:
    """``Layer`` : un handle de segments persistant, MAJ par frame depuis l'appariement. No-op si
    ``frame.solved is None``."""

    folder = "Correspondance SMPL↔G1"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._smpl_idx = np.asarray(ctx.correspondence.smpl_idx, np.int64)
        with gui.add_folder(self.folder):
            self._cb = gui.add_checkbox("lignes SMPL↔G1", True)
        self._h = server.scene.add_line_segments(
            "/correspondence", np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8),
            line_width=1.5)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        if frame.solved is None or not bool(self._cb.value):
            self._h.visible = False
            return
        seg = correspondence_segments(frame.human_cloud_world, frame.solved.robot_points_world,
                                      self._smpl_idx)                      # (M, 2, 3)
        col = np.broadcast_to(_LINE_RGB, (seg.shape[0], 2, 3)).astype(np.uint8)
        self._h.points = seg
        self._h.colors = col
        self._h.visible = True
```

> NOTE viser : selon la version de viser, l'attribut des segments est `.points` (ou `.line_points`). Aligner sur le helper `core.viser_ops.line_segments` posé par Phase A (qui encapsule déjà l'add + l'attribut) si l'affectation directe ne s'applique pas — la logique pure (`correspondence_segments`) reste identique.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_correspondence.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5 : Vérification viser MANUELLE** (après Task 5)

Viewer prod **avec solve**, dossier **« Correspondance SMPL↔G1 »**, cocher **lignes SMPL↔G1**. ATTENDU :
  - M segments cyan reliant la surface humaine (côté SMPL) au robot résolu (côté G1) ;
  - les lignes sont **courtes** et **non croisées** sur un appariement sain (mains↔mains, pieds↔pieds) ; un faisceau qui se croise/s'allonge localement = défaut de carte à inspecter ;
  - **sans solve** → dossier présent, lignes masquées (no-op), aucun crash.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/correspondence.py tests/test_viz_correspondence.py
git commit -m "feat(holov2): viz/layers/correspondence — lignes SMPL↔G1 à la config résolue (roadmap #4, solve-gated)"
```

---

### Task 3 : `layers/sdf_iso.py` — iso-surface des SDF de canaux (roadmap #6)

**Files:**
- Create: `src/viz/layers/sdf_iso.py`
- Test: `tests/test_viz_sdf_iso.py`

**Interfaces:**
- Consumes : `VizContext` (`channels[c].sdf`, `channels[c].object_idx`, `margin`), `VizFrame` (`pose.object_rot`, `pose.object_pos`), `UiState`, `core.colors.diverging`.
- Produces : `iso_band_points(sdf, band) -> (points_local (K,3) f64, dist (K,) f64)` (pure) ; `SdfIsoLayer(Layer)`.

Logique (portée de `viz/sdf.py`) : la « iso-surface » est approchée par la **bande proche-surface** `|d| < band` des nœuds de grille, colorée par distance signée (divergent : bleu intérieur / blanc surface / rouge extérieur) — le passage par zéro trace la surface. Les points sont en **frame locale du canal** ; on les élève en monde par la pose per-frame du canal (`object_idx is None` → monde directement ; sinon `pose.object_rot/pos[object_idx]`). Pré-solve OK (n'utilise pas `solved`).

- [ ] **Step 1 : Écrire le test (logique pure `iso_band_points`)**

```python
# tests/test_viz_sdf_iso.py
"""iso_band_points : un SDF -> (points locaux, distances) des noeuds dans la bande |d|<band. Grille
3x3x3 connue (un plan z=centre) -> seule la tranche centrale (d=0) est dans une bande etroite."""
import numpy as np

from src.prepare.contracts import SDF
from src.viz.layers.sdf_iso import iso_band_points


def _plane_sdf():
    """Grille 3x3x3, spacing 1, origin (0,0,0) ; distance signee = z-1 (plan a z=1, la tranche du milieu)."""
    nx = ny = nz = 3
    grid = np.zeros((nx, ny, nz))
    for k in range(nz):
        grid[:, :, k] = (k - 1) * 1.0            # z indices 0,1,2 -> d = -1, 0, +1
    witness = np.zeros((nx, ny, nz, 3))           # contenu indifferent pour ce test
    return SDF(grid=grid, witness=witness, origin=np.zeros(3), spacing=1.0, name="plane")


def test_band_keeps_only_zero_crossing_slice():
    sdf = _plane_sdf()
    pts, dist = iso_band_points(sdf, band=0.5)     # |d|<0.5 -> seulement la tranche k=1 (d=0)
    assert pts.shape == (9, 3) and dist.shape == (9,)
    assert np.allclose(dist, 0.0)
    # tous les points de la bande sont a z = origin_z + spacing*1 = 1.0
    assert np.allclose(pts[:, 2], 1.0)


def test_wider_band_keeps_more_nodes():
    sdf = _plane_sdf()
    pts, dist = iso_band_points(sdf, band=1.5)     # |d|<1.5 -> les 3 tranches (27 noeuds)
    assert pts.shape == (27, 3)
    assert set(np.round(np.unique(dist), 6)) == {-1.0, 0.0, 1.0}
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_sdf_iso.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.sdf_iso`).

- [ ] **Step 3 : Écrire `src/viz/layers/sdf_iso.py`**

```python
"""Couche sdf_iso (roadmap #6) — iso-surface (≈ surface) des SDF de canaux dans le viewer prod.

Approche l'iso-surface par la BANDE proche-surface ``|d| < band`` des noeuds de grille, colorée par
distance signee (divergent : bleu intérieur / blanc surface / rouge extérieur) ; le passage par zero
trace la surface. Logique portée de ``viz/sdf.py`` (``_node_coords`` + masque de bande + ``_diverging``,
ici via ``core.colors.diverging``). Les points sont en frame LOCALE du canal et eleves en monde par la
pose per-frame (ground = monde ; objet = ``frame.pose.object_*[object_idx]``). Pré-solve OK (n'utilise
pas ``solved``). Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.layer import UiState
from ..model import VizContext, VizFrame
from ...prepare.contracts import SDF


def _node_coords(sdf: SDF) -> np.ndarray:
    """(Nx, Ny, Nz, 3) coords locales de chaque noeud de la grille (porté de viz/sdf.py)."""
    nx, ny, nz = sdf.grid.shape
    xs = sdf.origin[0] + sdf.spacing * np.arange(nx)
    ys = sdf.origin[1] + sdf.spacing * np.arange(ny)
    zs = sdf.origin[2] + sdf.spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def iso_band_points(sdf: SDF, band: float) -> tuple[np.ndarray, np.ndarray]:
    """(points (K,3), dist (K,)) des noeuds de la grille dont ``|d| < band`` (la coquille proche-surface
    qui matérialise l'iso ≈ surface). Points en frame LOCALE du SDF."""
    coords = _node_coords(sdf)                       # (Nx,Ny,Nz,3) locale
    mask = np.abs(sdf.grid) < float(band)            # (Nx,Ny,Nz) bool
    return coords[mask].astype(np.float64), sdf.grid[mask].astype(np.float64)


class SdfIsoLayer:
    """``Layer`` : une coquille de bande par canal, posée en monde par frame. Pré-solve OK."""

    folder = "SDF iso (surface)"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._ctx = ctx
        with gui.add_folder(self.folder):
            self._cb = gui.add_checkbox("bande iso", False)
            self._band = gui.add_number("bande (m)", float(ctx.margin), min=0.005, max=0.5, step=0.005)
        # un handle par canal (statique en nombre ; points recalculés au band/à la pose)
        self._handles = []
        for ch in ctx.channels:
            h = server.scene.add_point_cloud(
                f"/sdf_iso/{ch.name}", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                point_size=0.006)
            self._handles.append(h)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        show = bool(self._cb.value)
        band = float(self._band.value)
        for ch, h in zip(self._ctx.channels, self._handles):
            if not show:
                h.visible = False
                continue
            pts_local, dist = iso_band_points(ch.sdf, band)              # (K,3) locale, (K,)
            if ch.object_idx is None:                                    # ground : déjà en monde
                pts_world = pts_local
            else:                                                        # objet : élever par la pose
                R = np.asarray(frame.pose.object_rot[ch.object_idx], np.float64)   # (3,3)
                t = np.asarray(frame.pose.object_pos[ch.object_idx], np.float64)   # (3,)
                pts_world = pts_local @ R.T + t
            h.points = pts_world.astype(np.float32)
            h.colors = colors.diverging(dist, max(band, 1e-9))           # bleu/blanc/rouge signé
            h.visible = True
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_sdf_iso.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5 : Vérification viser MANUELLE** (après Task 5)

Viewer prod (solve ou non), dossier **« SDF iso (surface) »**, cocher **bande iso**. ATTENDU :
  - une coquille de points colorés (bleu intérieur / blanc surface / rouge extérieur) épousant chaque canal : le **sol** (plan z=0) en monde, **chaque objet** posé à sa transformation per-frame qui suit l'objet dans le temps (jouer le slider) ;
  - augmenter **bande (m)** épaissit la coquille ; la diminuer la resserre sur le passage par zéro (la surface) ;
  - décocher → tous les nuages masqués.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/sdf_iso.py tests/test_viz_sdf_iso.py
git commit -m "feat(holov2): viz/layers/sdf_iso — bande iso ≈ surface des SDF de canaux dans le viewer prod (roadmap #6)"
```

---

### Task 4 : `layers/geodesic.py` — champ géodésique (roadmap #7)

**Files:**
- Create: `src/viz/layers/geodesic.py`
- Test: `tests/test_viz_geodesic.py`

**Interfaces:**
- Consumes : `VizContext` (`channels[c].geodesic`, `channels[c].object_idx`), `VizFrame` (`pose.object_rot/pos`), `UiState`, `core.colors.heat_distance`.
- Produces : `geo_normalized(geo_row) -> (P,) f64 in [0,1]` (pure) ; `geo_heat_colors(geo_row) -> (P,3) uint8` (wrap colors) ; `GeodesicLayer(Layer)`.

Logique : pour chaque canal portant une `GeodesicTable` (objets/terrain ; `None` pour le sol-plan → couche no-op sur ce canal), afficher ses `points` (en monde via la pose du canal) colorés par le champ géodésique mono-source `geo[src]` (la ligne du point source choisi), normalisé en [0,1] puis heatmap ; et optionnellement les `normals` en courts segments. Pré-solve OK.

- [ ] **Step 1 : Écrire le test (logique pure)**

```python
# tests/test_viz_geodesic.py
"""geo_normalized / geo_heat_colors : une ligne geodesique mono-source -> [0,1] puis (P,3) uint8.
Screen-free. La source (distance 0) -> 0 ; le max -> 1 ; formes/dtype verifies."""
import numpy as np

from src.viz.layers.geodesic import geo_heat_colors, geo_normalized


def test_normalized_source_zero_max_one():
    row = np.array([0.0, 1.0, 2.0, 4.0])             # geo[src] : src lui-meme = 0
    n = geo_normalized(row)
    assert n.shape == (4,)
    assert np.isclose(n[0], 0.0) and np.isclose(n[3], 1.0)
    assert np.all((n >= 0.0) & (n <= 1.0))


def test_normalized_constant_row_is_zero():
    n = geo_normalized(np.zeros(5))                  # tout a 0 -> pas de division par 0
    assert np.allclose(n, 0.0)


def test_heat_colors_shape_dtype():
    cols = geo_heat_colors(np.array([0.0, 0.5, 1.0, 2.0]))
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_geodesic.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.geodesic`).

- [ ] **Step 3 : Écrire `src/viz/layers/geodesic.py`**

```python
"""Couche geodesic (roadmap #7) — champ géodésique des canaux objets/terrain.

Chaque canal non-plan porte une ``GeodesicTable`` (``ctx.channels[c].geodesic``) : ses ``points`` de
surface + ``normals``, et la matrice ``geo`` dont la ligne ``geo[src]`` EST le champ géodésique
mono-source depuis le point ``src``. On affiche les points (élevés en monde par la pose du canal),
colorés par ce champ mono-source normalisé en heatmap (proche = bleu, loin = rouge), source choisie par
un index GUI ; les ``normals`` sont optionnellement tracées en courts segments. Canal sol-plan
(``geodesic is None``) -> no-op. Pré-solve OK. Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.layer import UiState
from ..model import VizContext, VizFrame

_NORMAL_LEN = 0.03                                   # longueur des segments de normale (m)
_NORMAL_RGB = np.array([60, 220, 200], np.uint8)


def geo_normalized(geo_row: np.ndarray) -> np.ndarray:
    """(P,) f64 dans [0,1] : distance géodésique mono-source normalisée par son max (source -> 0).
    Ligne constante (max ~ 0) -> tout 0 (pas de division par zero)."""
    d = np.asarray(geo_row, np.float64)
    hi = float(d.max()) if d.size else 0.0
    return d / hi if hi > 1e-12 else np.zeros_like(d)


def geo_heat_colors(geo_row: np.ndarray) -> np.ndarray:
    """(P,3) uint8 : heatmap de la géodésique mono-source normalisée (proche = bleu, loin = rouge)."""
    return colors.heat_distance(geo_normalized(geo_row), 1.0)


class GeodesicLayer:
    """``Layer`` : par canal géodésique, un nuage de points (heat mono-source) + des normales. La source
    est un index GUI ; canaux sans table -> ignorés. Pré-solve OK."""

    folder = "Champ géodésique"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._ctx = ctx
        # canaux porteurs d'une GeodesicTable, avec leur object_idx (binding de pose)
        self._geo = [(ch.geodesic, ch.object_idx, ch.name)
                     for ch in ctx.channels if ch.geodesic is not None]
        max_src = max((g.n_points - 1 for g, _, _ in self._geo), default=0)
        with gui.add_folder(self.folder):
            self._cb_pts = gui.add_checkbox("points (heat géodésique)", False)
            self._cb_nrm = gui.add_checkbox("normales", False)
            self._src = gui.add_slider("point source", 0, max(max_src, 0), 1, 0)
        self._h_pts, self._h_nrm = [], []
        for _g, _oi, name in self._geo:
            self._h_pts.append(server.scene.add_point_cloud(
                f"/geodesic/{name}/pts", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                point_size=0.008))
            self._h_nrm.append(server.scene.add_line_segments(
                f"/geodesic/{name}/nrm", np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8),
                line_width=1.5))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        show_pts, show_nrm = bool(self._cb_pts.value), bool(self._cb_nrm.value)
        src = int(self._src.value)
        for (geo, oi, _name), h_pts, h_nrm in zip(self._geo, self._h_pts, self._h_nrm):
            if oi is None:                                               # canal en monde
                R, t = np.eye(3), np.zeros(3)
            else:                                                        # objet : pose per-frame
                R = np.asarray(frame.pose.object_rot[oi], np.float64)
                t = np.asarray(frame.pose.object_pos[oi], np.float64)
            pw = np.asarray(geo.points, np.float64) @ R.T + t            # (P,3) monde
            if show_pts:
                s = min(src, geo.n_points - 1)
                h_pts.points = pw.astype(np.float32)
                h_pts.colors = geo_heat_colors(geo.geo[s])              # heat mono-source
                h_pts.visible = True
            else:
                h_pts.visible = False
            if show_nrm:
                nw = np.asarray(geo.normals, np.float64) @ R.T          # normales (rotation seule)
                seg = np.stack([pw, pw + nw * _NORMAL_LEN], axis=1).astype(np.float32)   # (P,2,3)
                h_nrm.points = seg
                h_nrm.colors = np.broadcast_to(_NORMAL_RGB, (seg.shape[0], 2, 3)).astype(np.uint8)
                h_nrm.visible = True
            else:
                h_nrm.visible = False
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_geodesic.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5 : Vérification viser MANUELLE** (après Task 5)

Viewer prod sur une scène **avec objet** (canal porteur d'une `GeodesicTable`), dossier **« Champ géodésique »** :
  - cocher **points (heat géodésique)** : le nuage de surface de l'objet coloré par la géodésique depuis le **point source** (bleu près de la source → rouge au plus loin sur la surface) ; déplacer le slider **point source** fait migrer le centre bleu sur la surface (et la heat suit la géodésie, pas l'euclidien — contourne les concavités) ;
  - cocher **normales** : courts segments cyan sortant de la surface (vérifie l'orientation des normales) ;
  - sur une scène **sans objet** (sol-plan seul, `geodesic is None`) → dossier présent mais aucun nuage (no-op), aucun crash ;
  - jouer le slider de frame : sur un objet mobile, le champ suit la pose de l'objet.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/geodesic.py tests/test_viz_geodesic.py
git commit -m "feat(holov2): viz/layers/geodesic — points/normales + heat géodésique mono-source des canaux (roadmap #7)"
```

---

### Task 5 : enregistrer les 4 couches dans `app.py` + doc VIZ.md

**Files:**
- Modify: `src/viz/app.py`
- Modify: `docs/VIZ.md`
- Test: `tests/test_viz_interaction_layers_import.py`

**Interfaces:**
- Consumes : `ContactsLayer`, `CorrespondenceLayer`, `SdfIsoLayer`, `GeodesicLayer` (Tasks 1–4).
- Produces : viewer prod câblé avec les 4 couches (chaque couche crée son dossier/toggle dans son `setup`).

- [ ] **Step 1 : Écrire le test (structurel, screen-free)**

```python
# tests/test_viz_interaction_layers_import.py
"""Les 4 couches d'interaction roadmap sont importables, instanciables, conformes au protocole Layer
(attribut ``folder: str`` + ``setup``/``update`` appelables) et enregistrees par ``app.py``. Screen-free
(aucun viser construit) : on n'instancie que les classes et on inspecte ``app.py`` pour la presence des
couches dans son assemblage."""
import inspect

from src.viz.layers.contacts import ContactsLayer
from src.viz.layers.correspondence import CorrespondenceLayer
from src.viz.layers.geodesic import GeodesicLayer
from src.viz.layers.sdf_iso import SdfIsoLayer


def test_layers_conform_to_protocol():
    for cls in (ContactsLayer, CorrespondenceLayer, SdfIsoLayer, GeodesicLayer):
        layer = cls()
        assert isinstance(layer.folder, str) and layer.folder
        assert callable(layer.setup) and callable(layer.update)
        # signatures attendues : setup(server, gui, ctx) / update(frame, ui)
        assert list(inspect.signature(layer.setup).parameters)[:3] == ["server", "gui", "ctx"]
        assert list(inspect.signature(layer.update).parameters)[:2] == ["frame", "ui"]


def test_app_registers_the_four_layers():
    import src.viz.app as app
    src = inspect.getsource(app)
    for name in ("ContactsLayer", "CorrespondenceLayer", "SdfIsoLayer", "GeodesicLayer"):
        assert name in src, f"{name} not wired in app.py"
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_interaction_layers_import.py -q`
Expected: FAIL (`test_app_registers_the_four_layers` : les classes ne sont pas encore référencées dans `app.py`).

- [ ] **Step 3 : Modifier `src/viz/app.py` — importer + enregistrer les 4 couches**

Repérer la **liste de couches** assemblée par `app.py` (la liste passée au `Player` ; Phase A la construit pour les couches prod, Phase B y a ajouté `RobotLayer`). Ajouter les imports près des autres imports de couches :

```python
from .layers.contacts import ContactsLayer
from .layers.correspondence import CorrespondenceLayer
from .layers.sdf_iso import SdfIsoLayer
from .layers.geodesic import GeodesicLayer
```

Et **après** la couche `robot` (Phase B) dans la liste de couches, ajouter les 4 instances :

```python
    layers = [
        # ... couches prod (Phase A) ...
        # RobotLayer(),                     # Phase B (robot résolu)
        ContactsLayer(),                    # roadmap #3 — contact cible vs atteint (solve-gated)
        CorrespondenceLayer(),              # roadmap #4 — lignes SMPL↔G1 (solve-gated)
        SdfIsoLayer(),                      # roadmap #6 — bande iso ≈ surface des SDF
        GeodesicLayer(),                    # roadmap #7 — champ géodésique des canaux
    ]
```

> Anchor exact selon A/B : la liste peut s'appeler `layers`, être construite dans `run_app`/`build_app`, ou être une fonction `default_layers()`. Insérer les 4 instances dans CETTE collection, après `robot`. Les couches solve-gated se masquent seules quand `solved is None` — aucune garde conditionnelle nécessaire à l'enregistrement.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_interaction_layers_import.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5 : Mettre à jour `docs/VIZ.md` — section dédiée (ne pas clobber A/B)**

Ajouter À LA FIN de `docs/VIZ.md` une section nouvelle (n'écraser aucune section posée par A/B) :

```markdown
## Couches d'interaction (roadmap #3 / #4 / #6 / #7)

Couches composables ajoutées au viewer prod (`app.py`), une par fichier `layers/`, lisant le SEUL
view-model. Géométrie d'affichage pure (testée en unitaire) + classe `Layer` mince (handles viser
persistants). Les deux premières sont **solve-gated** (no-op si `frame.solved is None`).

| Couche | Fichier | Donnée (view-model) | Solve-gated |
|---|---|---|---|
| **Contacts (robot)** (#3) | `layers/contacts.py` · `ContactsLayer` | `targets.robot_interaction.field` (cible) vs `solved.contact_achieved.field` (atteint) sur `solved.robot_points_world` | oui |
| **Correspondance SMPL↔G1** (#4) | `layers/correspondence.py` · `CorrespondenceLayer` | `human_cloud_world[ctx.correspondence.smpl_idx]` → `solved.robot_points_world` | oui |
| **SDF iso (surface)** (#6) | `layers/sdf_iso.py` · `SdfIsoLayer` | bande `|d|<band` des `ctx.channels[c].sdf`, posée par `frame.pose` | non |
| **Champ géodésique** (#7) | `layers/geodesic.py` · `GeodesicLayer` | `ctx.channels[c].geodesic` (points/normales + heat mono-source `geo[src]`), posée par `frame.pose` | non |

Assets statiques consommés : `VizContext.channels` (chaque `Channel` porte `sdf` + `geodesic` +
`object_idx`) et `VizContext.correspondence` — ajoutés au view-model et peuplés par `BakeSource` depuis
l'`InteractionContext` de `prepare`.

**Roadmap #5 « activité des contraintes » : hors périmètre / BLOQUÉ.** Afficher le slack par-contrainte
exige que `solve/` l'exporte dans `Step`/`FrameInfo` (changement de contrat `solve`, pas une tâche viz) ;
tant que `SolvedFrame` ne porte pas ce slack, #5 ne peut pas être une couche.
```

- [ ] **Step 6 : Vérification viser MANUELLE globale** (toutes couches ensemble)

Libérer le port (`fuser -k 8080/tcp`), lancer le viewer prod **avec solve** sur la scène démo, `max_frames` bas. ATTENDU : les 4 dossiers GUI apparaissent (**Contacts (robot)**, **Correspondance SMPL↔G1**, **SDF iso (surface)**, **Champ géodésique**) ; activer chacun reproduit les attendus des Steps 5 des Tasks 1–4 ; jouer le slider de frame ne provoque aucun flicker ni crash ; relancer **sans solve** masque proprement les deux couches solve-gated.

- [ ] **Step 7 : Commit**

```bash
git add src/viz/app.py docs/VIZ.md tests/test_viz_interaction_layers_import.py
git commit -m "feat(holov2): viz/app — enregistre contacts/correspondence/sdf_iso/geodesic + VIZ.md (roadmap #3/#4/#6/#7)"
```

---

## Self-Review

**1. Couverture du spec (roadmap #3, #4, #6, #7 ; #5 noté hors périmètre) :**
- #3 contacts → Task 1 (`ContactsLayer` + `contact_colors` testé). ✅
- #4 correspondance → Task 2 (`CorrespondenceLayer` + `correspondence_segments` testé). ✅
- #6 sdf_iso → Task 3 (`SdfIsoLayer` + `iso_band_points` testé). ✅
- #7 geodesic → Task 4 (`GeodesicLayer` + `geo_normalized`/`geo_heat_colors` testés). ✅
- #5 activité des contraintes → **NOTÉ hors périmètre / bloqué** (intro + VIZ.md), aucune tâche. ✅
- Enregistrement `app.py` + toggles → Task 5 ; chaque couche crée son propre dossier/checkbox dans `setup`. ✅
- Asset view-model manquant ajouté → Task 0 (`VizContext.channels` + `.correspondence`, peuplés par `BakeSource`), avec test déterminisme/forme. ✅

**2. Scan placeholders :** aucun `TODO`/`TBD`/`...` dans le code livré ; chaque step porte du code/des commandes réels. Les seules « inconnues » sont les **anchors A/B** (liste de couches d'`app.py`, kwargs de `BakeSource`, nom `demo_scene_spec`), explicitement signalées comme points d'ajustement au merge — pas des placeholders de logique. ✅

**3. Cohérence des noms de types :** `VizContext`/`VizFrame`/`SolvedFrame`/`Layer`/`UiState` (Phase A/B), `MultiChannelField`/`ContactEval`/`RobotInteractionTargets`/`FramePose` (`targets`), `Channel`/`SDF`/`GeodesicTable`/`CorrespondenceTable`/`InteractionContext` (`prepare`), `core.colors.{heat_distance,diverging,active_mask}` — consommés tels quels, jamais redéfinis. Champs lus exactement comme aux contrats (`solved.robot_points_world`, `solved.contact_achieved.field`, `targets.robot_interaction.field`, `correspondence.smpl_idx`, `channels[c].sdf/.geodesic/.object_idx`, `pose.object_rot/object_pos`, `geo.points/.normals/.geo`). Classes produites : `ContactsLayer`/`CorrespondenceLayer`/`SdfIsoLayer`/`GeodesicLayer` + helpers purs `contact_colors`/`correspondence_segments`/`iso_band_points`/`geo_normalized`/`geo_heat_colors`. ✅

**4. Invariants viz :** viser confiné aux `layers/`+`app.py` ; `model.py`/`sources.py` restent numpy-only (Task 0 Step 5 l'asserte) ; couches lisent uniquement le view-model ; `targets`/`solve` inchangés ; couches solve-gated no-op si `solved is None`. Tests dans `HoloV2/tests/`, env `holonew`, `max_frames` bas, commits français sans tag Claude. ✅
