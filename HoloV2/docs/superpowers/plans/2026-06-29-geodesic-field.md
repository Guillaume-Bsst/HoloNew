# Champ géodésique de surface précalculé — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Précalculer offline, par mesh (objet / sol-terrain), une table all-pairs de distances géodésiques sur les points du `object_cloud`, exposée au solveur via un helper de query différentiable lisant le champ mono-source à un `witness(q)` continu.

**Architecture:** Nouveau sous-module `prepare/geodesic/` (frère de `sdf/`), `AssetBuilder` scopé géométrie produisant un contrat `GeodesicTable` (frozen, numpy) rattaché à `Channel.geodesic`. Calcul = graphe k-NN gaté par normales + Dijkstra all-pairs (scipy). Query online = MLS degré-1 (valeur + gradient tangent), dans `targets`, numpy-only.

**Tech Stack:** Python 3.11, numpy (cœur + query, torch-free), scipy (`cKDTree`/`csgraph` dans le builder), trimesh (sampling dans le builder), pytest. Spec : `docs/superpowers/specs/2026-06-29-geodesic-field-design.md`.

## Global Constraints

- **Python de l'env** : `PY="$HOME/.holonew_deps/miniconda3/envs/holonew/bin/python"` ; toutes les commandes pytest le préfixent, **lancées depuis `HoloV2/`**.
- **Imports** : DANS `src/` relatifs, n'importer que la **surface publique** de l'amont (jamais un sous-module interne) ; tests en **absolu** (`from src.… import …`).
- **Contrats** : dataclasses `frozen`, **numpy-only** ; valider les invariants par `raise ValueError` explicite (jamais `assert`). Config = `frozen`, stdlib-only, valider les plages en `__post_init__`.
- **dtype** : compute `float64` ; arrays **stockés/cachés** `float32`. **Poids des imports** : `scipy`/`trimesh` UNIQUEMENT dans le builder `prepare/geodesic/build.py` ; `targets/interaction/geodesic.py` **numpy-only / torch-free**.
- **Cache** : `cache_key = hash(sous-config pertinente + géométrie)` ; un knob ne change que son asset.
- **Commits** : conventionnels, auteur `Guillaume-Bsst <guibesset@free.fr>` ; **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic).
- **Hors scope (utilisateur)** : le résidu de coût géodésique lui-même (assemblage `geo_value_grad → résidu`, pondération, Jacobiennes).

---

### Task 1: `GeodesicConfig` + intégration `PrepareConfig`

**Files:**
- Modify: `src/prepare/config.py` (ajouter `GeodesicConfig` ; `PrepareConfig += geodesic`)
- Test: `tests/test_geodesic_config.py`

**Interfaces:**
- Produces: `GeodesicConfig(k_neighbors:int=8, normal_gate:float=-0.5, max_points:int=6000)` (frozen, `__post_init__` valide les plages) ; `PrepareConfig.geodesic: GeodesicConfig`.

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_config.py` :
```python
"""GeodesicConfig: defaults sensés, validation des plages, et présence dans PrepareConfig (un seul
objet de knobs ; override inline). Le sampling (densité/seed) n'est PAS ici — il vient de CloudConfig."""
import pytest

from src.prepare.config import GeodesicConfig, PrepareConfig


def test_defaults():
    c = GeodesicConfig()
    assert c.k_neighbors == 8 and c.normal_gate == -0.5 and c.max_points == 6000


def test_prepareconfig_includes_geodesic_default():
    assert isinstance(PrepareConfig().geodesic, GeodesicConfig)
    # override inline, sans toucher aux autres sous-configs
    p = PrepareConfig(geodesic=GeodesicConfig(k_neighbors=12))
    assert p.geodesic.k_neighbors == 12
    assert p.sdf.spacing == 0.01            # défaut SdfConfig intact


@pytest.mark.parametrize("kwargs", [
    {"k_neighbors": 0}, {"normal_gate": 1.5}, {"normal_gate": -2.0}, {"max_points": 0},
])
def test_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        GeodesicConfig(**kwargs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'GeodesicConfig'`.

- [ ] **Step 3: Write minimal implementation**

Dans `src/prepare/config.py`, ajouter après `CorrespondenceConfig` :
```python
@dataclass(frozen=True)
class GeodesicConfig:
    """Graphe géodésique sur le cloud objet (``prepare/geodesic``). La DENSITÉ/seed ne sont PAS ici :
    la géodésique réutilise l'échantillonnage du ``object_cloud`` (``CloudConfig``) — un seul sampling
    canonique partagé par le cloud ET la table géodésique."""

    k_neighbors: int = 8       # k du graphe k-NN de surface (Dijkstra all-pairs scipy)
    normal_gate: float = -0.5  # arête i--j seulement si dot(n_i, n_j) > normal_gate, dans [-1, 1].
                               # DÉFAUT -0.5 : garde les faces perpendiculaires adjacentes (dot=0, ex.
                               # arêtes d'un cube) ET coupe les arêtes quasi-opposées (dot≈-1, ex.
                               # traversée d'une plaque fine). 0.0 scinderait un cube en 6 faces ;
                               # -1.0 ≈ aucun gating.
    max_points: int = 6000     # garde-fou : ValueError si P dépasse (stockage 4*P^2 octets) —
                               # baisser object_density ou relever ce knob en conscience

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {self.k_neighbors}")
        if not -1.0 <= self.normal_gate <= 1.0:
            raise ValueError(f"normal_gate must be in [-1, 1], got {self.normal_gate}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")
```

Dans la dataclass `PrepareConfig`, ajouter le champ (après `correspondence`) :
```python
    geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)
```
Et compléter sa docstring : `cloud` alimente le human cloud, la correspondence ET la géodésique (même sampling).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_config.py -q`
Expected: PASS (5 cas).

- [ ] **Step 5: Commit**

```bash
git add src/prepare/config.py tests/test_geodesic_config.py
git commit -m "feat(holov2): GeodesicConfig + branchement PrepareConfig (knobs graphe géodésique)"
```

---

### Task 2: Contrat `GeodesicTable` + `Channel.geodesic`

**Files:**
- Modify: `src/prepare/contracts.py` (ajouter `GeodesicTable` ; `Channel += geodesic`)
- Test: `tests/test_geodesic_contract.py`

**Interfaces:**
- Produces: `GeodesicTable(points:(P,3)f32, normals:(P,3)f32, geo:(P,P)f32, name:str, sampling_id:str="")` avec `n_points` et `__post_init__` (formes) ; `Channel(name, object_idx, sdf, geodesic: GeodesicTable | None = None)`.

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_contract.py` :
```python
"""GeodesicTable: invariants de forme (geo carrée (P,P), normals alignées aux points) ; et Channel
porte un geodesic optionnel (None = sol plan, défaut), sans casser la signature SDF existante."""
import numpy as np
import pytest

from src.prepare.contracts import GeodesicTable, Channel, SDF


def _sdf():
    return SDF(grid=np.zeros((2, 2, 2), np.float32), witness=np.zeros((2, 2, 2, 3), np.float32),
               origin=np.zeros(3), spacing=0.1, name="g")


def _table(P=4):
    return GeodesicTable(points=np.zeros((P, 3), np.float32), normals=np.zeros((P, 3), np.float32),
                         geo=np.zeros((P, P), np.float32), name="obj0", sampling_id="abc")


def test_table_n_points_and_ok():
    t = _table(5)
    assert t.n_points == 5


def test_table_rejects_non_square_geo():
    with pytest.raises(ValueError):
        GeodesicTable(points=np.zeros((4, 3), np.float32), normals=np.zeros((4, 3), np.float32),
                      geo=np.zeros((4, 5), np.float32), name="x")


def test_table_rejects_normals_shape_mismatch():
    with pytest.raises(ValueError):
        GeodesicTable(points=np.zeros((4, 3), np.float32), normals=np.zeros((3, 3), np.float32),
                      geo=np.zeros((4, 4), np.float32), name="x")


def test_channel_geodesic_defaults_none():
    ch = Channel("ground", None, _sdf())            # sol plan : pas de table
    assert ch.geodesic is None


def test_channel_carries_geodesic():
    ch = Channel("obj0", 0, _sdf(), geodesic=_table())
    assert isinstance(ch.geodesic, GeodesicTable)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_contract.py -q`
Expected: FAIL — `ImportError: cannot import name 'GeodesicTable'`.

- [ ] **Step 3: Write minimal implementation**

Dans `src/prepare/contracts.py`, ajouter (section `sdf / point_cloud`, après `SDF`) :
```python
@dataclass(frozen=True)
class GeodesicTable:
    """All-pairs géodésique (distance de graphe k-NN) sur les points de surface d'un mesh rigide, en
    frame locale. AUTO-CONTENU : porte SES points + normales, donc consommable sans le ``object_cloud``.
    La ligne ``geo[j]`` EST le champ géodésique mono-source depuis le point ``j`` (lookup O(1), ligne
    contiguë) — c'est ce qu'on lit à un ``witness(q)`` continu pour le résidu witness (côté solve).
    Géométrie rigide ⇒ pose-invariant (une translation/rotation préserve les géodésiques)."""

    points: np.ndarray    # (P, 3) f32  échantillons de surface (= sampling object_cloud), frame locale
    normals: np.ndarray   # (P, 3) f32  normale unitaire par point (gating snap/interp thin/concave)
    geo: np.ndarray       # (P, P) f32  geo[i, j] = géodésique de graphe i->j (symétrique)
    name: str             # nom de canal ("obj0"/"terrain") — provenance, aligné SDF/cloud
    sampling_id: str = "" # identité du sampling (densité/seed/topo) — provenance/garde-fou

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def __post_init__(self) -> None:
        p = self.points.shape[0]
        if self.geo.shape != (p, p):
            raise ValueError(f"geo shape {self.geo.shape} != (P, P) with P={p}")
        if self.normals.shape != self.points.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} != points shape {self.points.shape}")
```

Dans la dataclass `Channel`, ajouter le champ (après `sdf`) et compléter la docstring :
```python
    geodesic: "GeodesicTable | None" = None   # None = sol PLAN (le coût retombe sur l'euclidien
                                              # analytique, qui EST la géodésique exacte d'un plan) ;
                                              # sinon objet/terrain. Seule entorse au "jamais de None"
                                              # du sdf, assumée : le plan est le seul cas à forme close.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_contract.py -q && $PY -m pytest tests/test_sdf.py -q`
Expected: PASS partout (le second confirme que `Channel("ground", None, sdf)` reste valide).

- [ ] **Step 5: Commit**

```bash
git add src/prepare/contracts.py tests/test_geodesic_contract.py
git commit -m "feat(holov2): contrat GeodesicTable + Channel.geodesic (None=sol plan)"
```

---

### Task 3: Persistance `.npz` — `prepare/geodesic/cache.py`

**Files:**
- Create: `src/prepare/geodesic/__init__.py`
- Create: `src/prepare/geodesic/cache.py`
- Test: `tests/test_geodesic_cache.py`

**Interfaces:**
- Consumes: `GeodesicTable` (Task 2).
- Produces: `save_geo(table: GeodesicTable, path) -> None` ; `load_geo(path) -> GeodesicTable`.

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_cache.py` :
```python
"""Round-trip .npz de GeodesicTable : save->load reproduit exactement (writer/reader co-localisés)."""
import numpy as np

from src.prepare.contracts import GeodesicTable
from src.prepare.geodesic.cache import save_geo, load_geo


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    t = GeodesicTable(points=rng.standard_normal((6, 3)).astype(np.float32),
                      normals=rng.standard_normal((6, 3)).astype(np.float32),
                      geo=rng.standard_normal((6, 6)).astype(np.float32),
                      name="obj0", sampling_id="deadbeef")
    p = tmp_path / "sub" / "obj0.npz"           # parents créés à la volée
    save_geo(t, p)
    g = load_geo(p)
    assert np.array_equal(t.points, g.points)
    assert np.array_equal(t.normals, g.normals)
    assert np.array_equal(t.geo, g.geo)
    assert g.name == "obj0" and g.sampling_id == "deadbeef"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prepare.geodesic'`.

- [ ] **Step 3: Write minimal implementation**

`src/prepare/geodesic/__init__.py` :
```python
"""prepare/geodesic — table all-pairs de distances géodésiques par mesh (objets/terrain), build + cache.

Thin re-export de la surface publique ; logique de build dans ``build.py``, I/O .npz dans ``cache.py``."""
from .cache import load_geo, save_geo

__all__ = ["save_geo", "load_geo"]
```

`src/prepare/geodesic/cache.py` :
```python
"""(Dé)sérialisation .npz d'une ``GeodesicTable`` — save ET load au même endroit, pour que le writer
et le reader ne dérivent pas. ``geo`` est une matrice (P,P) f32 → ``np.savez_compressed``."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import GeodesicTable


def save_geo(table: GeodesicTable, path: Path) -> None:
    """Sérialise une ``GeodesicTable`` vers ``path`` (crée les dossiers parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), points=table.points, normals=table.normals, geo=table.geo,
                        name=np.array(table.name), sampling_id=np.array(table.sampling_id))


def load_geo(path: Path) -> GeodesicTable:
    """Inverse de ``save_geo``."""
    d = np.load(str(path), allow_pickle=False)
    return GeodesicTable(points=d["points"], normals=d["normals"], geo=d["geo"],
                         name=str(d["name"]), sampling_id=str(d["sampling_id"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prepare/geodesic/__init__.py src/prepare/geodesic/cache.py tests/test_geodesic_cache.py
git commit -m "feat(holov2): prepare/geodesic — persistance .npz (save_geo/load_geo)"
```

---

### Task 4: Fonctions pures du build — `prepare/geodesic/build.py`

**Files:**
- Create: `src/prepare/geodesic/build.py`
- Test: `tests/test_geodesic_build.py`

**Interfaces:**
- Consumes: `GeodesicTable` (Task 2), `CloudConfig`/`GeodesicConfig` (Task 1), `sample_object_surface` (référence de parité, `src/prepare/point_cloud/objects.py`).
- Produces (toutes pures, déterministes) :
  - `sample_surface_with_normals(vertices, faces, density:float, seed:int) -> (points:(P,3)f64, normals:(P,3)f64)`
  - `build_knn_graph(points:(P,3), normals:(P,3), k:int, normal_gate:float) -> scipy.sparse.csr_matrix (P,P)`
  - `all_pairs_geodesic(graph) -> (P,P) f32`  (lève `ValueError` si disconnecté)
  - `build_geodesic_table(vertices, faces, cloud_cfg:CloudConfig, geo_cfg:GeodesicConfig, name:str="") -> GeodesicTable`  (lève `ValueError` si P>max_points)

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_build.py` :
```python
"""prepare/geodesic build (fonctions pures) : parité du sampling avec le object_cloud, géométrie
géodésique sur formes connues (plan ≈ euclidien, sphère ≫ euclidien), invariants matrice, et les
garde-fous (graphe disconnecté, max_points, gating normales au niveau du graphe)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.prepare.config import CloudConfig, GeodesicConfig
from src.prepare.point_cloud.objects import sample_object_surface
from src.prepare.geodesic.build import (sample_surface_with_normals, build_knn_graph,
                                        all_pairs_geodesic, build_geodesic_table)


def _sphere(sub=3, r=0.2):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)      # fermé, bien connecté, normales sortantes
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _grid(n=30, side=1.0):
    """Nappe plane (points + normales +z) — graphe/géodésique contrôlés sans confondre avec le sampling."""
    xs = np.linspace(0.0, side, n)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    nrm = np.tile([0.0, 0.0, 1.0], (pts.shape[0], 1))
    return pts, nrm


def test_sampling_matches_object_cloud():
    # Parité EXACTE des points avec le object_cloud (même densité/seed/topo) : un seul sampling canonique.
    v, f = _sphere()
    pts_geo, nrm = sample_surface_with_normals(v, f, density=500.0, seed=0)
    pts_cloud = sample_object_surface(v, f, density=500.0, seed=0)
    assert np.array_equal(pts_geo, pts_cloud)
    assert nrm.shape == pts_geo.shape
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)   # normales unitaires


def test_plane_geodesic_close_to_euclidean():
    # Surface plate (nappe) : la géodésique de graphe ≈ euclidien (sur-estime un peu, jamais en dessous).
    pts, nrm = _grid(n=30, side=1.0)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=8, normal_gate=-0.5))
    eucl = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    far = eucl > 0.3
    ratio = D[far] / eucl[far]
    assert (ratio >= 0.999).all()                                    # jamais sous l'euclidien
    assert np.median(ratio) < 1.2                                    # sur-estime modérément


def test_sphere_geodesic_exceeds_euclidean():
    # Sphère (convexe, SANS raccourci) : la paire ~antipodale a une géodésique ≈ grand cercle (π·r)
    # ≫ la corde euclidienne (2r). NB un cylindre PLEIN ne marche pas : ses capuchons plats offrent un
    # raccourci à travers la face (ratio ~1.1), donc on prend une sphère.
    v, f = _sphere(sub=3, r=0.2)
    pts, nrm = sample_surface_with_normals(v, f, density=1500.0, seed=0)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=10, normal_gate=-1.0))
    eucl = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(eucl), eucl.shape)            # paire la plus écartée (~antipodale)
    assert D[i, j] > 1.3 * eucl[i, j]                               # le chemin suit la surface (grand cercle)


def test_matrix_invariants():
    pts, nrm = _grid(n=20)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=8, normal_gate=-0.5))
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T, atol=1e-5)
    assert D.dtype == np.float32


def test_disconnected_graph_raises():
    # Deux amas éloignés, k=1 → composantes disjointes → ValueError explicite.
    pts = np.concatenate([np.zeros((5, 3)), np.zeros((5, 3)) + [[10.0, 0, 0]]])
    pts += np.random.default_rng(0).normal(0, 1e-3, pts.shape)
    nrm = np.tile([0.0, 0, 1.0], (10, 1))
    with pytest.raises(ValueError):
        all_pairs_geodesic(build_knn_graph(pts, nrm, k=1, normal_gate=-1.0))


def test_normal_gate_cuts_opposite_normal_edges():
    # 2 points proches à normales ~opposées (faces d'une plaque) : l'arête directe existe SANS gating
    # et disparaît AVEC (dot≈-0.995 < -0.5). Une arête à même normale reste. Test au niveau du graphe.
    pts = np.array([[0.0, 0, 0.001], [0.0, 0, -0.001], [0.05, 0, 0.001], [0.05, 0, -0.001]])
    nrm = np.array([[0.0, 0, 1.0], [0.0, 0.1, -1.0], [0.0, 0, 1.0], [0.0, 0.1, -1.0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)              # bas ≈ (0, 0.0995, -0.995)
    g_open = build_knn_graph(pts, nrm, k=3, normal_gate=-1.0)      # garde tout sauf dot=-1 exact
    g_gated = build_knn_graph(pts, nrm, k=3, normal_gate=-0.5)
    assert g_open[0, 1] > 0                                        # 0-1 (opposés, proches) présent
    assert g_gated[0, 1] == 0                                      # coupé par le gating
    assert g_gated[0, 2] > 0                                       # 0-2 (même normale) conservé


def test_max_points_guard():
    v, f = _sphere()
    with pytest.raises(ValueError):
        build_geodesic_table(v, f, CloudConfig(object_density=800.0, seed=0),
                             GeodesicConfig(max_points=10))         # P >> 10


def test_build_table_shapes():
    v, f = _sphere()
    t = build_geodesic_table(v, f, CloudConfig(object_density=500.0, seed=0),
                             GeodesicConfig(normal_gate=-1.0), name="obj0")
    assert t.geo.shape == (t.n_points, t.n_points)
    assert t.points.dtype == np.float32 and t.geo.dtype == np.float32
    assert t.name == "obj0" and t.sampling_id != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_build.py -q`
Expected: FAIL — `ImportError` (`build.py` n'existe pas).

- [ ] **Step 3: Write minimal implementation**

`src/prepare/geodesic/build.py` :
```python
"""prepare/geodesic — construit la table all-pairs de distances géodésiques d'un mesh rigide
(objet ou terrain) sur les points du ``object_cloud``.

Offline, scopé géométrie, caché une fois (``GeodesicBuilder``) : un asset subject-/robot-free partagé
par toute séquence touchant le même mesh. Pipeline : ré-échantillonner la surface À L'IDENTIQUE du
``object_cloud`` (mêmes densité/seed → points bit-identiques) en récupérant les normales, construire un
graphe k-NN de surface (poids euclidien) GATÉ par les normales (coupe les arêtes "cross-gap" des
plaques fines), puis Dijkstra all-pairs (scipy). La matrice est consommée comme des CHAMPS mono-source
(``geo[j]``) — voir ``GeodesicTable``. Sampling du champ à un ``witness(q)`` continu : pas ici (online,
``targets/interaction/geodesic.py``).

scipy/trimesh sont confinés à ce builder (il tourne une fois et est caché) ; le contrat et la query
restent numpy-only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import GeodesicTable
from ..config import CloudConfig, GeodesicConfig
from .cache import load_geo, save_geo


def sample_surface_with_normals(vertices: np.ndarray, faces: np.ndarray, density: float,
                                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rejoue EXACTEMENT le sampling du ``object_cloud`` (``max(64, int(area*density))`` points,
    ``sample_surface_even`` déterministe en ``seed``) → points bit-identiques au cloud, PLUS la normale
    de surface par point (normale de la face échantillonnée). Pur : construit un trimesh interne, no I/O."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64),
                           faces=np.asarray(faces), process=False)
    n = max(64, int(float(mesh.area) * density))
    pts, fid = trimesh.sample.sample_surface_even(mesh, n, seed=seed)   # fid = index de face par point
    normals = np.asarray(mesh.face_normals)[np.asarray(fid)]
    return np.asarray(pts, np.float64), np.asarray(normals, np.float64)


def build_knn_graph(points: np.ndarray, normals: np.ndarray, k: int, normal_gate: float):
    """Graphe k-NN de surface (poids = distance euclidienne), GATÉ par normales : arête i--j seulement
    si ``dot(n_i, n_j) > normal_gate``. Symétrisé (non orienté). Pur ; renvoie une ``csr_matrix`` (P,P)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    pts = np.asarray(points, np.float64)
    nrm = np.asarray(normals, np.float64)
    p = pts.shape[0]
    kq = min(k + 1, p)                                          # +1 : le 1er voisin est soi-même
    dist, idx = cKDTree(pts).query(pts, k=kq)
    dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(p), kq - 1)
    dst = idx[:, 1:].reshape(-1)
    wts = dist[:, 1:].reshape(-1)
    keep = np.einsum("ij,ij->i", nrm[src], nrm[dst]) > normal_gate
    src, dst, wts = src[keep], dst[keep], wts[keep]
    a = csr_matrix((wts, (src, dst)), shape=(p, p))
    return a.maximum(a.T)                                       # non orienté : arête si l'un OU l'autre


def all_pairs_geodesic(graph) -> np.ndarray:
    """All-pairs plus court chemin (Dijkstra) sur le graphe de surface → (P,P) f32. Lève si le graphe
    est disconnecté (un ``inf`` = paire sans chemin) plutôt que de stocker des ``inf``."""
    from scipy.sparse.csgraph import shortest_path
    d = shortest_path(graph, method="D", directed=False)
    if not np.isfinite(d).all():
        raise ValueError("geodesic graph is disconnected (some pairs have no path) — "
                         "increase k_neighbors or lower normal_gate")
    d = 0.5 * (d + d.T)                                         # nettoie l'asymétrie FP
    return d.astype(np.float32)


def _sampling_id(cloud_cfg: CloudConfig, vertices: np.ndarray, faces: np.ndarray) -> str:
    """Hash stable du sampling (densité+seed) + géométrie — provenance/garde-fou (16 hex)."""
    h = hashlib.sha1()
    h.update(f"{cloud_cfg.object_density}|{cloud_cfg.seed}".encode())
    h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
    h.update(np.ascontiguousarray(faces, np.int64).tobytes())
    return h.hexdigest()[:16]


def build_geodesic_table(vertices: np.ndarray, faces: np.ndarray, cloud_cfg: CloudConfig,
                         geo_cfg: GeodesicConfig, name: str = "") -> GeodesicTable:
    """Échantillonne (à l'identique du cloud) puis calcule la table géodésique. Garde-fou ``max_points``
    (stockage 4*P^2). Pur : no I/O, ne mute pas ses inputs."""
    pts, nrm = sample_surface_with_normals(vertices, faces, cloud_cfg.object_density, cloud_cfg.seed)
    p = pts.shape[0]
    if p > geo_cfg.max_points:
        raise ValueError(f"geodesic sampling has P={p} > max_points={geo_cfg.max_points} "
                         f"(storage is 4*P^2 bytes) — lower object_density or raise max_points")
    geo = all_pairs_geodesic(build_knn_graph(pts, nrm, geo_cfg.k_neighbors, geo_cfg.normal_gate))
    return GeodesicTable(points=pts.astype(np.float32), normals=nrm.astype(np.float32), geo=geo,
                         name=name, sampling_id=_sampling_id(cloud_cfg, vertices, faces))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_build.py -q`
Expected: PASS (8 cas).

- [ ] **Step 5: Commit**

```bash
git add src/prepare/geodesic/build.py tests/test_geodesic_build.py
git commit -m "feat(holov2): prepare/geodesic — build pur (sampling+normales, graphe k-NN gaté, Dijkstra all-pairs)"
```

---

### Task 5: `GeodesicBuilder` (AssetBuilder : cache_key/build/save/load)

**Files:**
- Modify: `src/prepare/geodesic/build.py` (ajouter `GeodesicBuilder`)
- Modify: `src/prepare/geodesic/__init__.py` (exporter `GeodesicBuilder`, `build_geodesic_table`)
- Test: `tests/test_geodesic_builder.py`

**Interfaces:**
- Consumes: les fonctions pures (Task 4), `save_geo`/`load_geo` (Task 3).
- Produces: `GeodesicBuilder` avec `cache_key(cloud_cfg, geo_cfg, vertices, faces) -> str`, `build(cloud_cfg, geo_cfg, vertices, faces, name="") -> GeodesicTable`, `save`, `load`.

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_builder.py` :
```python
"""GeodesicBuilder : déterminisme (build x2 identique), round-trip cache (save->load == build), et
sensibilité de la clé (sampling + knobs graphe + géométrie). Calqué sur test_sdf."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.prepare.config import CloudConfig, GeodesicConfig
from src.prepare.geodesic.build import GeodesicBuilder


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.2)     # fermé → graphe connecté
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


_CC = CloudConfig(object_density=500.0, seed=0)
_GC = GeodesicConfig(normal_gate=-1.0)


def test_determinism():
    v, f = _sphere()
    b = GeodesicBuilder()
    a = b.build(_CC, _GC, v, f)
    c = b.build(_CC, _GC, v, f)
    assert np.array_equal(a.points, c.points)
    assert np.array_equal(a.geo, c.geo)


def test_cache_roundtrip(tmp_path):
    v, f = _sphere()
    b = GeodesicBuilder()
    t = b.build(_CC, _GC, v, f, name="obj0")
    p = tmp_path / "obj0.npz"
    b.save(t, p)
    g = b.load(p)
    assert np.array_equal(t.geo, g.geo)
    assert np.array_equal(t.points, g.points)
    assert g.name == "obj0"


def test_cache_key_sensitivity():
    v, f = _sphere()
    b = GeodesicBuilder()
    k = b.cache_key(_CC, _GC, v, f)
    assert k == b.cache_key(_CC, _GC, v, f)
    assert k != b.cache_key(CloudConfig(object_density=800.0, seed=0), _GC, v, f)   # densité
    assert k != b.cache_key(CloudConfig(object_density=500.0, seed=1), _GC, v, f)   # seed
    assert k != b.cache_key(_CC, GeodesicConfig(k_neighbors=12, normal_gate=-1.0), v, f)  # k
    assert k != b.cache_key(_CC, GeodesicConfig(normal_gate=0.0), v, f)             # gate
    v2 = v.copy(); v2[0] += 0.1
    assert k != b.cache_key(_CC, _GC, v2, f)                                        # géométrie
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_builder.py -q`
Expected: FAIL — `ImportError: cannot import name 'GeodesicBuilder'`.

- [ ] **Step 3: Write minimal implementation**

À la fin de `src/prepare/geodesic/build.py`, ajouter :
```python
class GeodesicBuilder:
    """``AssetBuilder`` de la table géodésique d'un mesh (objet/terrain). Scopé GÉOMÉTRIE (+ le
    sampling ``CloudConfig`` qui fixe les points, + les knobs graphe ``GeodesicConfig``) : deux
    séquences partageant un mesh partagent la table cachée, indépendamment du subject/robot. Le runner
    enveloppe ``build``/``load`` dans un ``prof.span("geodesic")``. ``max_points`` est un garde-fou
    (ne change pas l'asset produit) ⇒ HORS clé."""

    def cache_key(self, cloud_cfg: CloudConfig, geo_cfg: GeodesicConfig, vertices: np.ndarray,
                  faces: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(f"{cloud_cfg.object_density}|{cloud_cfg.seed}|{geo_cfg.k_neighbors}|"
                 f"{geo_cfg.normal_gate}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()

    def build(self, cloud_cfg: CloudConfig, geo_cfg: GeodesicConfig, vertices: np.ndarray,
              faces: np.ndarray, name: str = "") -> GeodesicTable:
        return build_geodesic_table(vertices, faces, cloud_cfg, geo_cfg, name=name)

    def save(self, table: GeodesicTable, path: Path) -> None:
        return save_geo(table, path)

    def load(self, path: Path) -> GeodesicTable:
        return load_geo(path)
```

Mettre à jour `src/prepare/geodesic/__init__.py` :
```python
from .build import GeodesicBuilder, build_geodesic_table
from .cache import load_geo, save_geo

__all__ = ["GeodesicBuilder", "build_geodesic_table", "save_geo", "load_geo"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_builder.py -q`
Expected: PASS (3 cas).

- [ ] **Step 5: Commit**

```bash
git add src/prepare/geodesic/build.py src/prepare/geodesic/__init__.py tests/test_geodesic_builder.py
git commit -m "feat(holov2): GeodesicBuilder — AssetBuilder caché par géométrie (déterminisme + round-trip)"
```

---

### Task 6: Câblage runner — `_build_channels` peuple `Channel.geodesic`

**Files:**
- Modify: `src/prepare/runner.py` (`_build_channels` : import `GeodesicBuilder`, build par objet/terrain, passe au `Channel` ; `prof.span("geodesic")` dans `_run`)
- Test: `tests/test_geodesic_runner.py`

**Interfaces:**
- Consumes: `GeodesicBuilder` (Task 5), `_load_or_build`/`_build_channels` (`src/prepare/runner.py`), `Channel.geodesic` (Task 2).
- Produces: `_build_channels(...)` renvoie des `Channel` avec `geodesic` non-`None` pour objets/terrain, `None` pour le sol plat.

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_runner.py` :
```python
"""_build_channels rattache une GeodesicTable par objet (et None au sol plat). Scène minimale montée à
la main (un petit cube exporté), force=True pour bâtir sans cache. _build_channels n'utilise ni la
calibration ni le robot → valeurs factices acceptables."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.obs import NULL
from src.prepare.config import PrepareConfig, SdfConfig, CloudConfig, GeodesicConfig
from src.prepare.contracts import (GroundedScene, Calibration, SceneSpec, RobotSpec, GeodesicTable)
from src.prepare.runner import _build_channels


def _spec(tmp_path, ground=None):
    robot = RobotSpec(name="g1", urdf_path=tmp_path / "x.urdf", link_names=("a",), dof=1, height=1.2)
    return SceneSpec(dataset="demo", motion_path=tmp_path, robot=robot,
                     ground_mesh_path=ground, cache_dir=tmp_path)


def _grounded(mesh_path):
    pose = np.tile([0.0, 0, 0, 1, 0, 0, 0], (2, 1))
    calib = Calibration(human_offset=0.0, object_offset=0.0, root_frame=np.eye(4))
    return GroundedScene(joint_pos=np.zeros((2, 3, 3)), joint_names=("a", "b", "c"),
                         object_poses=(pose,), object_mesh_paths=(mesh_path,), calibration=calib,
                         fps=30.0)


def _cfg():
    # coarse + clairsemé → test rapide
    return PrepareConfig(sdf=SdfConfig(spacing=0.05, margin=0.05),
                         cloud=CloudConfig(object_density=200.0, seed=0),
                         geodesic=GeodesicConfig(normal_gate=-1.0))


def test_object_channel_gets_geodesic_flat_ground_none(tmp_path):
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    mesh_path = tmp_path / "box.obj"; box.export(mesh_path)
    channels = _build_channels(_grounded(mesh_path), _spec(tmp_path), _cfg(), tmp_path, NULL, force=True)
    assert channels[0].name == "ground" and channels[0].geodesic is None       # sol plat
    assert isinstance(channels[1].geodesic, GeodesicTable)                      # objet
    assert channels[1].geodesic.n_points == channels[1].geodesic.geo.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_runner.py -q`
Expected: FAIL — `AssertionError` (`channels[1].geodesic is None`, pas encore câblé).

- [ ] **Step 3: Write minimal implementation**

Dans `src/prepare/runner.py` :

1. Ajouter l'import (à côté de `from .sdf import SdfBuilder, build_plane_sdf`) :
```python
from .geodesic import GeodesicBuilder
```

2. Dans `_build_channels`, instancier le builder à côté de `sdf_builder = SdfBuilder()` :
```python
    geo_builder = GeodesicBuilder()
```

3. Sol-terrain : dans la branche `else` (mesh de sol), après avoir bâti `ground_sdf`, bâtir aussi la table puis remplacer la construction `channels = [Channel("ground", None, ground_sdf)]`. Restructurer ainsi :
```python
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
```

4. Boucle objets : ajouter le build géodésique et le passer au `Channel` :
```python
    for i, mesh_path in enumerate(grounded.object_mesh_paths):
        v, f = load_mesh(mesh_path)
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
```

5. Dans `_run`, renommer le span SDF pour englober la géodésique (les deux sont bâtis dans `_build_channels`). Remplacer `with prof.span("sdf", n=grounded.n_objects + 1):` par :
```python
        with prof.span("sdf+geodesic", n=grounded.n_objects + 1):
            channels = _build_channels(grounded, spec, config, cache_dir, prof, force=force)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing runner test to confirm no regression**

Run: `cd HoloV2 && $PY -m pytest tests/test_runner_prepare.py tests/test_runner_validate.py -q`
Expected: PASS (mêmes que sur l'arbre de base ; le câblage est additif). Si ces tests étaient déjà rouges/lents sur l'arbre de base, le noter sans bloquer.

- [ ] **Step 6: Commit**

```bash
git add src/prepare/runner.py tests/test_geodesic_runner.py
git commit -m "feat(holov2): runner câble Channel.geodesic (objets/terrain ; sol plat=None)"
```

---

### Task 7: Helper de query online — `targets/interaction/geodesic.py`

**Files:**
- Create: `src/targets/interaction/geodesic.py`
- Modify: `src/targets/interaction/__init__.py` (export `geo_value_grad`, `nearest_index`)
- Modify: `src/targets/__init__.py` (ré-export depuis le package)
- Test: `tests/test_geodesic_query.py`

**Interfaces:**
- Consumes: `GeodesicTable` (Task 2) — lit `.points` et `.geo` uniquement (numpy-only, torch-free).
- Produces:
  - `nearest_index(points:(P,3), xyz:(Q,3) | (3,)) -> (Q,) int`  (snap `witness_ref` → source, offline)
  - `geo_value_grad(table:GeodesicTable, source_idx:(Q,) int, query_xyz:(Q,3), k:int=6) -> (g:(Q,), grad:(Q,3))`  (MLS degré-1 ; valeur + gradient tangent, différentiable)

- [ ] **Step 1: Write the failing test**

`tests/test_geodesic_query.py` :
```python
"""Helper de query géodésique (numpy-only) : nearest_index (snap), et geo_value_grad par MLS degré-1.
Reproduction EXACTE d'un champ linéaire (le fit degré-1 est exact dessus), gradient vs différences
finies sur un champ non-linéaire, et import via la surface publique targets."""
import numpy as np

from src.prepare.contracts import GeodesicTable
from src.targets.interaction.geodesic import nearest_index, geo_value_grad


def _table_from_field(points, field):
    P = len(points)
    geo = np.tile(field.astype(np.float32), (P, 1))     # geo[src] = field pour toute source (test)
    return GeodesicTable(points=points.astype(np.float32), normals=np.tile([0, 0, 1.0], (P, 1)).astype(np.float32),
                         geo=geo, name="t")


def test_nearest_index():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert nearest_index(pts, np.array([0.9, 0.1, 0])) == 1
    assert list(nearest_index(pts, np.array([[0.1, 0.1, 0], [0.1, 0.9, 0]]))) == [0, 2]


def test_linear_field_reproduced_exactly():
    # champ f(x) = a·x + c → MLS degré-1 doit rendre g=f(query) et grad=a EXACTEMENT.
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(200, 3)); pts[:, 2] = 0.0      # surface plate (2D dans z=0)
    a = np.array([2.0, -3.0, 0.0]); c = 0.5
    field = pts @ a + c
    table = _table_from_field(pts, field)
    q = np.array([[0.2, -0.1, 0.0], [-0.3, 0.4, 0.0]])
    g, grad = geo_value_grad(table, source_idx=np.array([0, 0]), query_xyz=q, k=8)
    assert np.allclose(g, q @ a + c, atol=1e-6)
    assert np.allclose(grad[:, :2], a[:2], atol=1e-6)            # gradient (composantes dans le plan)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, size=(400, 3)); pts[:, 2] = 0.0
    p0 = np.array([0.0, 0.0, 0.0])
    field = np.linalg.norm(pts - p0, axis=1)                    # non-linéaire (cône)
    table = _table_from_field(pts, field)
    q = np.array([[0.3, 0.2, 0.0]])
    g, grad = geo_value_grad(table, np.array([0]), q, k=12)
    eps = 1e-4
    fd = np.array([(geo_value_grad(table, np.array([0]), q + d, k=12)[0][0]
                    - geo_value_grad(table, np.array([0]), q - d, k=12)[0][0]) / (2 * eps)
                   for d in (np.array([[eps, 0, 0]]), np.array([[0, eps, 0]]))])
    assert np.allclose(grad[0, :2], fd, atol=2e-2)


def test_public_surface_import():
    from src.targets import geo_value_grad as g1, nearest_index as g2
    assert callable(g1) and callable(g2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_query.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.targets.interaction.geodesic'`.

- [ ] **Step 3: Write minimal implementation**

`src/targets/interaction/geodesic.py` :
```python
"""Query online du champ géodésique précalculé (``prepare.GeodesicTable``), q-DÉPENDANTE car lue à un
``witness(q)`` continu. NUMPY-ONLY / torch-free (``targets`` reste léger) : pas de scipy — le k-NN est
un brute-force vectorisé sur les Q witnesses (P modéré, batch ``(Q,P)``).

``nearest_index`` snappe un point fixe (``witness_ref``) sur sa source la plus proche (OFFLINE, pas de
gradient). ``geo_value_grad`` lit le champ mono-source ``geo[source]`` à un ``query_xyz`` continu par
MLS degré-1 (moindres carrés locaux pondérés, fit ``f≈c+b·(y-x)``) → valeur ``c`` ET gradient ``b``,
naturellement tangent à la surface (les voisins engendrent le plan tangent), de norme ~1 façon eikonale.
Différentiable et vectorisé sur Q. Le résidu de coût qui consomme ``(g, grad)`` est écrit par l'utilisateur."""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import GeodesicTable


def nearest_index(points: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Indice du point de surface le plus proche de ``xyz`` (snap ``witness_ref`` → source, offline).
    ``xyz`` (3,) ou (Q,3) → (Q,) int. Brute-force numpy."""
    pts = np.asarray(points, np.float64)
    q = np.atleast_2d(np.asarray(xyz, np.float64))
    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)            # (Q, P)
    return np.argmin(d2, axis=1)


def geo_value_grad(table: GeodesicTable, source_idx: np.ndarray, query_xyz: np.ndarray,
                   k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Champ géodésique depuis ``source_idx`` lu à ``query_xyz`` par MLS degré-1.
    Renvoie ``(g:(Q,), grad:(Q,3))``. Vectorisé sur Q ; reproduit un champ localement linéaire."""
    pts = np.asarray(table.points, np.float64)                      # (P,3)
    q = np.atleast_2d(np.asarray(query_xyz, np.float64))            # (Q,3)
    src = np.atleast_1d(np.asarray(source_idx, np.int64))           # (Q,)
    P, Q = pts.shape[0], q.shape[0]
    k = min(k, P)
    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)           # (Q,P)
    nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]              # (Q,k) k plus proches
    fields = table.geo[src]                                         # (Q,P) champ mono-source par query
    f = np.take_along_axis(fields.astype(np.float64), nn, axis=1)   # (Q,k)
    p = pts[nn]                                                     # (Q,k,3)
    dq = p - q[:, None, :]                                          # (Q,k,3) coords locales centrées en q
    dist = np.sqrt(np.take_along_axis(d2, nn, axis=1))             # (Q,k)
    h = dist.mean(axis=1, keepdims=True) + 1e-12                    # bande adaptative par query
    w = np.exp(-(dist / h) ** 2)                                   # (Q,k) poids gaussiens
    X = np.concatenate([np.ones((Q, k, 1)), dq], axis=2)           # (Q,k,4) design [1, dq]
    Xw = X * w[:, :, None]
    A = np.einsum("qki,qkj->qij", Xw, X) + 1e-9 * np.eye(4)         # (Q,4,4) normal eq. + ridge
    rhs = np.einsum("qki,qk->qi", Xw, f)                           # (Q,4)
    sol = np.linalg.solve(A, rhs)                                  # (Q,4) : [c, bx, by, bz]
    return sol[:, 0], sol[:, 1:]
```

Mettre à jour `src/targets/interaction/__init__.py` :
```python
from .pointclouds import pose_cloud
from .eval import eval_fields
from .transport import transport
from .targets import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
```

Mettre à jour `src/targets/__init__.py` (ajouter au re-export et à `__all__`) :
```python
from .interaction import eval_fields, pose_cloud, geo_value_grad, nearest_index
from .contracts import MultiChannelField

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField", "geo_value_grad", "nearest_index"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloV2 && $PY -m pytest tests/test_geodesic_query.py -q`
Expected: PASS (4 cas).

- [ ] **Step 5: Confirm targets stays torch-free**

Run: `cd HoloV2 && $PY -c "import sys; import src.targets.interaction.geodesic; assert 'torch' not in sys.modules and 'scipy' not in sys.modules, 'targets query must be numpy-only'; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/targets/interaction/geodesic.py src/targets/interaction/__init__.py src/targets/__init__.py tests/test_geodesic_query.py
git commit -m "feat(holov2): targets — geo_value_grad/nearest_index (query géodésique MLS, numpy-only)"
```

---

### Task 8: Intégration docs de structure (source unique)

**Files:**
- Modify: `docs/ARCHITECTURE.md`, `docs/PREPARE.md`, `docs/CACHE.md`, `docs/TARGETS.md`
- (Pas de test ; vérification = relecture + suite verte + grep de présence)

**Interfaces:**
- Consumes: tout le travail des Tasks 1–7 (noms, chemins, comportements).

- [ ] **Step 1: ARCHITECTURE.md**

Dans l'arbre `prepare/`, après la ligne `point_cloud/ LIVRABLE : nuages …`, ajouter :
```
      geodesic/        LIVRABLE : table all-pairs géodésique par mesh (objets/terrain ; sol plat = aucune table)
```
À la ligne des sous-configs (`config.py … CalibrationConfig/SdfConfig/CloudConfig/CorrespondenceConfig/PrepareConfig`), insérer `GeodesicConfig` avant `PrepareConfig`. À la ligne des types `prepare` (`… SDF/Channel · PointCloud/CorrespondenceTable …`), ajouter `GeodesicTable`.

- [ ] **Step 2: PREPARE.md**

- Arbre `prepare/` : après `point_cloud/ NUAGES + correspondance`, ajouter `geodesic/  meshes objets/terrain -> table géodésique (cachée) ; sol plat -> aucune table ; expose save_geo/load_geo (GeodesicBuilder délègue)`.
- Sous-configs (`config.py … CalibrationConfig/SdfConfig/CloudConfig/CorrespondenceConfig`) : ajouter `GeodesicConfig`.
- « Contrat commun des builders offline » : passer **3 livrables → 4** (calibration, sdf, point_cloud, **geodesic**) et **5 builders → 6** (ajouter `GeodesicBuilder`).
- Liste des champs `Channel` : remplacer/compléter par « `Channel` (`object_idx` + `sdf` TOUJOURS présent ; `geodesic` `GeodesicTable | None`, `None` = sol plan → coût euclidien analytique) ».
- Liste des types « champs (assets) » : ajouter `GeodesicTable` (table all-pairs (P,P), objets/terrain).

- [ ] **Step 3: CACHE.md**

- Table des assets : ajouter une ligne `| **Géodésique objet/terrain** | mesh (hash géom) + sampling (densité/seed) + knobs graphe (k, normal_gate) | geodesic | — | toutes scènes avec ce mesh |`.
- Arbre `HoloV2/cache/` : après la ligne `sdf/`, ajouter `geodesic/        <geom_hash>_<cfg>.npz            (objets/terrain ; sol plat = aucune table)`.
- Règles d'invalidation : ajouter `- changer geodesic ou cloud → rebuild table géodésique seulement`.

- [ ] **Step 4: TARGETS.md**

Dans la surface publique de `targets`, ajouter : « ré-export `geo_value_grad`/`nearest_index` (depuis `targets/interaction/geodesic.py`) — lecture **différentiable** (MLS degré-1, valeur + gradient tangent) du champ géodésique précalculé `prepare.GeodesicTable` à un `witness(q)` continu, pour le résidu witness côté solve/utilisateur ; `nearest_index` snappe `witness_ref` sur sa source offline. Numpy-only, torch-free. »

- [ ] **Step 5: Vérifier la cohérence + grep de présence**

Run:
```bash
cd HoloV2 && grep -l geodesic docs/ARCHITECTURE.md docs/PREPARE.md docs/CACHE.md docs/TARGETS.md
```
Expected: les 4 fichiers listés.

- [ ] **Step 6: Suite complète du sous-module + py_compile**

Run:
```bash
cd HoloV2 && $PY -m pytest tests/test_geodesic_config.py tests/test_geodesic_contract.py \
  tests/test_geodesic_cache.py tests/test_geodesic_build.py tests/test_geodesic_builder.py \
  tests/test_geodesic_runner.py tests/test_geodesic_query.py -q
$PY -m py_compile src/prepare/config.py src/prepare/contracts.py src/prepare/runner.py \
  src/prepare/geodesic/build.py src/prepare/geodesic/cache.py src/targets/interaction/geodesic.py
```
Expected: tous les tests PASS ; `py_compile` silencieux.

- [ ] **Step 7: Commit**

```bash
git add docs/ARCHITECTURE.md docs/PREPARE.md docs/CACHE.md docs/TARGETS.md
git commit -m "docs(holov2): intègre prepare/geodesic à l'archi (structure + cache + surface targets)"
```

---

## Self-Review (couverture du spec)

- **Sous-module `prepare/geodesic/`** (build + cache, scopé géométrie) → Tasks 3, 4, 5. ✓
- **Contrat `GeodesicTable` + `Channel.geodesic`** → Task 2. ✓
- **`GeodesicConfig` + `PrepareConfig`** → Task 1. ✓
- **Helper `geo_value_grad` + `nearest_index` (numpy-only, MLS degré-1, différentiable)** → Task 7. ✓
- **Calcul graphe k-NN gaté normales + Dijkstra all-pairs (scipy)** → Task 4. ✓
- **Réutilisation du sampling object_cloud (points bit-identiques)** → Task 4 (`test_sampling_matches_object_cloud`). ✓
- **Sol plat = pas de table (`None`) ; objets/terrain = table** → Tasks 2, 6. ✓
- **Câblage runner + `prof.span`** → Task 6. ✓
- **Ré-exports targets** → Task 7. ✓
- **Intégration docs (ARCHITECTURE/PREPARE/CACHE/TARGETS) + plomberie config** → Tasks 1, 8. ✓
- **Garde-fous** : disconnecté (Task 4), max_points (Task 4), short-circuit thin-plate via gating normales (Task 4). ✓
- **Tests builder obligatoires (déterminisme + round-trip)** → Task 5. ✓
- **Chemin d'upgrade potpourri3d** : documenté dans le spec ; substitution derrière `GeodesicBuilder.build` sans toucher au contrat — pas une tâche d'implémentation (YAGNI). ✓

Tous les types/signatures sont cohérents entre tasks : `GeodesicTable(points, normals, geo, name, sampling_id)`, `GeodesicConfig(k_neighbors, normal_gate, max_points)`, `GeodesicBuilder.cache_key(cloud_cfg, geo_cfg, vertices, faces)`/`build(...)`, `geo_value_grad(table, source_idx, query_xyz, k)`, `nearest_index(points, xyz)`.
