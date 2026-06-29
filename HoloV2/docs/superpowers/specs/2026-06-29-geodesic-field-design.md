# Spec — Champ géodésique de surface précalculé (`prepare/geodesic`)

**Date** : 2026-06-29 · **Étage** : `prepare` (nouveau sous-module + contrat) + `targets` (helper de query) · **Statut** : conçu

## Problème

Le solveur reçoit, par point de contact et par canal, des quantités de **référence** `distance` +
`witness` (point de surface le plus proche), et recalcule à chaque itération SQP les quantités
**courantes** `distance(q)` + `witness(q)` sur les clouds posés (robot FK@q, objets à leur pose). Pour
faire converger correctement, les deux résidus naturels sont :

- **distance** : `(distance(q) − distance_ref)²` — composante **normale** du contact (profondeur/écart).
- **witness** : la dissemblance entre `witness(q)` et `witness_ref`, deux points **sur la surface** —
  composante **tangentielle** (« où sur la surface on touche »).

Pour le terme witness, la **distance euclidienne** `‖witness(q) − witness_ref‖` est trompeuse sur les
géométries fines/concaves : deux points proches en euclidien (faces opposées d'une plaque, intérieur
/extérieur d'un sac, deux côtés d'une anse) peuvent être loin **sur le mesh**. Si le robot dérive vers
la face d'en face, l'euclidien dit « presque bon » alors que c'est un contact différent. La **distance
géodésique** (le long de la surface) capture correctement « même région de contact », précisément là
où la justesse compte le plus. De plus son gradient (équation eikonale) est de **norme ~1 et tangent à
la surface**, pointant le long du chemin → très bien conditionné pour faire glisser un contact vers la
bonne région.

On veut ce champ **précalculé offline** (géométrie figée d'un mesh rigide) pour que l'online se réduise
à des lookups + une interpolation locale ultra-rapides.

## Périmètre (ce que ce design livre)

1. **Un sous-module `prepare/geodesic/`** (frère de `sdf/`, `point_cloud/`) produisant, par mesh
   (objet ou sol-terrain), une **table all-pairs de distances géodésiques** sur les points de surface
   du `object_cloud`, cachée par géométrie. Asset auto-contenu (porte SES points + normales).
2. **Le contrat `GeodesicTable`** (dans `prepare/contracts.py`) + son rattachement au `Channel`
   (`Channel.geodesic: GeodesicTable | None`).
3. **La config `GeodesicConfig`** (dans `prepare/config.py`) + son intégration à `PrepareConfig`.
4. **Un helper de query pur** `geo_value_grad` (dans `targets/interaction/geodesic.py`, numpy-only,
   ré-exporté par `targets`) : lit le champ géodésique mono-source à un `witness(q)` continu par
   **MLS degré-1**, renvoyant **valeur + gradient** différentiables.

## Non-objectifs (YAGNI / à l'utilisateur)

- **Le résidu de coût géodésique lui-même** (assemblage `geo_value_grad(witness(q)) → résidu`,
  pondération, composition avec les Jacobiennes) — **écrit par l'utilisateur**, hors de ce design
  (cohérent avec le spec « solve / field-eval » : les fonctions de coût sont hors scope).
- **Sol plat** : aucune table (géodésique d'un plan = euclidien, forme close). `Channel.geodesic =
  None` ⇒ le coût retombe sur l'euclidien analytique. Seuls objets + sol-**terrain** ont un asset.
- **Méthode géodésique exacte** (MMP/heat method) : la v1 est une distance de **graphe k-NN** (Dijkstra
  scipy), suffisante pour un coût de contact monotone, et **insensible à la qualité topologique** des
  meshes dataset (souvent non-manifold). L'upgrade `potpourri3d.PointCloudHeatSolver` est documenté
  comme substitution future derrière le même `cache_key`/contrat (cf. « Alternatives »).
- Aucune relocalisation ni suppression de type existant ; aucune modification de `point_cloud/objects.py`.

## Idée pivot — pas une LUT « snap-snap », mais des champs mono-source interpolables

`witness(q)` est un point **continu et fonction de `q`** (sortie interpolée de la grille witness du
SDF) ; `witness_ref` est **fixe** (calculé offline). Une matrice all-pairs dont on **snapperait les
deux bouts** sur le point le plus proche serait **constante par morceaux** ⇒ **gradient nul** ⇒
inutilisable pour un solveur Jacobien. La reformulation qui sauve la différentiabilité :

- Côté **fixe** (`witness_ref`) : snap sur le landmark le plus proche **offline** (constant, pas de gradient).
- Côté **mobile** (`witness(q)`) : lecture du champ `geo[j, ·]` (géodésique depuis le landmark `j`)
  **interpolé** à la position continue par MLS degré-1 → `g` lisse + gradient tangent.

L'**all-pairs sur les P landmarks EST exactement la pile des P champs mono-source** : on précalcule la
matrice, on la **consomme comme des champs**. La ligne `geo[j]` est le champ depuis `j` (lookup O(1),
contigu).

```
OFFLINE (prepare/geodesic, scopé géométrie)        ONLINE (coût utilisateur, q-dép.)
 mesh ─sample_surface_even(seed)─► P points          witness_ref ─nearest_index─► j  (offline, snap)
   + normales (face_index)                            witness(q) ─MLS deg-1 sur geo[j,·]─► (g, ∇g)
   ─graphe k-NN gaté normales─► Dijkstra all-pairs       └─► résidu witness = f(g)   (utilisateur)
   ─► GeodesicTable{points, normals, geo (P,P)}
```

## Placement & fichiers

```
prepare/geodesic/
  __init__.py        # exporte GeodesicBuilder, build_geodesic_table, save_geo, load_geo
  build.py           # fonctions pures + GeodesicBuilder (AssetBuilder)
  cache.py           # save_geo / load_geo (.npz)
targets/interaction/
  geodesic.py        # geo_value_grad, nearest_index (numpy-only, torch-free)
```

- **Modifier** `prepare/contracts.py` : `+ GeodesicTable` ; `Channel += geodesic: GeodesicTable | None = None`.
- **Modifier** `prepare/config.py` : `+ GeodesicConfig` ; `PrepareConfig += geodesic`.
- **Modifier** `prepare/runner.py` : `_build_channels` construit la table par objet/terrain (load-or-build,
  `prof.span("geodesic")`), la passe au `Channel`.
- **Modifier** `prepare/__init__.py` : exporter `GeodesicConfig`/`GeodesicTable` si l'`__init__` ré-expose
  déjà les contrats/config (aligner sur l'existant `SdfConfig`/`SDF`).
- **Modifier** `targets/interaction/__init__.py` + `targets/__init__.py` : ré-export `geo_value_grad`,
  `nearest_index` (à côté de `pose_cloud`/`eval_fields`).
- **Docs de structure** (source unique — `CLAUDE.md`) : **ARCHITECTURE.md**, **PREPARE.md**, **CACHE.md**,
  **TARGETS.md** (voir « Intégration architecture » ci-dessous).

## Contrat — `GeodesicTable` (frozen, numpy-only)

```python
@dataclass(frozen=True)
class GeodesicTable:
    """All-pairs géodésique (distance de graphe k-NN) sur les points de surface du mesh, frame locale.
    Auto-contenu : porte SES points + normales, consommable sans le object_cloud. La ligne geo[j] EST
    le champ géodésique mono-source depuis le point j (lookup O(1), contigu)."""
    points: np.ndarray    # (P, 3) f32  échantillons de surface, frame locale (= sampling object_cloud)
    normals: np.ndarray   # (P, 3) f32  normale unitaire par point (gating snap/interp thin/concave)
    geo: np.ndarray       # (P, P) f32  geo[i,j] = géodésique de graphe i->j (symétrique)
    name: str             # nom de canal ("obj0"/"terrain") — provenance, aligné SDF/cloud
    sampling_id: str = "" # identité du sampling (densité/seed/topo) — provenance/garde-fou

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def __post_init__(self) -> None:
        if self.geo.shape != (self.points.shape[0], self.points.shape[0]):
            raise ValueError(f"geo shape {self.geo.shape} != (P, P) avec P={self.points.shape[0]}")
        if self.normals.shape != self.points.shape:
            raise ValueError(f"normals shape {self.normals.shape} != points shape {self.points.shape}")
```

Rattachement au canal (parallèle exact du `sdf`) :

```python
@dataclass(frozen=True)
class Channel:
    name: str
    object_idx: int | None
    sdf: SDF
    geodesic: GeodesicTable | None = None   # None = sol PLAN (coût → euclidien analytique) ; sinon
                                            # objet/terrain. Le None est ASSUMÉ : le plan est le seul
                                            # cas où la géodésique a une forme close (= euclidien).
```

> Décision : `| None` plutôt qu'une table « plane » factice. C'est la seule entorse au « jamais de None »
> du SDF, et elle est **honnête** — sur un plan la géodésique n'a pas besoin d'être stockée, elle se
> calcule à la volée. Le coût détecte `None → euclidien` (qui EST la géodésique exacte d'un plan).
>
> Le `cKDTree` de snap n'est **pas** dans le contrat (numpy-only/frozen) : le helper de query fait un
> k-NN brute-force vectorisé depuis `points`. `geo` stockée **pleine** `(P,P)` (pas triangle sup) pour
> que `geo[j]` soit une ligne contiguë → champ mono-source en O(1).

## Config — `GeodesicConfig` (frozen, stdlib-only)

```python
@dataclass(frozen=True)
class GeodesicConfig:
    """Graphe géodésique sur le cloud objet (prepare/geodesic). La DENSITÉ/seed ne sont PAS ici : la
    géodésique réutilise l'échantillonnage du object_cloud (CloudConfig) → un seul sampling canonique."""
    k_neighbors: int = 8       # k du graphe k-NN de surface (Dijkstra all-pairs scipy)
    normal_gate: float = 0.0   # arête i--j seulement si dot(n_i,n_j) > normal_gate ∈ [-1,1]
                               # (coupe les arêtes "cross-gap" des plaques fines/concaves ; 0 = même hémisphère)
    max_points: int = 6000     # garde-fou : ValueError si P dépasse (stockage 4·P² o) →
                               # baisser object_density ou relever ce knob en conscience

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {self.k_neighbors}")
        if not -1.0 <= self.normal_gate <= 1.0:
            raise ValueError(f"normal_gate must be in [-1, 1], got {self.normal_gate}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")
```

`PrepareConfig += geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)`.

## Builder — `prepare/geodesic/build.py` (fonctions pures + `GeodesicBuilder`)

Fonctions pures (déterministes ; no I/O ; ne mutent pas leurs inputs ; trimesh/scipy autorisés dans un
builder) :

- `sample_surface_with_normals(vertices, faces, density, seed) -> (points (P,3), normals (P,3))` :
  rejoue **exactement** le sampling du cloud (`n = max(64, int(area*density))`, `sample_surface_even`
  déterministe en `seed`) → points **bit-identiques** au `object_cloud`, + normales via les
  `face_index` retournés (`mesh.face_normals[fid]`). Aucun changement à `point_cloud/objects.py`.
- `build_knn_graph(points, normals, k, normal_gate) -> scipy.sparse.csr_matrix (P,P)` :
  `cKDTree.query(points, k+1)` (le 1er voisin = soi), arêtes `(i, idx[i,1:])` poids euclidien, **gatées
  par normales** (`dot(n_i,n_j) > normal_gate`), **symétrisées** (graphe non orienté).
- `all_pairs_geodesic(graph) -> (P,P) f32` : `scipy.sparse.csgraph.shortest_path(graph, method='D',
  directed=False)` ; `if not np.isfinite(D).all(): raise ValueError("graphe disconnecté — augmente
  k_neighbors / baisse normal_gate")` ; `0.5*(D+D.T)` (nettoie l'asymétrie FP) ; `astype(f32)`.
- `build_geodesic_table(vertices, faces, cloud_cfg, geo_cfg, name="") -> GeodesicTable` : compose ;
  `if len(points) > geo_cfg.max_points: raise ValueError(...)`.

```python
class GeodesicBuilder:
    """AssetBuilder de la table géodésique d'un mesh. Scopé géométrie : la clé hashe le sampling
    (object_density + seed, qui FIXE les points) + les knobs graphe (k, normal_gate). max_points est un
    garde-fou (ne change pas l'asset produit) ⇒ hors clé."""
    def cache_key(self, cloud_cfg, geo_cfg, vertices, faces) -> str:
        h = hashlib.sha1()
        h.update(f"{cloud_cfg.object_density}|{cloud_cfg.seed}|{geo_cfg.k_neighbors}|{geo_cfg.normal_gate}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()
    def build(self, cloud_cfg, geo_cfg, vertices, faces, name="") -> GeodesicTable: ...
    def save(self, t, path): save_geo(t, path)
    def load(self, path): return load_geo(path)
```

## Cache — `prepare/geodesic/cache.py` (calqué sur `sdf/cache.py`)

`np.savez_compressed(points, normals, geo, name, sampling_id)` ; `load_geo` = l'inverse
(`allow_pickle=False`). Persistance dans un seul fichier, writer et reader co-localisés.

## Helper de query — `targets/interaction/geodesic.py` (numpy-only, vectorisé)

```python
def nearest_index(points, xyz):                  # snap witness_ref -> source (OFFLINE), vectorisé
    """argmin ‖xyz - points‖ ; (Q,3)->(Q,) int. Brute-force numpy (pas de scipy côté targets)."""

def geo_value_grad(table, source_idx, query_xyz, k=6):
    """Champ géodésique depuis source_idx, lu à query_xyz (witness(q) continu) par MLS degré-1 : fit
    local pondéré f(y)≈c+b·(y-x) sur les k plus proches points (val=c, ∇=b ; gradient tangent à la
    surface, norme ~1 façon eikonale). Renvoie (g:(Q,), grad:(Q,3)). Différentiable, vectorisé sur Q ;
    reproduit exactement geo[source, m] à un point d'échantillon m. k-NN brute-force numpy → targets
    reste numpy-only/torch-free."""
```

> **MLS degré-1 plutôt qu'IDW** : l'IDW (Shepard) a des cusps (gradient nul) aux points d'échantillon ;
> le fit linéaire pondéré donne **valeur lisse ET gradient propre** (le vecteur `b`), naturellement
> **tangent** (les voisins de surface engendrent le plan tangent). Système `(Q,4,4)` résolu par
> `np.linalg.solve` vectorisé. Le helper est **infrastructure** (pas le coût) ; l'utilisateur le câble
> dans son résidu.

## Câblage runner — extension de `_build_channels`

Réutilise le `(v, f)` déjà chargé pour le SDF ; sous `prof.span("geodesic", n=...)` dans `_run`,
exactement comme `sdf`/`point_cloud` :

```python
geo_builder = GeodesicBuilder()
geo = _load_or_build(geo_builder, "geodesic",
        geo_builder.cache_key(config.cloud, config.geodesic, v, f),
        lambda v=v, f=f, i=i: geo_builder.build(config.cloud, config.geodesic, v, f, name=f"obj{i}"),
        cache_dir, prof, force=force)
channels.append(Channel(f"obj{i}", i, sdf, geodesic=geo))   # sol PLAN → geodesic=None
```

Le sol-terrain (`spec.ground_mesh_path is not None`) reçoit aussi une table ; le sol plat reste
`geodesic=None`.

## Intégration architecture (config + docs de structure)

Le sous-module doit être **branché partout où `sdf` l'est**, pas seulement créé. Checklist d'intégration :

**Plomberie config** — `GeodesicConfig` est un citoyen de première classe de `PrepareConfig` :
- `PrepareConfig += geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)` ⇒ `PrepareConfig()`
  inclut le défaut géodésique ; override inline `PrepareConfig(geodesic=GeodesicConfig(k_neighbors=12))`.
- Le builder lit **uniquement** `config.cloud` (sampling) + `config.geodesic` (graphe) et les hashe dans
  sa clé ⇒ un changement de knob géodésique n'invalide **que** l'asset géodésique.
- Le futur CLI tyro attaché au point d'entrée de run expose donc `--geodesic.k-neighbors`, etc., sans
  travail supplémentaire (il dérive de la dataclass).

**Câblage runtime** — `Channel.geodesic` peuplé dans `_build_channels` (objets + sol-terrain), `None`
pour le sol plat ; `prof.span("geodesic", n=...)` dans `_run` à côté de `sdf`/`point_cloud`.

**Docs de structure à mettre à jour** (CLAUDE.md : `ARCHITECTURE.md` = source unique ; ne pas la redupliquer) :
- **ARCHITECTURE.md** : ajouter `geodesic/` à l'arbre `prepare/` (ligne LIVRABLE, après `point_cloud/`) ;
  `GeodesicConfig` à la liste des sous-configs ; `GeodesicTable` à la liste des types `prepare`.
- **PREPARE.md** : nouvelle ligne LIVRABLE `geodesic/` ; `GeodesicConfig` dans les sous-configs ; passer
  le décompte builders **5 → 6** (`+ GeodesicBuilder`) et livrables **3 → 4** ; mentionner
  `Channel.geodesic` dans la liste des champs `Channel` (« `sdf` TOUJOURS présent ; `geodesic` `| None`,
  `None` = sol plan »).
- **CACHE.md** : ajouter la ligne de table cache **Géodésique** (clé = hash géom + sampling + knobs
  graphe ; sous-dossier `geodesic`) ; entrée `geodesic/<geom_hash>_<cfg>.npz` dans l'arbre
  `HoloV2/cache/` ; règle d'invalidation « changer `geodesic`/`cloud` → rebuild table géodésique ».
- **TARGETS.md** : noter le ré-export `geo_value_grad`/`nearest_index` dans la surface publique
  (`targets/interaction` → `targets`), avec une phrase sur l'usage (lecture différentiable du champ à
  `witness(q)` pour le résidu witness côté solve/utilisateur).

> Garde-fou de cohérence : après l'intégration, `PrepareConfig()` doit construire sans erreur, `prepare`
> sur une scène à ≥1 objet doit produire des `Channel.geodesic` non-`None` pour les objets et `None`
> pour le sol plat, et un second run doit faire **cache hit** sur `geodesic/`.

## Tests (`HoloV2/tests/`)

- **Pure** : (a) points géo == points `object_cloud` (même seed/densité) ; (b) plaque plate →
  géodésique ≈ euclidien (tol sur-estime-graphe) ; (c) cylindre → antipodaux ≈ ½ circonférence ≫
  euclidien ; (d) `geo[i,i]=0`, symétrie ; (e) graphe disconnecté → `ValueError` ; (f) **gating
  normales** : plaque fine, 2 points faces opposées → géodésique « fait le tour » (≫ épaisseur).
- **Builder** (obligatoire CLAUDE.md) : déterminisme (build ×2 identique) + round-trip cache
  (`save`→`load` == `build`).
- **Query** : (a) à un point d'échantillon, `geo_value_grad` reproduit `geo[src, m]` ; (b) gradient
  analytique vs différences finies ; (c) plaque plate → `g ≈ ‖x−src‖`, `∇g` tangent ~unitaire.
- **Contrat** : `GeodesicTable.__post_init__` lève sur formes incohérentes ; `GeodesicConfig`
  `__post_init__` lève sur knobs hors plage.

## Conformité aux règles d'or (`CLAUDE.md`)

- **#1 dépendances à sens unique** : `prepare` produit l'asset ; `targets` expose le helper pur ; le
  coût (aval, `solve`/utilisateur) consomme. Aucun cycle ; aval importe la surface publique amont.
- **#2 chaque étage possède SES types + SA config** : `GeodesicTable` ∈ `prepare/contracts.py`,
  `GeodesicConfig` ∈ `prepare/config.py`. Pas de duplication.
- **#3 cœur pur, effets aux extrémités** : builder pur ; I/O cache dans le runner (`_load_or_build`).
- **#4 data-oriented** : SoA numpy `(P,P)` f32, tout le lourd amorti offline ; online = lookups + petit
  solve vectorisé.
- **#5 observabilité aux seams** : `prof.span("geodesic")` dans le runner seul.
- **#7 critère de découpe = dépendance à `q`** : la table est `q`-indépendante (géométrie rigide) ⇒
  `prepare`. La query interpole à `witness(q)` ⇒ consommée online.
- **#8 homogénéité** : `Channel.geodesic` parallèle à `Channel.sdf` ; même builder objets ET terrain.
  Le `None` du sol plat est l'unique cas à forme close (assumé, documenté).
- **#9 abstraction MAIS YAGNI** : méthode graphe simple en v1 ; substitution heat method documentée
  derrière le même contrat, **pas** implémentée tant qu'un besoin de précision ne l'exige.
- **Poids des imports** : `trimesh`/`scipy` uniquement dans le builder ; `targets/interaction/geodesic.py`
  numpy-only/torch-free ; `contracts.py`/`config.py` numpy-/stdlib-only.

## Alternatives écartées (et chemin d'upgrade)

- **A\* au lieu de Dijkstra** : A\* accélère le **point-à-point** ; notre build a besoin du
  **single-source → tous** (le champ entier par source). Quand la cible = tous les nœuds, l'heuristique
  admissible vaut 0 ⇒ A\* dégénère en Dijkstra. All-pairs = P balayages Dijkstra (scipy, C vectorisé),
  strictement mieux que P² requêtes A\*.
- **`igl.heat_geodesic`/`exact_geodesic` (installé)** : mesh-based ⇒ sensible à la qualité topologique
  (meshes dataset souvent non-manifold) + mapping cloud↔mesh à gérer. Écarté pour la robustesse.
- **`potpourri3d.PointCloudHeatSolver` (nouvelle dép)** : heat method natif point-cloud, plus précis/
  robuste. **Chemin d'upgrade** documenté : substituable derrière `GeodesicBuilder.build` sans toucher
  au contrat `GeodesicTable` ni à l'aval, si le gradient interpolé de la v1-graphe s'avère insuffisant.
- **Table « plane » factice pour zéro-None** : rejetée — stocker l'euclidien d'un plan est inutile ;
  `geodesic=None` est l'expression honnête du cas à forme close.
- **Snapper aussi `witness(q)`** : tuerait le gradient (constant par morceaux) ; on interpole (MLS).

## Risques & garde-fous

- **Short-circuit Isomap** (plaque plus fine que l'espacement d'échantillonnage) : neutralisé par le
  **gating normales** ; testé explicitement (test f).
- **Graphe disconnecté** (`k`/`normal_gate` trop agressifs) : `ValueError` explicite au build (pas de
  `inf` stocké).
- **Stockage `4·P²`** : garde-fou `max_points` ; densité pilotée par `CloudConfig.object_density`.
- **Qualité du gradient interpolé** : MLS degré-1 (lisse, tangent) ; si insuffisant → upgrade heat method.
```