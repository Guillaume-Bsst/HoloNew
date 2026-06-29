# HoloV2 — Étape 1 `prepare/` (et le concept q-indépendant : prepare + targets)

## 1. Concept (2 étages)

### Étage 1 — Loading + Grounding (indissociables) = CALIBRATION
But : produire la **scène grounded** = scène cible de référence. Contient le
**modèle SMPL complet** (mesh, pas juste les joints) + les **meshes objets**,
posés cohéremment sur le sol.
C'est de la **calibration** : faite **offline / en amont** (en téléopération, les
gens la feraient aussi avant). Sortie = un jeu de paramètres figés.

### Étage 2 — 2 traitements ORTHOGONAUX depuis la scène grounded
Ils ne se connaissent pas ; consomment tous deux la scène grounded ; alimentent
**un seul solve**.

**A — style** (ex-« body ») : suivi de posture, ignore l'objet. Adapté G1 via le
**mapping articulaire** (provisoire — à reconcevoir, voir `targets/style/`).

**B — interaction** : SDF → nuages de points → éval multi-canal → transport
SMPL→G1. Adapté G1 via la **correspondance**.

```
                          ┌─► [A] style ──────────────► StyleTargets ──────────────┐
SCÈNE GROUNDED ───────────┤                                                         ├─► SOLVE
(SMPL complet + objets)   └─► [B] interaction ─► RobotInteractionTargets (G1) ──────┤
                                              └─► EnvironmentInteractionTargets ─────┘
```

---

## 2. Frontière OFFLINE / ONLINE (contrainte d'archi first-class)

Tout est conçu pour un usage **online streaming** (1 frame brute → cibles
rapidement). Deux phases, seam explicite :

**Phase calibration (offline, une fois)** — coûteux et stable :
scale · floor offset · cadrage root · table de correspondance SMPL→G1 ·
**SDF des objets/terrain (frame local — rigide ⇒ calculés 1 fois ; le sol plat = SDF de plan exact, non caché)** ·
échantillonnage des nuages · BVH/KD-trees. Tout ça est **cacheable sur disque**,
clé = géométrie ; deux séquences partageant un objet partagent l'asset.
Stockage : **`HoloV2/cache/`** (hors du package python, géré par `prepare/` via un
chemin de config ; régénérable → gitignoré, sauf le défaut `correspondence/corr_neutral.npz`).

**Phase per-frame (online, cheap)** — que de l'application :
appliquer la calibration · poser les nuages (skinning SMPL / rigide objet) ·
transformer en frame-local → **sampler la grille SDF** (trilinéaire, zéro Coal) ·
gather via la correspondance.

API du seam — chaque traitement expose **deux** points d'entrée :
```python
prepare(scene, calibration) -> Context        # offline, build-once
process_frame(ctx, raw_frame) -> Targets      # online, pur, cheap
# l'offline n'est que :  for f in seq: process_frame(ctx, f)
```

**3 règles qui garantissent l'online** (sinon le seam est faux) :
1. Opérations séquence-globales (floor drop = min sur la séquence) → **paramètre
   de calibration** figé en amont, jamais recalculé par frame.
2. `trimesh.contains` par frame (inside/outside, lent) **interdit** online → le
   signe vient de la **grille SDF** précalculée.
3. Signaux temporels (foot-sticking, etc.) **causaux** (fenêtre glissante, passé
   seulement), jamais de look-ahead.

**Règle structurante du multi-canal** : *le humain est toujours une source de
probes, jamais un champ* (il se déforme → pas de SDF cheap). Un nuage ne peut
donc être évalué **online que contre un SDF rigide** (sol + objets).

---

## 3. L'évaluation = matrice homogène (clouds × canaux)

- Canaux = sol + N objets = **N+1** (TOUS des SDF ; sol : SDF de plan par défaut, ou SDF terrain ; objets : SDF)
- Clouds = SMPL + N objets = **N+1 nuages**
- Chaque cloud est évalué contre **tous** les canaux → chaque point porte **N+1
  canaux**. Sortie homogène, zéro cas particulier. Le canal « self » (cloud objet
  *i* vs son SDF *i*) ≈ 0 — gardé pour l'homogénéité, masqué à l'usage si besoin.

---

## 4. Arborescence

Trois étapes top-level : `prepare/` (offline) → `targets/` (online) → `solve/`. Chaque étage possède
sa surface publique co-localisée : `contracts.py` (types) + `config.py` (knobs). SMPL/meshes ne sont
chargés/instanciés QUE dans `prepare/`.

Entrée : **`SceneSpec`** (data identity : dataset, séquence, `RobotSpec`, model dirs) — distinct
de **`PrepareConfig`** (knobs d'algo ; schéma `prepare/config.py`). `PrepareConfig()` donne le défaut,
override inline `PrepareConfig(sdf=SdfConfig(spacing=0.005))`. Le loader transforme `SceneSpec` -> `RawMotion`.

```
src/                    (sous `HoloV2/` ; chaque étage porte SES types + SA config, co-localisés)
  prepare/             ÉTAPE 1 — TOUT l'offline (seul endroit qui instancie SMPL/meshes/robot)
    contracts.py         SORTIE PUBLIQUE — types de DONNÉES de prepare (protocols BodyModel/RobotModel/AssetBuilder ;
                         SceneSpec/RobotSpec, RawMotion/SmplParams, ObjectMesh/Calibration/GroundedScene, SDF/Channel,
                         GeodesicTable, PointCloud/CorrespondenceTable, InteractionContext) — pas la config — numpy-only
    config.py            knobs : PrepareConfig + sous-configs CalibrationConfig/SdfConfig/CloudConfig/CorrespondenceConfig/GeodesicConfig
    load/                loaders OFFLINE (sous-package)
      base.py              protocol MotionLoader + registre
      datasets/            un loader par dataset (omomo, hodome, sfu, hoim3) -> RawMotion (params + chemins)
      smpl.py              SmplParams -> BodyModel (instancie SMPL, FK os)
      smpl2smplx.py        transfert de paramètres SMPL -> SMPL-X
      mesh.py              chemin -> (verts, faces) local (trimesh ; poses ajoutées à l'assemblage scène)
      robot.py             RobotSpec -> RobotModel (FK yourdfpy, AGNOSTIQUE) + pose de repos correspondance (keyée par robot)
      frames.py            conventions de frame partagées
    # --- les 4 LIVRABLES (build-once) ; persistance UNIFORME : save_<asset>/load_<asset> co-localisés, le builder délègue ---
    calibration/         grounding ROBOT-FREE & BODY-FREE : human_offset (foot-joint pct) + object_offset (objets partagé) + root
                         ; expose save_calibration/load_calibration (CalibrationBuilder délègue)
    sdf/                 meshes objets/terrain -> SDF (caché) ; sol plat -> SDF de plan (build_plane_sdf, non caché)
                         ; expose save_sdf/load_sdf (SdfBuilder délègue)
    point_cloud/         NUAGES + correspondance
      sampling.py          SurfaceSampling (tri_idx,bary,sampling_id) — échantillonnage canonique PARTAGÉ
      human.py             surface SMPL -> PointCloud (skinning creux K~4) ; HumanCloudBuilder (par sujet)
      objects.py           surface objet -> PointCloud rigide K=1 ; ObjectCloudBuilder (par géométrie)
      cache.py             save+load .npz du PointCloud (partagé par les 2 builders, qui délèguent)
      correspondence/      humain↔robot par OT par-segment (ROBOT-AGNOSTIQUE)
        segments.py          15 segments + mapping joint/lien->segment + label des samples humains
        robot_surface.py     échantillonne la surface robot (shell watertight) par lien -> RobotSurface
        ot_couple.py         OT entropique par segment (POT) -> smpl_idx (main↔main, pied↔pied)
        build.py             CorrespondenceBuilder : génère sampling + source humaine neutre + OT
                             -> (CorrespondenceTable, SurfaceSampling) ; regenerate() régénère corr_neutral.npz
        cache.py             save+load .npz de la correspondance, ensemble -> (CorrespondenceTable, SurfaceSampling)
    geodesic/            meshes objets/terrain -> table géodésique all-pairs (cachée) ; sol plat -> aucune table
                         ; expose save_geo/load_geo (GeodesicBuilder délègue). Réutilise le sampling du
                         object_cloud (même densité/seed → points bit-identiques), scopé géométrie.
                         **Note** : contrairement à SDF/cloud (agnostiques à la connectivité), ce livrable LÈVE
                         sur un graphe k-NN déconnecté (mesh mince/non-manifold) — un mesh problématique peut donc
                         interrompre ``prepare`` et nécessite un ajustement de ``GeodesicConfig.k_neighbors`` /
                         ``normal_gate`` pour la scène concernée.
      build.py             GeodesicBuilder + build_geodesic_table : k-NN gaté normales + Dijkstra all-pairs (scipy)
      cache.py             save_geo/load_geo — sérialisation .npz de GeodesicTable (points+normals+geo)
    scene.py             applique la calibration -> GroundedScene
    runner.py            prepare(scene_spec, config) : load-or-build + assemble ; build_all()
                         Les meshes objets sont chargés UNE FOIS dans _run(), puis partagés par le build
                         SDF/géodésique (_build_channels) ET le build object_cloud — chaque fichier mesh
                         est lu une seule fois par prepare.
    # sorties : Calibration + SDF(objets/terrain) + PointClouds(+corr) -> GroundedScene + InteractionContext

  targets/             ÉTAPE 2 — construction ONLINE des 2 canaux de cibles
    contracts.py         SORTIE PUBLIQUE — types de targets (ContactField/MultiChannelField, StyleTargets,
                         Robot/EnvironmentInteractionTargets, FrameTargets, FramePose, FrameTrace) ;
                         importe la sortie publique de prepare via `..prepare.contracts`
    config.py            knobs de targets (à venir)
    style/               demo joints -> StyleTargets (mapping articulaire, provisoire)
    interaction/         pointclouds · eval · transport · targets
      …                    -> RobotInteractionTargets + EnvironmentInteractionTargets
    pipeline.py          process_frame -> FrameTargets ; trace_frame -> FrameTrace

  viz/                 viewer (lit FrameTrace + assets prepare)
  solve/               ÉTAPE 3 (plus tard) — contracts.py + config.py + point d'entrée
```

**Dépendances (acyclique par construction)** : pipeline linéaire, deps aval seulement · `prepare/`
instancie SMPL/meshes et produit {GroundedScene, InteractionContext} (calib dans `grounded.calibration` ; exposés par
`prepare/contracts.py`) · `targets/` les consomme **via leur type** (`from ..prepare.contracts import
…`, jamais le code interne de `prepare/`) et produit `FrameTargets` · `solve/` consomme `FrameTargets`
(`from ..targets.contracts import …`). Aucun cycle ; SMPL/meshes jamais touchés hors `prepare/`.

**Note** : les builders d'assets d'interaction (`sdf`, `correspondence`) vivent dans
`prepare/`, séparés de la logique online (`interaction/eval`, `transport`) restée
dans `targets/`. Lien = uniquement le type d'asset (dans `prepare/contracts.py`), pas de
couplage de code.

**Binding nuage humain ↔ correspondance** (subtilité clé). `CorrespondenceTable.smpl_idx`
indexe l'**ordre des points** du nuage humain. La correspondance est bâtie sur un humain **neutre**
(template) alors que le nuage runtime est celui du **sujet** ⇒ les deux doivent partager le **même
échantillonnage** `(tri_idx, bary)` (= `SurfaceSampling`, identité portée par `sampling_id`). Le nuage
sujet **ne ré-échantillonne donc pas** : il reprend ce `SurfaceSampling` et recalcule seulement son
skinning (offsets `= rest_point − rest_joint[os]` dans le frame natif — la rotation de repos des os,
pur `Q`, s'annule). La correspondance est **construite** par `correspondence/build.py` (OT par-segment
sur un humain NEUTRE) qui **génère** le `SurfaceSampling` et l'**embarque** dans `corr_neutral.npz` ;
`correspondence/cache.py` le relit, et `human.py` le réutilise sur le mesh du SUJET. Garde-fou
`sampling_id == smpl_sampling_id` asserté au runner (cf. `CACHE.md`).

**Contrat commun des builders offline** — **4 livrables** (calibration, sdf, point_cloud, geodesic) produits par
**6 builders** : `CalibrationBuilder`, `SdfBuilder`, `HumanCloudBuilder`, `ObjectCloudBuilder`,
`CorrespondenceBuilder` (le livrable point_cloud en regroupe 3), `GeodesicBuilder`. Tous suivent le même protocol ; la
persistance est co-localisée (`save_<asset>`/`load_<asset>` module-level, le builder délègue) :
```python
class AssetBuilder(Protocol):
    def cache_key(self, config, *inputs) -> str: ...  # hash(sous-config pertinente + inputs)
    def build(self, config, *inputs) -> Asset: ...    # calcul offline lourd
    def load(self, path) -> Asset: ...
    def save(self, asset, path) -> None: ...
```

---

## 5. Les classes (contrats)

**Source de vérité : le `contracts.py` de chaque étage** — pas de duplication ici (zéro drift).
Inventaire, par module.

### `prepare/contracts.py` (sortie publique de prepare)
- **Protocols** : `BodyModel` (+ `stature`, propriété rest-mesh du sujet), `RobotModel`, `AssetBuilder`
- **entrée (data identity)** : `RobotSpec`, `SceneSpec`. La **config** (`PrepareConfig` + sous-configs)
  n'est PAS un contrat de données : elle vit dans `prepare/config.py` (`PrepareConfig()` = défaut,
  override inline `PrepareConfig(sdf=SdfConfig(spacing=0.005))`).
- **load** : `SmplParams` (avec MAINS), `RawMotion` (J_demo)
- **scène / calib** : `ObjectMesh` (+ `static`), `Calibration` (grounding seul, body-free), `GroundedScene`
  (porte `body` = moteur de posage live + `calibration` ; meshes objets = PATHS)
- **champs (assets)** : `SDF` (grille, objets/terrain) · `Channel` (`object_idx` + `sdf` TOUJOURS
  présent ; `geodesic: GeodesicTable | None`, `None` = sol plan → coût euclidien analytique exact) ·
  `GeodesicTable` (table all-pairs (P,P), objets/terrain ; points + normales + matrice géodésique)
- **nuages** : `PointCloud` (skinning creux + `sampling_id`), `CorrespondenceTable`
  (`smpl_idx`/`link_idx`/`offset_local` + `smpl_sampling_id` qui doit matcher `sampling_id`)
- **contexte** : `InteractionContext` (`channels` = ground + objets ; invariants documentés). `InteractionContext` porte aussi `robot_cloud` (M points de correspondance en nuage K=1, ordre FK) + `robot` (moteur FK, symétrique à `GroundedScene.body`) pour la réévaluation online dans `solve`.

### `targets/contracts.py` (sortie publique de targets ; importe les types prepare amont)
- **champs per-frame** : `ContactField` · `MultiChannelField` (`(C,P)`)
- **cibles** : `StyleTargets` (provisoire), `RobotInteractionTargets` (field SEUL — binding statique
  dans le context), `EnvironmentInteractionTargets`, `FrameTargets`
- **état / visu** : `FramePose` (`bone_rot`/`bone_pos`, J_bones), `FrameTrace`

---

## 6. Décisions encore ouvertes

1. ~~Convention de scale dans `calibration`~~ TRANCHÉ : `calibration` est ROBOT-FREE **et BODY-FREE**
   (grounding seul) ; la **stature** vit sur le `BodyModel` (`body.stature`, sujet réel, betas-FK — pure
   propriété du rest mesh). La scale humain→robot = `robot_height / body.stature` est une grandeur de la
   PAIRE (humain, robot), possédée et appliquée par la couche `correspondence`/`transport` (là où les
   deux surfaces se rencontrent), jamais bakée dans la scène — exposée comme `InteractionContext.scale`
   le jour où `transport` est codé (différée d'ici là, YAGNI).
2. Datasets concrets pris en charge par `load/` (plusieurs d'emblée).
3. Contenu réel de l'objectif de `style` (à reconcevoir).
