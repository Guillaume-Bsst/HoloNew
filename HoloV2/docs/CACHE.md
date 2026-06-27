# HoloV2 — Cache des assets build-once (`HoloV2/cache/`)

Les sorties de `prepare/` sont **régénérables mais coûteuses** → cachées. Principe central :
**rien n'est caché « par scène » ; chaque item est caché à la granularité de SES
dépendances**, et tout dépend d'une **`PrepareConfig`** (schéma `config_types/prepare.py`, valeurs via
la factory `default_prepare_config()` de `config_values/prepare.py`). Une scène est
**assemblée** à partir d'items cachés individuellement.

## Ce qui est caché, et de quoi ça dépend (la clé)

| Item | Inputs (données) | Sous-config | Amont | Portée (partagé entre) |
|---|---|---|---|---|
| **Calibration** | sujet (betas, genre) + prise | `calibration` | — | (sujet, prise) |
| **SDF objet** | mesh objet (hash géom) | `sdf` | — | toutes scènes avec cet objet |
| **Ground (plat)** | emprise XY scène (défaut) | `sdf` | — | **non caché** (SDF de plan exact, rebuild instantané) |
| **Ground (terrain)** | mesh terrain (hash géom) | `sdf` | — | toutes scènes avec ce terrain |
| **Nuage objet** | mesh objet (hash géom) | `cloud` | — | toutes scènes avec cet objet |
| **Nuage humain** | sujet (betas, genre, body-model) | `cloud` | — | par sujet |
| **Correspondance** | robot (id) + template + surface G1 | `correspondence` **+ `cloud`** | nuage (échantillonnage) | (robot, template) |

→ SDF/nuage objet = **par géométrie** (réutilisés partout) ; nuage humain = par sujet ;
correspondance = par (robot, template) ; calibration = par (sujet, prise). Le **ground** n'est
PAS une vérité constante : plat par défaut (**SDF de plan** exact via `build_plane_sdf`, non caché —
rebuild instantané), mais il **reflète l'environnement** (escalier/pente/climbing) via une **SDF de
terrain** — alors cachée comme un objet (`SceneSpec.ground_mesh_path`). On ne recalcule jamais un
asset partagé pour chaque scène.

## La chaîne de dépendance (le piège)

La **correspondance** est bâtie SUR le nuage (elle apparie des points par identité via
`smpl_idx`). Sa clé **inclut donc `cloud`** (densité/seed). Le `seed` d'échantillonnage est
partagé par le nuage humain ET la correspondance : **ils doivent s'accorder** (sinon
`smpl_idx` pointe vers un autre ordre de points). Idem en cascade : `cloud` change ⇒ nuage +
correspondance invalidés.

**Garde-fou typé** : `PointCloud.sampling_id` (humain) doit égaler
`CorrespondenceTable.smpl_sampling_id` — `runner.prepare` l'assert à l'assemblage (sinon
transport silencieusement faux).

**État** : la correspondance est **reconstruite en V2** par `correspondence/build.py` (OT
par-segment ; `prepare/load/robot.py` échantillonne la surface robot). Son builder **génère**
l'échantillonnage `(tri_idx, bary)` sur l'humain neutre et l'**embarque** dans `corr_neutral.npz`
(régénérable : `python -m src.prepare.point_cloud.correspondence.build`) ; `cache.py` le relit et
le nuage humain (sujet) le réutilise. Conforme à la chaîne ci-dessus (correspondance bâtie SUR le
nuage/sampling) ; garde-fou `sampling_id`↔`smpl_sampling_id` inchangé.

**Déterminisme** : les builders DOIVENT être déterministes (seed fixe partout, OT
déterministe) — sinon un cache-hit diffère d'un rebuild. Invariant à tester.

## Le mécanisme de clé

```
cache_key(item) = hash( sous-config pertinente  +  inputs pertinents  +  clés des items amont )
```
- **inputs** : géométrie → hash des verts/faces (ou du fichier mesh) ; sujet → bytes betas +
  genre ; robot → nom/urdf.
- chaque builder ne hashe **que ses dépendances** → changer un param n'invalide **que** les
  items touchés (et leur aval). `PrepareConfig` et ses sous-configs sont `frozen` + hashables.
- `AssetBuilder.cache_key(config, *inputs) -> str` (protocol dans le package `contracts/`).

## Layout

```
HoloV2/cache/
  calibration/    <subject>_<take>_<cfg>.npz
  sdf/            <geom_hash>_<cfg>.npz            (objets/terrain ; sol plat = SDF de plan, non caché)
  cloud/
    human/        <subject>_<cfg>.npz
    object/       <geom_hash>_<cfg>.npz
  correspondence/ <robot>_<template>_<cfg>.npz
```
(Un sidecar `<name>.json` de provenance — config + inputs résolus — pourra éventuellement être ajouté
par le runner plus tard ; rien ne l'écrit aujourd'hui.)
Gitignore : `cache/**` ignoré sauf le défaut `correspondence/corr_neutral.npz` (committé).

## Assemblage d'une scène

```
prepare/runner.prepare(scene_spec: SceneSpec, config: PrepareConfig) -> (GroundedScene, InteractionContext, Calibration)
```
- `SceneSpec` = (dataset, séquence, `RobotSpec`, model dirs) ; le loader en tire sujet/objets/prise.
- le runner dérive la clé de chaque item depuis `SceneSpec + PrepareConfig`, fait **load-or-build**
  pour chacun, **assert** `sampling_id` ↔ `smpl_sampling_id`, puis **assemble** les 3 sorties.

## Cibles bakées (optionnel)

Les `targets/` (per-frame) peuvent être recalculées en ligne depuis les assets cachés, OU
**bakées** (séquence de `FrameTargets`). Une bake dépend de la config **complète** (tout l'amont
`PrepareConfig` + la future config `targets`, margin/canaux), donc sa clé = hash de ces deux configs
+ `scene_spec`. À placer sous `cache/targets/<scene>_<cfg>.npz` si on l'active.

## Invalidation — résumé
- changer `sdf` → rebuild SDF seulement.
- changer `cloud` → rebuild nuages **et** correspondance (aval).
- changer `calibration` → rebuild calibration seulement.
- changer betas (sujet) → rebuild nuage humain + calibration ; SDF/correspondance(template) intacts.
- changer un objet (géométrie) → rebuild SDF + nuage de cet objet ; reste intact.
