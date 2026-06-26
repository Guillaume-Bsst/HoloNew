# HoloV2 — Cache des assets build-once (`HoloV2/cache/`)

Les sorties de `prepare/` sont **régénérables mais coûteuses** → cachées. Principe central :
**rien n'est caché « par scène » ; chaque item est caché à la granularité de SES
dépendances**, et tout dépend d'une **`Config`** (`contracts.Config`). Une scène est
**assemblée** à partir d'items cachés individuellement.

## Ce qui est caché, et de quoi ça dépend (la clé)

| Item | Inputs (données) | Sous-config | Amont | Portée (partagé entre) |
|---|---|---|---|---|
| **Calibration** | sujet (betas, genre) + prise | `calibration` | — | (sujet, prise) |
| **GridSDF objet** | mesh objet (hash géom) | `sdf` | — | toutes scènes avec cet objet |
| **Ground (plat)** | — (`PlaneField` analytique, défaut) | — | — | **non caché** (pas de grille) |
| **Ground (terrain)** | mesh terrain (hash géom) | `sdf` | — | toutes scènes avec ce terrain |
| **Nuage objet** | mesh objet (hash géom) | `cloud` | — | toutes scènes avec cet objet |
| **Nuage humain** | sujet (betas, genre, body-model) | `cloud` | — | par sujet |
| **Correspondance** | robot (id) + template + surface G1 | `correspondence` **+ `cloud`** | nuage (échantillonnage) | (robot, template) |

→ GridSDF/nuage objet = **par géométrie** (réutilisés partout) ; nuage humain = par sujet ;
correspondance = par (robot, template) ; calibration = par (sujet, prise). Le **ground** n'est
PAS une vérité constante : par défaut un `PlaneField` plat (non caché), mais il **reflète
l'environnement** (escalier/pente/climbing) via un `GridSDF` de terrain — alors caché comme un
objet (`SceneSpec.ground_mesh_path`). On ne recalcule jamais un asset partagé pour chaque scène.

## La chaîne de dépendance (le piège)

La **correspondance** est bâtie SUR le nuage (elle apparie des points par identité via
`smpl_idx`). Sa clé **inclut donc `cloud`** (densité/seed). Le `seed` d'échantillonnage est
partagé par le nuage humain ET la correspondance : **ils doivent s'accorder** (sinon
`smpl_idx` pointe vers un autre ordre de points). Idem en cascade : `cloud` change ⇒ nuage +
correspondance invalidés.

**Garde-fou typé** : `PointCloud.sampling_id` (humain) doit égaler
`CorrespondenceTable.smpl_sampling_id` — `runner.prepare` l'assert à l'assemblage (sinon
transport silencieusement faux).

**Déterminisme** : les builders DOIVENT être déterministes (seed fixe partout, OT
déterministe) — sinon un cache-hit diffère d'un rebuild. Invariant à tester.

## Le mécanisme de clé

```
cache_key(item) = hash( sous-config pertinente  +  inputs pertinents  +  clés des items amont )
```
- **inputs** : géométrie → hash des verts/faces (ou du fichier mesh) ; sujet → bytes betas +
  genre ; robot → nom/urdf.
- chaque builder ne hashe **que ses dépendances** → changer un param n'invalide **que** les
  items touchés (et leur aval). `Config` est `frozen` + hashable.
- `AssetBuilder.cache_key(config, *inputs) -> str` (contrat dans `contracts.py`).

## Layout

```
HoloV2/cache/
  calibration/    <subject>_<take>_<cfg>.npz
  sdf/            <geom_hash>_<cfg>.npz            (objets — le sol est analytique, non caché)
  cloud/
    human/        <subject>_<cfg>.npz
    object/       <geom_hash>_<cfg>.npz
  correspondence/ <robot>_<template>_<cfg>.npz
  # sidecar <name>.json par item : provenance lisible (config + inputs résolus) pour debug
```
Gitignore : `cache/**` ignoré sauf le défaut `correspondence/corr_neutral.npz` (committé).

## Assemblage d'une scène

```
prepare/runner.prepare(scene_spec: SceneSpec, config: Config) -> (GroundedScene, InteractionContext, Calibration)
```
- `SceneSpec` = (dataset, séquence, `RobotSpec`, model dirs) ; le loader en tire sujet/objets/prise.
- le runner dérive la clé de chaque item depuis `SceneSpec + Config`, fait **load-or-build**
  pour chacun, **assert** `sampling_id` ↔ `smpl_sampling_id`, puis **assemble** les 3 sorties.

## Cibles bakées (optionnel)

Les `targets/` (per-frame) peuvent être recalculées en ligne depuis les assets cachés, OU
**bakées** (séquence de `FrameTargets`). Une bake dépend de la config **complète** (tout
l'amont + `margin`/canaux), donc sa clé = hash de `Config` entier + `scene_spec`. À placer
sous `cache/targets/<scene>_<cfg>.npz` si on l'active.

## Invalidation — résumé
- changer `sdf` → rebuild SDF seulement.
- changer `cloud` → rebuild nuages **et** correspondance (aval).
- changer `calibration` → rebuild calibration seulement.
- changer betas (sujet) → rebuild nuage humain + calibration ; SDF/correspondance(template) intacts.
- changer un objet (géométrie) → rebuild SDF + nuage de cet objet ; reste intact.
