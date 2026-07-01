# PA-HOI loader (avec trajectoire objet) — design

**Date** : 2026-07-01
**Étage** : `prepare/load`
**But** : brancher le dataset PA-HOI (Physics-Aware HOI, Noitom mocap) dans HoloV2 —
humain SMPL-X **+ trajectoire d'objet 6-DoF par frame** — via un nouveau `@register_loader("pahoi")`.

## Contexte dataset

Racine canonique (déjà en place, même convention que les autres datasets) :
`data/00_raw_datasets/PA-HOI_Dataset/`

```
Mocap_data/
  cap_res_bvh_s1/<seq>/            # sujet 1 ; <seq> = "1_001".."1_282"
    <seq>.npz                      # params SMPL-X DIRECTS (voir clés ci-dessous)
    <seq>_concat.npz               # variante AMASS poses(T,165) — NON utilisée
    <seq>.npy, <seq>_worldpos.csv  # squelette mocap Noitom 72 joints — NON utilisé
    <seq>.bvh, <seq>_reg_finger*   # NON utilisés
  cap_res_bvh_s2/<seq>/            # sujet 2 ; <seq> = "2_001"..
  cap_res_fbx/<seq>_h.fbx          # humain FBX (Noitom) — NON utilisé
  cap_res_fbx/<seq>_o.fbx          # OBJET FBX animé — SOURCE DE LA TRAJECTOIRE
  frames.txt / frame_sub2.txt      # <seq>:<nframes>  <objet>  <poids>  <action>
Object_mesh/<name>.fbx             # meshes HD canoniques — NON utilisés (voir décision mesh)
Object_attributes.json             # {objet: {size, weight, shape}} — métadonnée, hors périmètre loader
texts/                             # annotations langage — hors périmètre
```

### Clés du `<seq>.npz` (params SMPL-X natifs, Y-up)
`global_orient (T,3)`, `body_pose (T,21,3)`, `lhand_pose (T,15,3)`, `rhand_pose (T,15,3)`,
`jaw_pose (T,3)`, `leye_pose (T,3)`, `reye_pose (T,3)`, `betas (10,)`, `expression (T,10)`,
`transl (T,3)`. **Attention** : clés `lhand_pose`/`rhand_pose` (≠ HODome `left_hand_pose`), et
poses en `(T,J,3)` à aplatir en `(T,J*3)`.

### `<seq>_o.fbx` (prouvé par probe pur-Python)
FBX **binaire v7500**, export **Noitom**. Un seul nœud `Model` (Mesh) nommé `<idx>_<objet>`
(ex. `01_milkbox`), avec :
- **Geometry** statique : mesh proxy **24 verts** (boîte englobante) dans le repère local objet.
- **AnimStack/AnimLayer** : nœuds `T` (Lcl Translation) et `R` (Lcl Rotation), chacun 3 `AnimCurve`
  (`d|X/d|Y/d|Z`). **1 key par frame** (208 keys pour T=208) @ **30 fps**. Ordre de rotation = XYZ
  par défaut (aucun Pre/Post/Geometric ni Lcl Scaling animé).
- Unités : translation **cm**, rotation **degrés**, axes **Y-up** (natif, comme les params SMPL-X).

## Alignement des repères (dé-risqué empiriquement)

Le `.npz` SMPL-X et le `_o.fbx` sont **dans le même monde natif Y-up**, à l'échelle cm→m près.
Vérifié par Umeyama `Hips(_worldpos.csv, cm) → transl(.npz, m)` sur 1_001/1_050/1_200 :
**R ≈ identité, scale ≈ 0.01, résidu ~1 cm** (le résidu = écart marqueur-Hips vs joint-pelvis, du
bruit, PAS un désalignement d'axe). ⇒ **aucune correction par-séquence** : l'objet suit exactement
le même `YUP_TO_ZUP` que l'humain (cf. `load/frames.py`), déjà utilisé par HODome/HOI-M3.

## Architecture

Deux ajouts, zéro modif des contrats existants (le contrat `RawMotion` couvre déjà objets+params).

### 1. `src/prepare/load/fbx.py` — lecteur FBX binaire minimal (numpy-only, zéro dépendance)
Pur-Python (aucune lib FBX dispo dans l'env ; règle « aucune dépendance tierce »).
Surface publique unique :

```python
def read_object_fbx(path: Path) -> ObjectFbx: ...
# ObjectFbx (frozen, numpy-only) :
#   rot_native   (T, 3, 3)  # rotation objet, repère natif Y-up
#   transl_native(T, 3)     # translation objet, mètres, repère natif Y-up
#   vertices     (V, 3)     # mesh proxy, mètres, repère LOCAL objet
#   faces        (F, 3) int
#   name         str        # "milkbox" (index de préfixe retiré)
#   fps          float
```

Interne (fonctions pures) :
- `_parse(bytes) -> node tree` : records FBX v7500 (offsets uint64), props primitives/array
  (zlib), string/raw ; sentinelle null-record 25 o.
- Marcher `Objects` → `Model`/`Geometry`/`AnimationCurveNode`/`AnimationCurve` ; `Connections`
  (`OP` curve→curvenode via `d|X/Y/Z`, curvenode→model via `Lcl Translation`/`Lcl Rotation`).
- `Geometry.Vertices` → `(V,3)` ; `PolygonVertexIndex` → triangulation fan (indices négatifs =
  fin de polygone en complément à un). ×0.01 (cm→m).
- Courbes → `(T,3)` translation (×0.01) + `(T,3)` Euler deg. `R_native = scipy 'XYZ'` (ordre FBX
  par défaut ; convention validée par test — voir Risques).
- Échantillonnage : si les 3+3 courbes ont le même nb de keys, lecture directe ; sinon interp
  linéaire sur la timeline `KeyTime` (fallback défensif).

### 2. `src/prepare/load/datasets/pahoi.py` — `@register_loader("pahoi")`
Calque de `hodome.py` :
```python
params = _smpl_params(d)                 # npz DIRECT (reshape (T,J,3)->(T,J*3), clés lhand/rhand)
body   = build_body_model(params, spec.smpl_model_dir)
joints = body.bone_positions(params)[:, :len(SMPLX_BODY_JOINTS)]   # (T,22,3) Z-up
ofbx   = read_object_fbx(_object_fbx_path(spec.motion_path))       # <seq>_o.fbx frère
poses  = object_pose_zup(ofbx.rot_native, ofbx.transl_native)[:T]  # (T,7) Z-up, helper partagé
mesh   = _write_proxy_obj(ofbx, cache_dir)                          # .obj local caché -> Path
return RawMotion(joint_pos=joints, joint_names=SMPLX_BODY_JOINTS, fps=ofbx.fps,
                 source_format="pahoi", object_poses_raw=(poses,),
                 object_mesh_paths=(Path(mesh),), smpl_params=params)
```
- `_object_fbx_path` : `<...>/<seq>/<seq>.npz` → `<...>/cap_res_fbx/<seq>_o.fbx` (remonter de 2, sujet-agnostique).
- `object_mesh_paths` attend un **fichier** → on écrit le proxy en `.obj` (repère local, m) dans
  `cache_dir/pahoi_meshes/` (idiome OMOMO/HODome). Si `spec.object_mesh_paths` fourni, override.
- Objet absent (`_o.fbx` manquant) ⇒ dégrade en corps-seul (`object_poses_raw=()`), comme HODome.

### 3. `paths.toml` + `paths.toml.example`
Ajouter :
```toml
[datasets.pahoi]
motion = ".../data/00_raw_datasets/PA-HOI_Dataset/Mocap_data"
```
`--motion-path` relatif = `cap_res_bvh_s1/1_001/1_001.npz`. `meta` inutile (proxy embarqué).

## Décision : source du mesh objet = **proxy 24-verts embarqué** (choix utilisateur)
Le proxy du `_o.fbx` est aligné pile avec sa propre trajectoire (zéro recalage). Les meshes HD
`Object_mesh/` (mappables par nom, mais repère local différent → recalage) sont **hors périmètre**
(follow-up possible si le contact fin l'exige).

## Tests (`HoloV2/tests/`, gate via `datapaths.PAHOI`)
- `test_fbx.py` (op pure, PAS de gate data — utilise un `_o.fbx` réel seulement si présent, sinon
  un mini-FBX synthétique construit dans le test) : formes `read_object_fbx`, `T` == nb frames npz,
  fps=30, mesh non vide, translation Y (verticale) croissante sur un « grab » connu.
- `test_load_pahoi.py` (gate `PAHOI`) : `RawMotion` — `joint_pos (T,22,3)`, `smpl_params` non nul,
  `len(object_poses_raw)==1`, `poses (T,7)` quats normés, mesh path existe. Parité d'alignement :
  la trajectoire objet reste « proche » du poignet actif pendant l'interaction (borne large).
- `datapaths.py` : ajouter `PAHOI = _opt("datasets","pahoi","motion")`.

## Risques / points à valider en implémentation
1. **Ordre d'Euler FBX** : XYZ par défaut supposé ; valider `scipy 'XYZ'` vs `'xyz'` par un test de
   plausibilité (mesh objet en contact avec la main pendant « grab/hold »). Trancher par les données.
2. **Triangulation** : `PolygonVertexIndex` FBX (indices négatifs = fin de polygone). Le proxy est
   une boîte (quads) → fan-triangulation simple suffit.
3. **1 key/frame** : vérifié sur échantillon ; garder l'interp `KeyTime` en fallback.
4. **Robustesse parser** : certains `Object_mesh/*.fbx` HD ont fait échouer le probe (type prop 'a')
   — hors périmètre (proxy uniquement) ; le parser ne vise QUE les `_o.fbx` (petits, réguliers).

## Hors périmètre (follow-ups notés)
- Meshes HD canoniques (recalage proxy→`Object_mesh/`).
- Attributs physiques (`Object_attributes.json` : size/weight/shape) + action/poids (`frames.txt`)
  → cibles « physics-aware » côté `targets/`.
- Ménage disque : supprimer le doublon `~/Downloads/PA-HOI_Dataset` (+ zip) après vérif d'intégrité.
