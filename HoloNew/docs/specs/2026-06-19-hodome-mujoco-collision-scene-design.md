# Spec 2 — HODome MuJoCo collision scene (object non-penetration + movable)

Date : 2026-06-19
Statut : design validé, prêt pour plan d'implémentation
Dépend de : Spec 1 (object-loading unification + HODome contact channel) — mergé.

## Contexte

Spec 1 a câblé le canal contact/SDF objet pour HODome (`object_sdf`, `_obj_poses_raw`,
`_obj_poses_mj`, probe). Il manque la **couche scène MuJoCo** : pour activer la
non-pénétration objet par collision (`activate_obj_non_penetration`) et la pose objet
*solved*/movable, le solveur charge un MJCF robot+objet `g1_<robot>_w_<object_name>.xml`
contenant le corps objet avec un free-joint et un geom de collision. Pour OMOMO ce fichier
est pré-généré et committé (`models/g1/g1_29dof_w_largebox.xml`). HODome n'en a pas : ses
objets viennent d'archives `scaned_object/<token>.tar`, arbitraires.

Constat (machinerie réutilisée, prouvé en explorant le code) :
- Le swap de scène se fait déjà dans `test_socp.py` (et `gmr_socp`, `holosoma`) quand
  `activate_obj_non_penetration and load_object_scene and object_name not in (None,"ground")`,
  via `ROBOT_URDF_FILE.replace(".urdf", f"_w_{object_name}.xml")`.
- La non-pénétration TEST-SOCP est **basée sur la collision MuJoCo**
  (`_update_jacobians_and_phis_from_q` calcule des distances signées robot↔geom objet),
  d'où le besoin du geom de collision objet dans la scène.
- `_obj_poses_mj` pilote le free-joint objet par frame ; la pose objet *solved* est
  réécrite dans `q[-7:]` (movable). Le free-joint fournit les 7 DOF.
- Le geom objet OMOMO (`largebox.xml`) est un unique `<geom type="mesh">` : MuJoCo en prend
  le **hull convexe**. C'est la seule façon historique — holosoma classique n'a jamais fait
  de décomposition convexe (seul `largebox`, convexe, était bundlé).

## Objectif

Générer à la volée le scene xml robot+objet pour HODome (`g1_<robot>_w_<token>.xml`), calqué
sur `g1_29dof_w_largebox.xml`, et le câbler via `SCENE_XML_FILE`, de sorte que la
non-pénétration MuJoCo et la pose objet movable s'activent pour HODome **sans modifier le
solveur**.

## Décisions de conception

- **Hull convexe** : geom unique `type="mesh"` (MuJoCo hull convexe), exactement comme
  `largebox.xml`. Aucune nouvelle dépendance (coacd/vhacd absents de l'env). Les concavités
  ne sont pas capturées — limite historique du système, jamais exercée car seul largebox
  servait. La décomposition convexe (coacd) est un raffinement **hors-scope**.
- **Inertie/masse = défaut fixe** comme `largebox.xml` (`mass=0.1`,
  `diaginertia="0.002 0.002 0.002"`). **Pas** de calcul d'inertie depuis le mesh : on
  n'ajoute rien qui n'était pas dans la version de base.
- **Génération à la volée + cache disque** (pas de pré-génération committée : objets HODome
  arbitraires). Clé = token, dans le répertoire cache HODome (tmp), comme les autres
  artefacts HODome prep.
- **Solveur inchangé** : toute la machinerie (swap de scène, `_obj_poses_mj`, non-pénétration,
  movable) est réutilisée telle quelle.

## Hors périmètre

- Décomposition convexe (coacd/vhacd) — dépendance absente, jamais dans le système de base.
- Inertie/masse calculée depuis le mesh.
- Multi-objet, multi-humain (cf. Spec 1).
- Tuning des paramètres de friction/solref/solimp au-delà des défauts largebox.

## Architecture & composants

### Composant A — Générateur de scene xml (`src/data_loaders/hodome_scene.py`, nouveau)

Fonction `build_hodome_scene_xml(robot_xml_path, token, mesh_obj_path, cache_dir=None) -> Path` :

1. Lit le MJCF robot de base (`robot_xml_path`, ex. `models/g1/g1_29dof.xml`).
2. Injecte, calqué sur `g1_29dof_w_largebox.xml` :
   - dans `<asset>` : `<mesh name="{token}_mesh" file="{abs mesh_obj_path}" scale="1 1 1"/>`
   - avant `</worldbody>` :
     ```xml
     <body name="{token}_link">
       <freejoint/>
       <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
       <geom type="mesh" mesh="{token}_mesh" pos="0 0 0" quat="1 0 0 0"
             friction="0.9 0.5 0.5" solref="0.02 1" solimp="0.9 0.95 0.001"
             rgba="0.7 0.8 0.9 0.7"/>
     </body>
     ```
3. Écrit `g1_<robot>_w_<token>.xml` dans `cache_dir` (défaut : répertoire cache HODome),
   idempotent (réutilise le cache). Le `mesh file=` est un chemin **absolu** (le .obj est
   dans le cache tar, pas à côté du scene xml) pour éviter tout souci de `meshdir`.

L'injection se fait par insertion ciblée de chaînes (mêmes hypothèses que
`create_new_scene_xml_file` : `<asset>` et `</worldbody>` présents dans le MJCF robot).

### Composant B — Câblage (`examples/robot_retarget.py`)

Dans le bloc façade object_interaction (Spec 1 : pose-file dataset → `OBJECT_NAME`=token) :
quand `cfg.dataset == "hodome"` et task object_interaction avec objet, résoudre le mesh via
l'`object_source` du loader (Spec 1), appeler `build_hodome_scene_xml(...)` et poser
`constants.SCENE_XML_FILE` = le chemin généré. Le swap existant (test_socp `__init__`,
gated `activate_obj_non_penetration and load_object_scene`) charge alors ce MJCF.

Note : le swap utilise `ROBOT_URDF_FILE.replace(".urdf", f"_w_{object_name}.xml")`. Le scene
xml généré DOIT donc être nommé/placé pour matcher ce chemin attendu, OU `SCENE_XML_FILE`
doit être consulté en priorité. Mirroir du chemin multi_boxes/climbing qui passe déjà par
`SCENE_XML_FILE`. Le plan d'implémentation tranchera le mécanisme exact (priorité
`SCENE_XML_FILE` quand non vide), en gardant OMOMO/largebox inchangé.

### Composant C — Mesh asset

Le geom référence le `.obj` extrait du tar (déjà caché par `extract_hodome_object_mesh`,
Spec 1). `build_hodome_scene_xml` reçoit ce chemin et l'inscrit en absolu dans l'asset.

## Flux de données (HODome, après Spec 2)

```
--dataset hodome --task-type object_interaction  (+ activate_obj_non_penetration)
  └─ facade : object_source non vide ⇒ object_interaction (Spec 1)
  └─ robot_retarget : OBJECT_NAME=token ; mesh via object_source
                      build_hodome_scene_xml(g1_29dof.xml, token, mesh) -> scene xml
                      SCENE_XML_FILE = scene xml
  └─ test_socp.__init__ : swap -> charge g1_..._w_<token>.xml ⇒ has_dynamic_object=True
  └─ solve : _obj_poses_mj pilote le free-joint ; non-pénétration MuJoCo active ;
             pose objet solved réécrite dans q[-7:] (movable)
```

## Stratégie de test

- **Génération** : `build_hodome_scene_xml` sur le MJCF g1 + un mesh box de test → fichier
  écrit ; contient `<freejoint/>`, `<mesh name="<token>_mesh"`, `<body name="<token>_link"`.
- **Parse MuJoCo** : `mujoco.MjModel.from_xml_path(scene)` charge sans erreur ; le modèle a
  un free-joint objet (nq robot + 7).
- **has_dynamic_object** : un retargeter HODome object_interaction avec
  `activate_obj_non_penetration=True` a `has_dynamic_object=True` et `_obj_poses_mj` de forme
  `(T,7)`.
- **Solve court** : `retarget(max_frames=3)` avec non-pénétration → `qpos` fini, `qpos[:,-7:]`
  porte la pose objet ; pas d'exception (cf. [[run-tests-low-max-frames]]).
- **Non-régression** : OMOMO largebox object_interaction inchangé (scene xml committé
  toujours utilisé ; `SCENE_XML_FILE` non posé pour OMOMO → chemin `.replace` historique).

## Risques & points d'attention

- **Nommage/priorité du scene xml** : le swap historique calcule le nom par `.replace`. Pour
  HODome il faut soit générer au chemin attendu, soit prioriser `SCENE_XML_FILE`. Garder
  OMOMO/largebox sur son chemin committé (ne pas régresser). Tranché au plan.
- **`meshdir`/chemins** : le MJCF robot de base a `<compiler meshdir=...>` pour ses propres
  meshes ; le mesh objet HODome est ailleurs → l'inscrire en **chemin absolu** dans l'asset
  pour ne pas dépendre de `meshdir`.
- **Échelle objet** : HODome `hodome_object_poses` est en Z-up à l'échelle réelle ; le mesh
  du tar est dans son frame natif cohérent avec ces poses (Spec 1) → `scale="1 1 1"`.
- **Hull convexe** : objets HODome concaves approximés par leur hull — attendu, documenté.
