# Spec 1 — Unification du chargement objet + canal contact HODome

Date : 2026-06-19
Statut : design validé, prêt pour plan d'implémentation

## Contexte

L'audit du chargement SMPL/objet a relevé plusieurs incohérences (numérotées #1–#5).
Cette spec couvre **#1, #2, #3 et la couche contact/SDF de #5**. La couche scène
MuJoCo de #5 (génération d'URDF de collision pour les meshes scannés HODome,
non-pénétration par collision, objet *movable* solved) est reportée en **Spec 2**,
qui dépend de celle-ci.

Aujourd'hui l'alimentation du canal objet du solveur est codée en dur pour OMOMO :

- mesh objet via `constants.OBJECT_MESH_FILE = models/{obj}/{obj}.obj` (objets bundlés),
- poses objet via `load_intermimic_data(<task>.pt)`.

HODome n'a ni `.pt` ni mesh bundlé (le mesh vient d'une archive `scaned_object/<token>.tar`,
les poses de `object_R/object_T`). De plus `facade.normalize_dataset_cfg` **force
`task_type=robot_only`** pour tout dataset smplx, donc l'objet HODome n'est qu'un overlay
viewer, jamais branché dans le solveur.

En parallèle, trois échantillonneurs de surface objet coexistent avec des conventions
différentes (grille SDF, `sample_object_surface` à densité, `load_object_data` à count
fixe), du code mort traîne (`contact/object_input.py`, `probes.make_object_grid`), et un
fallback silencieux masque l'absence de `obj_scale`.

## Objectifs

1. **#5a** Rendre l'alimentation objet du solveur **agnostique au dataset** via une
   abstraction `ObjectSource`, et activer le canal contact/SDF objet pour HODome
   (probe `SmplxGroundProbe` + termes D/X/P).
2. **#1** N'avoir **qu'un seul** échantillonneur de surface objet *even* (basé densité),
   réutilisé par le solveur (movable), le viewer et le chemin object_interaction.
3. **#2** Supprimer le code mort objet.
4. **#3** Remplacer le fallback silencieux `obj_scale` par une erreur explicite.

## Hors périmètre (non-objectifs)

- **Spec 2** : génération d'URDF/scène de collision MuJoCo HODome, non-pénétration par
  collision (`activate_obj_non_penetration` + `load_object_scene`), pose objet *movable*
  solved. La présente spec n'active QUE le canal contact/SDF (termes D/X/P + probe), qui
  ne requiert aucun corps MuJoCo objet.
- **Multi-humain** : le solveur est mono-robot (un human→G1, une correspondance OT, une
  scène Pinocchio). C'est un epic séparé, hors de la couche loading.
- **Multi-objet réel** : non requis par OMOMO ni HODome (mono-objet par séquence). Le
  contrat est néanmoins *façonné* pour l'accepter sans réécriture future (voir Décisions).

## Décisions de conception

- **Contrat `object_source()` liste-ready, consommation mono.** La méthode retourne
  `list[ObjectSource]` ; le builder consomme **un seul** objet avec `assert len(sources) <= 1`.
  Le multi-objet devient une extension purement additive (ni OMOMO ni HODome n'en ont besoin
  aujourd'hui). Liste vide = pas d'objet pour cette séquence.
- **`object_source()` est une méthode abstraite obligatoire** : chaque `MotionLoader`
  l'implémente explicitement (pas de défaut hérité), pour forcer chaque dataset à déclarer
  son comportement objet. Les datasets sans objet (lafan/sfu/climbing) retournent `[]`.
- **Sampling unifié sur la densité** (pts/m²), pas un count fixe.
- **Repli legacy préservé** : quand `cfg.dataset` n'est pas fourni (invocation OMOMO legacy
  par chemins explicites), le builder garde l'ancien chemin (`OBJECT_MESH_FILE` + `.pt`),
  pour ne casser aucun comportement existant.

## Architecture & composants

### Composant A — Abstraction `ObjectSource` (`src/data_loaders/base.py`)

```python
@dataclass(frozen=True)
class ObjectSource:
    mesh_path: Path          # mesh objet à charger (déjà recentré + scalé à la taille réelle)
    poses_raw: np.ndarray    # (T, 7) [qw, qx, qy, qz, x, y, z], MÊME monde Z-up que human_joints
```

`MotionLoader` gagne une méthode abstraite :

```python
@abstractmethod
def object_source(self, *, motion_path, obj_path, model_path,
                  task_type, constants, motion_data_config,
                  smpl_model_dir=None) -> list[ObjectSource]:
    """Sources objet de la séquence dans le monde Z-up des human_joints.
    Liste vide quand la séquence n'a pas d'objet (ou task_type robot_only)."""
```

Implémentations :

- **`OmomoMixedLoader`** : si `obj_path is None` ou `robot_only` → `[]`. Sinon mesh =
  `models/{obj}/{obj}.obj` bundlé si présent, sinon `captured_objects/{obj}.obj`
  recentré sur son centroïde × `obj_scale` (du `.p`, cf. #3) ; poses =
  `load_intermimic_data(motion_path)[1]`.
- **`HoDomeLoader`** : si `obj_path is None` ou `robot_only` → `[]`. Sinon mesh =
  `extract_hodome_object_mesh(token, scaned_object/)` ; poses = `hodome_object_poses(obj_path)`.
  Le token = 2e segment du stem (`sub3_box_001 → box`), `scaned_object/` voisin de `obj_path`.
- **`LegacyLoader`** (lafan/sfu/climbing) : `[]` (climbing garde son câblage objet existant,
  inchangé par cette spec).

Le mesh fourni par `object_source` est **déjà à la taille réelle et recentré** (le viewer et
le solveur n'ont plus à dupliquer centrage/scale).

### Composant B — Builder agnostique (`src/test_socp/builder.py`)

Un résolveur unique remplace le hardcode `OBJECT_MESH_FILE` + `load_intermimic_data` :

- Si `cfg.dataset` est set : `sources = resolve_loader(cfg.dataset).object_source(...)`,
  `assert len(sources) <= 1`, puis `mesh_file, poses_raw = sources[0]` si non vide.
- Sinon (repli legacy) : `mesh_file = constants.OBJECT_MESH_FILE`,
  `poses_raw = load_intermimic_data(pt_path)[1]`.

Le `mesh_file` et `poses_raw` ainsi obtenus alimentent **les trois consommateurs existants**
sans changer leur logique :

1. `object_sdf = load_or_build_object_sdf(mesh_file, ...)` (gardé tel quel),
2. `rt._obj_poses_raw` (D/X/P + probe),
3. `rt._obj_poses_mj` (drive MuJoCo — uniquement si `activate_obj_non_penetration and
   load_object_scene`, donc inactif pour HODome en Spec 1 ; voir Spec 2).

`SmplxGroundProbe` est déjà toujours construite ; elle gagne un `object_sdf` non-nul +
`obj_poses_raw` pour HODome, donc les termes D/X/P s'activent sans autre changement.

### Composant C — Lever le `robot_only` forcé (`facade.py` + `examples/robot_retarget.py`)

- `facade.normalize_dataset_cfg` : ne plus forcer `robot_only` pour smplx **quand le loader
  retourne une source objet non vide et `obj_path` existe**. Concrètement : hodome+objet
  conserve `object_interaction` ; sfu (pas d'objet) reste sans objet (naturellement `[]`).
- `validate_config` (robot_retarget.py:170) : relâcher la contrainte « object_interaction
  requires smplh » pour autoriser smplx+objet.
- `OBJECT_NAME` renseigné pour hodome (token de la séquence) afin que le gate builder
  `object_name not in (None, "ground")` passe. `OBJECT_MESH_FILE` devient non-pertinent pour
  le SDF (mesh via `object_source`), mais reste posé pour le repli legacy.

Le chemin de chargement humain smplx du builder (gated `data_format == "smplx"`, lit le npz)
est **indépendant** du `task_type` : il fonctionne identiquement en `object_interaction`.

### Composant D — Unification du sampling even (#1)

- `movable.sample_object_surface(mesh_file, density, seed)` (even, densité pts/m², min 64,
  frame objet-local) devient **le** sampler de surface objet even.
- Consommateurs basculés dessus :
  - viewer points objet (`view_stages.py:249` et `:291`),
  - `robot_retarget` object_interaction (`:349`) — le scale `smpl_scale` pour la variante
    `*_demo` est appliqué côté appelant (multiplication), plus dans le sampler.
- `load_object_data` (`src/utils.py`) : **conservé uniquement** pour le chemin climbing
  pondéré (`surface_weights` / `use_face_normals`, robot_retarget.py:367) et l'option bbox.
  Son usage even non-pondéré est retiré.
- Commentaire `view_stages.py:247` corrigé (devient vrai : même sampler que le solveur movable).

### Composant E — Nettoyages (#2, #3)

- **#2** Supprimer `src/test_socp/contact/object_input.py` et `probes.make_object_grid`
  (+ ses constantes `OBJECT_GRID_DENSITY` si orphelines) après un grep final confirmant
  l'absence d'import vivant.
- **#3** `view_stages` (chemin `captured_objects` non centré) : `obj_scale` introuvable →
  **lever `ValueError` explicite** (« obj_scale manquant pour <seq>, mesh non centré : taille
  incorrecte ») au lieu du warning + fallback unité.

## Flux de données (HODome, après Spec 1)

```
--dataset hodome --motion-name <seq>  (--task-type object_interaction)
  └─ facade.resolve_paths_by_name → motion(.npz), obj(.npz), model(smplx)
  └─ facade.normalize_dataset_cfg → object_source non vide ⇒ garde object_interaction
  └─ HoDomeLoader.load            → human_joints (Z-up), smpl_scale
  └─ HoDomeLoader.object_source   → [ObjectSource(tar-mesh, hodome_object_poses Z-up)]
  └─ builder                      → object_sdf (mesh), _obj_poses_raw (poses)
                                   → SmplxGroundProbe interroge le SDF ⇒ D/X/P actifs
```

## Stratégie de test

- `object_source()` par dataset : omomo (mesh + poses cohérents), hodome (tar extrait,
  poses Z-up, longueur = T), legacy/sfu (`[]`). Étend `tests/test_data_loaders_*`.
- **Parité legacy** : sur une séquence OMOMO sans `--dataset`, le builder produit le même
  `object_sdf.dims` et les mêmes `_obj_poses_raw` qu'avant le refactor.
- **HODome object_interaction** : le builder construit un `object_sdf` non-nul et
  `_obj_poses_raw` de forme `(T,7)` ; au moins une frame a un champ contact actif.
- **Sampler densité** : déterministe à seed fixe ; densité×aire respectée (≥ min 64).
- **#3** : séquence sans `obj_scale` sur un mesh `captured_objects` → `ValueError`.
- Non-régression : la suite existante (`test_object_interaction`, `test_movable`,
  `test_contact_*`, `test_hodome_*`) passe.

## Risques & points d'attention

- `object_source` doit garantir le **même monde Z-up** que `human_joints` pour HODome
  (poses déjà passées par `hodome_object_poses` → Z-up : OK).
- Repli legacy : bien tester que l'absence de `--dataset` n'emprunte jamais le chemin loader.
- HODome reste **sans** non-pénétration MuJoCo en Spec 1 : sans contrainte de collision, les
  termes D/X/P seuls peuvent laisser une légère interpénétration. C'est attendu et adressé
  par Spec 2.
