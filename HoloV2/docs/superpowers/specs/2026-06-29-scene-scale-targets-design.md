# Spec — Scaling de scène configurable (style + interaction), appliqué aux références

**Date** : 2026-06-29 · **Étage** : `targets` (`config` + `style/build` + interaction refs) · **Statut** : conçu

## Problème

Aujourd'hui les deux canaux de `targets` vivent à des **échelles différentes** :

- `style/build.py` produit des cibles **scalées** vers la morpho robot (ancre pelvis, `SCALE[corps]×ratio`,
  xy natif, z par `ratio`) — porté de GMR (`gmr_socp`).
- `interaction` évalue le nuage humain par **FK brute** (`pose_cloud(human_cloud, bone_rot, bone_pos)`,
  **non scalé**) contre les SDF objet/sol à l'échelle réelle, puis transporte sur le robot.

Le solveur reçoit donc des cibles **incohérentes** : *« mets la main ICI (réduit) »* (style) vs *« le
point de contact doit toucher l'objet LÀ (pleine échelle humaine) »* (interaction). Les deux termes se
combattent. On veut **une échelle de scène cohérente, configurable, partagée par les deux canaux**, sans
corrompre la détection des contacts.

Précédent : Holosoma natif (`../HoloNew/src/holosoma/preprocess.py`) scalait déjà la scène entière
(joints + poses objets) avec un facteur uniforme `robot_height/human_height`, **en amont** de
l'évaluation. Mais avec des ancres mixtes (xy vers l'origine ; objet z autour de sa hauteur frame-0
`z0`). On en reprend l'esprit, corrigé (z aussi ancré sol ; et appliqué côté refs, pas en amont).

## Décisions cadrées (issues du brainstorming)

| Décision | Choix retenu |
|---|---|
| But | Échelle de scène configurable, cohérente **style ↔ interaction** |
| Facteur | `SceneScaleConfig(scale_xy, scale_z: float \| None)`, `None → ratio = stature / human_height_assumption` |
| Ancre | **statique** : xy autour de l'**origine monde**, z autour du **sol** (z=0). Vraie similarité. **Pas** d'ancre pelvis-par-frame (couplerait la trajectoire objet au mouvement du bassin) |
| Ordre | **Évaluer sur la scène RÉELLE**, puis scaler en **étape finale** sur les refs |
| Portée | **refs uniquement** : positions style, trajectoire objet (`object_pos`), witness **sol**. **Aucune** modif des packages d'éval |
| Morphologique | le per-membre `0.9 / 0.8` reste un raffinement **style-only** (n'a pas de sens sur un objet) |
| Witness objet | stocké en frame **local** → suit la pose objet scalée, **rien à coder** |

## Périmètre

**Livré :**

1. `SceneScaleConfig` dans `targets/config.py`, composée dans `TargetsConfig` (à côté de `StyleConfig`).
2. Une fonction pure partagée `apply_scene_scale(points, scale_xy, scale_z, ratio)` (similarité ancrée
   origine/sol), seule source de la math d'échelle → pas de dérive entre les canaux.
3. `style/build.py` : le scaling d'axes du placement piloté par `SceneScaleConfig` (au lieu du hardcodé
   `sx=sy=1.0` / `sz=ratio`).
4. `interaction` (côté refs) : scale de la **trajectoire objet** (`object_pos` remis au solveur) + des
   **witness sol** (frame monde).
5. Tests (parité/numérique + cohérence cross-canal) + docs (`TARGETS.md`).

**Non-objectifs :**

- Le `solve` (comment il consomme des refs scalées avec une éval en échelle réelle) — décision côté
  `solve`, écrit par l'utilisateur.
- Le **redimensionnement des objets** : ils gardent leur taille réelle (le robot manipule la vraie
  boîte) → résidu de contact inhérent, documenté plus bas.
- Toute modif des packages d'éval (`evaluator.py`, `interaction/eval.py`, `style/eval.py`).

## Idée pivot — évaluer le RÉEL, scaler les refs en sortie

La scène réelle (SMPL + objets non scalés) est la **source de vérité pour *quelle* interaction a lieu**.
Scaler **avant** l'évaluation corrompt l'**assignation** des contacts :

> Exemple : poser une boîte sur une table. Si on scale la boîte vers le sol *avant* d'évaluer, elle
> passe **sous** la table ; le SDF trouve le **sol** comme surface la plus proche → on enregistre un
> contact boîte↔sol au lieu de boîte↔table. Faux.

Donc : **éval sur le réel** (assignation, flags `active`, witnesses corrects) **puis** scale en étape
finale sur les refs. Le scale ne sert qu'à *guider la trajectoire* du résultat à l'échelle robot, pas à
détecter les contacts. Les contacts gardent leur sens (boîte→table), la trajectoire scalée amène la
boîte au bon endroit.

## Le modèle de scale

`SceneScaleConfig` (frozen, dans `targets/config.py`) :

```
scale_xy: float | None = None   # None -> ratio ; un float = facteur fixe sur x,y
scale_z:  float | None = None   # None -> ratio ; un float = facteur fixe sur z
```

avec `ratio = stature / human_height_assumption` (déjà calculé par le style). Composée dans
`TargetsConfig(style=StyleConfig(), scene_scale=SceneScaleConfig())`.

Transformée (similarité diagonale, ancre xy=origine, z=sol=0) :

```
apply_scene_scale(p) = ( S_xy · p_x , S_xy · p_y , S_z · p_z )
   où S_a = ratio si cfg.scale_a is None, sinon cfg.scale_a
```

(le `ground_height` de `StyleConfig` reste l'ancre z ; ici 0).

## Application par canal (étape finale, après éval réelle)

| Canal | Quantité scalée | Frame | Note |
|---|---|---|---|
| **style** | positions des liens | monde | remplace le `sx=sy=1.0 / sz=ratio` hardcodé ; le morphologique `0.9/0.8` reste appliqué aux vecteurs pelvis-local |
| **interaction — objets** | `object_pos` (trajectoire/centre) | monde | porte les witness objet **locaux** correctement (objet à position scalée, taille réelle) |
| **interaction — sol** | `witness` du canal sol | monde | `witness.xy *= S_xy`, `witness.z` reste sur le plan ; `direction` inchangée (+z) ; `distance` (hauteur) `*= S_z` |
| **interaction — witness objet** | — | local | **rien** : stocké en local (`fields.py:50-52`), reconstruit `object_pos + R·witness_local` ⇒ suit la pose scalée |

Une **seule** fonction `apply_scene_scale` partagée par les deux canaux ⇒ ownership par-canal sans
dérive (un seul endroit pour la math). Pas de mutation de `FramePose` (la viz reste sur la scène
réelle).

## Ref vs éval — pourquoi ça ne touche pas l'évaluation

Le seam `targets → solve` a **deux sorties parallèles** (`evaluator.py:7-8`) :

> « Les références par frame (`list[FrameTargets]`) sont une sortie **PARALLÈLE** (`pipeline`), **pas une
> entrée de l'évaluateur**. »

- **Références** (`pipeline` + `style/build` + `refs.py`) — où le robot doit aller. **Seul endroit
  scalé.**
- **Évaluateur** (`Evaluator` → `style_eval` + `contact_eval`) — état courant @ `q` + jacobiennes,
  construit des assets **statiques** de l'`InteractionContext`, contre la scène **réelle**. Il ne lit
  jamais les refs.

⇒ Ajouter le scale côté refs laisse `evaluator.py`, `interaction/eval.py`, `style/eval.py` **intacts**.
Conséquence assumée : refs = scalé, éval = réel ; le `solve` (futur) devra le savoir — affaire de
conception côté `solve`, **pas** une modif d'éval.

## Limitation inhérente (documentée, non corrigée)

Le style scale le corps autour de l'origine/sol (la main → `S·(t + R·l)`), l'interaction met le contact
sur la vraie surface objet (`S·t + R·l`, objet gardé à taille réelle). Écart au contact ≈
`(1 − S)·R·l` = `(1 − scale) × (offset du point de contact dans l'objet)`, soit **quelques cm**. Le
solveur fait un compromis. C'est inhérent à « scaler le corps mais pas la taille de l'objet » ; le seul
« fix » serait de redimensionner l'objet (refusé : le robot manipule la vraie boîte).

## Réconciliation morphologique × échelle (résolue)

Le morphologique (`0.9 / 0.8`, **y compris le `0.9` du pelvis sur z** : `scaled_root_z =
scale_torso_legs · ratio · root_z`, le pelvis étant dans le groupe torse/jambes) **reste intégralement
dans `style/build.py`**, inchangé. La `SceneScaleConfig` ne remplace QUE le **facteur de placement** —
le `ratio` et le `1.0` xy codés en dur du root — et pilote l'interaction. Pas de factorisation qui
« sort » le morphologique du style ; pas de bassin qui remonte.

Conséquences :

- **défaut `None, None`** → `ratio` partout : le **xy du root devient scalé par `ratio`** (le `(b′)`
  voulu) ; le reste (z bassin `0.9·ratio`, membres `morph·ratio`) inchangé.
- **`SceneScaleConfig(scale_xy=1.0, scale_z=None)`** → **comportement natif actuel à l'identique**.

Détail laissé au plan (sans impact sur ce qui précède) : si on expose des valeurs **anisotropes
explicites** (`scale_xy ≠ scale_z` ≠ `ratio`), faut-il que les vecteurs membres pelvis-local suivent
l'anisotropie ou restent isotropes `morph·ratio` ? Choix par défaut **isotrope `morph·ratio`**
(préserve le natif) ; l'anisotropie ne joue alors que sur le **placement du root** + l'interaction. Cas
d'usage réels (uniforme `ratio`, ou `xy=1.0 / z=ratio`) couverts dans les deux cas.

**Critère d'acceptation (test de parité)** : `SceneScaleConfig(scale_xy=1.0, scale_z=None)` reproduit
**exactement** la sortie style actuelle (et donc la parité V1 `test_style_matches_v1_scale_offset`) ;
un test séparé vérifie le nouveau défaut (`None → ratio` scale aussi le xy du root).

## Fichiers touchés

| Fichier | Changement |
|---|---|
| `src/targets/config.py` | `SceneScaleConfig` + champ `scene_scale` dans `TargetsConfig` + helper `apply_scene_scale` (ou module dédié) |
| `src/targets/style/build.py` | placement piloté par `SceneScaleConfig` (remplace le hardcodé) ; morphologique conservé |
| `src/targets/interaction/refs.py` (et/ou `pipeline.py`) | scale `object_pos` (ref) + witness sol, en étape finale |
| `src/targets/pipeline.py` | passe `cfg.scene_scale` aux deux canaux |
| `tests/test_style.py` | maj parité/numérique selon la décision root ; cas `scale_xy`/`scale_z` explicites |
| nouveau test | cohérence cross-canal : style et interaction scalent de façon identique (même `apply_scene_scale`) |
| `docs/TARGETS.md` | documenter l'échelle de scène + l'ordre éval-réel→scale-refs |

## Tests

- **Unitaire** `apply_scene_scale` : ancre origine/sol, `None→ratio`, anisotropie xy≠z, point sur le sol invariant en z.
- **Équivalence native** (le critère clé) : `SceneScaleConfig(scale_xy=1.0, scale_z=None)` reproduit **exactement** la sortie style actuelle ⇒ la parité V1 (`test_style_matches_v1_scale_offset`) reste verte épinglée à cette config.
- **Nouveau défaut** : `None, None` scale le xy du root par `ratio` (test distinct du natif).
- **Style** : numérique (placement + morphologique).
- **Interaction** : witness sol scalé (xy par `S_xy`, hauteur par `S_z`) ; trajectoire objet scalée ; witness objet local inchangé (suit la pose).
- **Cross-canal** : pour un même `SceneScaleConfig`, style et interaction appliquent la même similarité (anti-dérive).
- **Non-régression éval** : `evaluator`/`eval.py`/`style_eval` inchangés (les tests d'éval existants passent tels quels).
