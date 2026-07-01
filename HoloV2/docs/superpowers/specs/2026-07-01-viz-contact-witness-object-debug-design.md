# Spec — Viz debug solve : witness (cible/atteint) + contacts objets + clouds objets source/résolu

**Date** : 2026-07-01 · **Module** : `viz/` (couches, consommateur) · **Statut** : conçu

Extension du viz pour **débugger l'étage `solve`** côté contact/interaction. Aujourd'hui la couche
`contacts` montre le **contact cible vs atteint sur les points robot** (coloré par distance) — utile
pour la distance. Il manque : (1) le **witness** (point de surface le plus proche) cible ET atteint,
(2) la **même chose côté objets** (pas seulement le robot), et (3) — parce que **les objets sont des
variables de décision de l'optimiseur** — la possibilité de voir les **clouds objets source (observés)
ET résolus** (les deux états), et d'ancrer les contacts/witness objets à leur état respectif.

**Découverte clé (aucun plombage).** Toutes les données existent déjà dans `VizFrame` :
- Robot : `targets.robot_interaction.field` (cible) + `solved.contact_achieved.field` (atteint) —
  chacun un `MultiChannelField` portant `distance/direction/witness/active` (contracts.py:43-56).
- Objets : `targets.env_interaction.per_object[k]` (cible) + **`solved.contact_achieved.env[k].field`**
  (atteint) — car `ContactEval` porte déjà `.env: tuple[ContactEnvEval, ...]` (contracts.py:273), calculé
  par `ev.contacts(q, object_rot, object_pos)`. Le cloud objet résolu se calcule par transfo relative
  depuis le cloud source. Donc **feature = uniquement des couches viz** (aucune modif `BakeSource`/`SolvedFrame`).

## Décisions cadrées

> **CORRECTION (2026-07-01, post-implémentation, commit `ebef42f`).** La vue montre la **scène RÉSOLUE** :
> `contacts` (robot) et `object_contacts` affichent **cible ET atteint sur la géométrie RÉSOLUE** (cloud +
> pose objet résolus), seul le **CHAMP** diffère (cible = champ cible, atteint = champ atteint). Une ligne
> witness part toujours d'un point du nuage affiché (résolu). ⇒ dans les lignes ci-dessous et les composants,
> **toute mention « cible → cloud/pose SOURCE » est remplacée par « cible → cloud/pose RÉSOLU »**. Le nuage
> **source** de chaque objet reste comparable via le toggle de la couche `objects` (source orange vs résolu vert).

| Décision | Choix |
|---|---|
| Structure | **2 couches par-côté**, auto-contenues (distance + witness dedans) : `contacts` (robot, étendue) + `object_contacts` (objets, nouvelle). Pas de couches witness séparées. |
| Objets = variables | Fil conducteur **source vs résolu** partout. Couche `objects` gagne un toggle **cloud résolu** ; `object_contacts` ancre CIBLE→état source, ATTEINT→état résolu. |
| Mapping witness (« en ref objet ») | Le witness d'un canal objet est en **frame objet-local** → mappé monde via la pose objet : **cible = pose source** (`pose.object_rot/pos`), **atteint = pose résolue** (`solved.object_poses`). Canal sol = déjà monde. (Même nuance pour le witness robot sur les canaux objet.) |
| Données | **Zéro plombage** — tout lu depuis `VizFrame` (`targets.*`, `solved.contact_achieved.field/.env`, `pose.object_*`, `solved.object_poses`). |
| Cloud résolu | Calculé par transfo relative `T_résolu ∘ T_source⁻¹` appliquée au cloud source (objet rigide K=1) — pas besoin du rest-cloud, pas de champ `VizContext` nouveau. |
| Convention couleur | **source/cible = orange**, **résolu/atteint = vert** (cohérent avec la couche `contacts` existante), OU heatmap distance/active sur le canal sélectionné. Witness = lignes fines, mêmes teintes. |

## Composants

### 1. `layers/contacts.py` (robot) — ÉTENDUE
Existant (gardé) : 2 nuages sur `solved.robot_points_world`, colorés distance/active — cible
(`targets.robot_interaction.field`) vs atteint (`solved.contact_achieved.field`) sur le canal sélectionné.
Ajout : **lignes witness** pour les probes **actives** du canal sélectionné (subsample déterministe ~400) :
- cible : `robot_point[m] → witness_cible` (`targets.robot_interaction.field.witness`, mappé via pose **source** pour un canal objet) ;
- atteint : `robot_point[m] → witness_atteint` (`solved.contact_achieved.field.witness`, mappé via pose **résolue**).
Nouveaux sous-toggles : `witness cible`, `witness atteint`.

### 2. `layers/object_contacts.py` (objets) — NOUVELLE
Miroir de `contacts`, par objet `k`, **solve-gated** :
- **CIBLE** : `targets.env_interaction.per_object[k]` dessinée sur le cloud **source** (`object_clouds_world[k]`) ;
  witness `cloud_src[p] → witness` mappé via pose **source**.
- **ATTEINT** : `solved.contact_achieved.env[k].field` dessinée sur le cloud **résolu** (`object_cloud_solved(k)`) ;
  witness mappé via pose **résolue**.
Colorée distance/active sur le canal sélectionné ; sous-toggles distance/witness × cible/atteint.
Enregistrée dans `app.py` (toujours ajoutée, se masque si `solved is None`).

### 3. `layers/objects.py` (existante) — ajout cloud résolu
Sous-toggle **« cloud objet résolu »** : dessine `object_cloud_solved(k)` (posé à `solved.object_poses[k]`),
en plus du cloud source. Solve-gated. Montre de combien l'optimiseur a déplacé chaque objet.

## Helpers purs (numpy-only, testés)
- `object_cloud_solved(cloud_src, R_src, t_src, R_solved, t_solved) -> (P,3)` : applique
  `T_solved ∘ T_src⁻¹` au cloud source (objet rigide). Cas connu → positions connues.
- `witness_segments(probe_pts, witness_local, active, R_obj, t_obj) -> (S,2,3)` : pour les probes actives,
  segments `probe → (witness_local @ R_objᵀ + t_obj)` (canal objet) ou `probe → witness_local` (canal sol,
  `R_obj=I, t_obj=0`). Subsample déterministe (rng graine fixe) à un plafond (~400). Indices connus → segments connus.

## Flux de données (rappel : tout dans `VizFrame`)
```
CIBLE robot   : targets.robot_interaction.field           (witness via pose SOURCE si canal objet)
ATTEINT robot : solved.contact_achieved.field             (witness via pose RÉSOLUE)
CIBLE objet k : targets.env_interaction.per_object[k]      sur object_clouds_world[k]        (pose SOURCE)
ATTEINT objet k: solved.contact_achieved.env[k].field      sur object_cloud_solved(k)        (pose RÉSOLUE)
cloud résolu k: object_cloud_solved(object_clouds_world[k], pose.object_*[k], solved.object_poses[k])
```

## Tests
- **Helpers purs** : `object_cloud_solved` (transfo relative, cas connu) ; `witness_segments` (probe→witness,
  canal-local→monde, subsample) — known input → known output.
- **`update()`** (fakes server/gui duck-typés, sans viser réel) pour `contacts` + `object_contacts` :
  gardes données-manquantes (`solved is None` / `targets` None / canal inconnu → no-op + hide) ; **solve-gating**
  (atteint + cloud résolu masqués si `solved is None`) ; **happy-path `visible = self._cb.value`** ;
  **pattern toggle-en-pause** (`_last_frame`/`_last_ui` mémorisés + tous les contrôles câblés `.on_update` →
  ré-invoque `update` — le fix de la revue finale, obligatoire pour toute couche à plusieurs contrôles).
- **Couleurs** via `core.colors` (déjà testé). Subsample witness déterministe.
- Un smoke réel (dans le smoke solve existant `test_viz_app_solve_smoke`) : `object_contacts` + les witness
  s'exécutent sur de vrais handles viser sans exception.

## Invariants / garde-fous
- `viz` = **consommateur pur** : viser confiné aux couches / `core.viser_ops` ; helpers purs numpy-only.
- **Aucune modif** de `targets`/`solve`/`BakeSource`/`SolvedFrame`/`model` — la donnée est déjà là.
- Couches conformes au protocole `Layer` (`folder`/`setup`/`update`), gardes obligatoires, `visible=cb.value`,
  câblage toggle-en-pause. FRANÇAIS commentaires/docstrings, symboles anglais.
- Witness robot canal-sol déjà monde (comme la couche `fields`) ; canal objet mappé par la pose adéquate
  (source/résolue) — c'est la seule subtilité de correctness, testée par `witness_segments`.
