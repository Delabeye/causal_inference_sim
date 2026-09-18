# Contrôle relationnel de la formation

Ce document décrit le fonctionnement actuel du swarm, la modification du
contrôleur de formation et les vérités terrain produites pour les modèles de
relational inference.

## 1. Objectif de la modification

Dans l'ancien mode `direct_offset`, la position demandée au follower était
calculée directement par :

\[
p_i^{target}=p_j+R(\psi_j)d_{ij}^{\star}.
\]

Le contrôleur recevait donc la géométrie finale de la formation comme une cible
absolue. Pour un modèle NRI/dNRI/SRI, cela crée un raccourci prédictif : le
follower peut conserver sa place parce que le contrôleur lui fournit déjà la
bonne position, même si le modèle n'identifie pas correctement l'interaction
entre les drones.

Le mode `relational` retire ce raccourci. L'offset
`d*` reste nécessaire pour définir la géométrie, mais il n'est plus ajouté à la
position cible. Il devient la position d'équilibre d'une interaction élastique
dirigée.

## 2. Convention du graphe

Une arête `j -> i` signifie que le drone receveur `i` utilise l'état communiqué
par le drone émetteur `j` pour produire sa commande.

Dans les matrices sauvegardées :

- la ligne est le receveur `i` ;
- la colonne est l'émetteur `j` ;
- `A[i,j] = 1` représente donc `j -> i`.

Les formations `triangle`, `trail`, `v` et `echelon` utilisent une étoile
dirigée : le leader est la référence de tous les followers. La formation
`line` utilise une chaîne dirigée : chaque follower dépend de son prédécesseur.

## 3. État utilisé par un follower

Pour chaque follower `i`, le swarm utilise :

- sa position et sa vitesse courantes `p_i, v_i` ;
- la dernière position et vitesse reçues de sa référence `p_j, v_j` ;
- le yaw communiqué de la référence `psi_j` ;
- l'offset local de l'arête `d*_{ij}`.

L'offset est exprimé dans le repère de la formation puis transformé dans le
repère monde :

\[
d_{ij}^{world}=R(\psi_j)d_{ij}^{\star}.
\]

Pour une ligne, cet offset est local à chaque arête. Par exemple, si tous les
drones doivent être séparés de 1 m, chaque arête de la chaîne possède
`[-1, 0, 0]`, au lieu de donner au dernier drone un offset absolu de `[-3,0,0]`
par rapport à son prédécesseur.

## 4. Interaction attractive et amortissement

La position relative observée et son erreur sont :

\[
r_{ij}=p_i-p_j,
\qquad
e^p_{ij}=d_{ij}^{world}-r_{ij}.
\]

L'attraction est une correction de vitesse proportionnelle à cette erreur :

\[
v_{ij}^{attr}=k_p e^p_{ij}.
\]

L'amortissement aligne les vitesses et réduit les oscillations :

\[
e^v_{ij}=v_j-v_i,
\qquad
v_{ij}^{damp}=k_v e^v_{ij}.
\]

La correction relationnelle est saturée sans modifier sa direction :

\[
c_{ij}=v_{ij}^{attr}+v_{ij}^{damp},
\]

\[
\bar c_{ij}
=
\min\left(1,\frac{v_{rel}^{max}}{\|c_{ij}\|}\right)c_{ij}.
\]

La vitesse commandée au follower est finalement :

\[
\boxed{v_i^{cmd}=v_j+\bar c_{ij}}.
\]

Le premier terme transporte le follower avec sa référence. Les deux autres ne
servent qu'à corriger la configuration relative. La cible de position envoyée
au PID est un court point de visée construit depuis la position propre du
follower :

\[
\boxed{p_i^{target}=p_i+T_{lookahead}v_i^{cmd}}.
\]

Il n'existe donc plus de terme `p_j + offset` dans la cible du mode relationnel.

## 5. Répulsion

La répulsion UAV-UAV existante reste calculée localement dans `entities/uav.py`
à partir des états reçus. Pour une distance inférieure au rayon de sécurité,
elle pousse le receveur dans la direction opposée à l'émetteur. Chaque
contribution est conservée séparément par émetteur, puis les contributions UAV
et obstacle sont saturées ensemble.

Cette répulsion est ajoutée dans la finalisation de la vitesse avant le PID. Il
n'y a pas de deuxième répulsion centrale dans le mode relationnel. Le mode
historique `direct_offset` conserve son ancienne correction centrale pour
permettre les comparaisons et la reproduction d'anciens datasets.

La dynamique conceptuelle devient donc :

\[
v_i^{final}
=v_j+v_{ij}^{attr}+v_{ij}^{damp}
+\Delta v_i^{repulsion},
\]

avant saturation par les limites de vitesse et conversion en commande moteur
par le PID.

## 6. Démarrage de la formation

1. Tous les drones effectuent leur montée verticale.
2. Le leader reste immobile tant que les followers ne sont pas prêts.
3. L'application de la cible relationnelle est interpolée pendant la durée de
   transition configurée.
4. En mode relationnel, le leader n'est libéré que lorsque chaque erreur de
   formation est inférieure à `ready_tolerance`.
5. Une fois le leader libéré, sa vitesse est transmise dans toute la topologie.

Cette phase de convergence est utile pour l'apprentissage : les forces
attractives sont non nulles et rendent les arêtes observables.

## 7. Configuration YAML

La configuration se trouve sous `swarm[].formation` :

```yaml
formation:
  control_mode: relational
  attraction_gain: 0.8
  velocity_alignment_gain: 0.35
  relational_lookahead_s: 0.25
  max_relational_correction_speed: 1.5
  ready_tolerance: 0.25
```

- `control_mode`: `relational` ou ancien `direct_offset` ;
- `attraction_gain`: conversion de l'erreur de position en correction de vitesse ;
- `velocity_alignment_gain`: amortissement de l'erreur de vitesse ;
- `relational_lookahead_s`: horizon du point de visée donné au PID ;
- `max_relational_correction_speed`: saturation de l'attraction + amortissement ;
- `ready_tolerance`: erreur maximale permettant au leader de commencer sa mission.

Les valeurs réellement utilisées sont copiées dans
`run_<id>_formation_<swarm>.json`. Cela permet de reconstruire exactement la loi
de commande associée à chaque run.

## 8. Vérité terrain et fichiers produits

### CSV relationnel

`run_<id>_interactions.csv` contient une ligne par paire dirigée et par mise à
jour du contrôle. Les nouveaux champs incluent :

- `relative_position_error_*` ;
- `relative_velocity_error_*` ;
- `formation_attraction_raw_*` ;
- `formation_attraction_*` après gain et saturation ;
- `formation_damping_*` ;
- `formation_transport_v*` ;
- `formation_correction_scale` pour la saturation relationnelle ;
- `formation_application_scale` pour la transition progressive ;
- `formation_control_mode`.

### Matrices relationnelles

`run_<id>_ground_truth_matrices.npz` contient notamment :

Le schéma de ces matrices est maintenant en version 2.

- `structural` : arêtes de référence configurées ;
- `active` : mécanisme effectivement utilisé au pas courant ;
- `attraction_norm` : intensité attractive par arête ;
- `damping_norm` : intensité d'amortissement par arête ;
- `repulsion_norm` : intensité répulsive UAV-UAV par arête ;
- `relative_error_norm` : écart à l'équilibre par arête ;
- `target_delta_norm` et `rpm_delta_norm` : effets des ablations de contrôle.

### Trace d'apprentissage

`run_<id>_learning_trace.npz`, schéma version 2, ajoute les tenseurs agrégés par
nœud :

- `formation_error` ;
- `formation_attraction` ;
- `formation_damping` ;
- `formation_scales` (saturation et transition) ;
- `formation_control_modes`.

Les contributions complètes par paire restent dans le CSV et les matrices de
vérité terrain.

## 9. Interprétation existence/intensité

Le graphe structurel et l'intensité instantanée ne doivent pas être confondus :

\[
A_{ij}=1
\]

signifie que `j -> i` est une dépendance configurée, tandis que

\[
W_{ij}^{attr}(t)=A_{ij}\|v_{ij}^{attr}(t)\|
\]

mesure l'action instantanée. À l'équilibre, une arête peut exister avec une
attraction presque nulle. L'alignement de vitesse ou une perturbation future
peuvent néanmoins la réactiver.

Pour évaluer un modèle :

- comparer l'existence à `structural` ;
- comparer l'activité à `active` ;
- comparer une intensité apprise aux canaux attraction, damping et répulsion ;
- ne pas transformer automatiquement une intensité nulle en absence d'arête.

## 10. Recommandations pour le prochain dataset

- Conserver des phases de convergence, des virages et des variations de vitesse.
- Randomiser légèrement les positions initiales autour de la formation.
- Garder des runs sans intervention pour mesurer la prédiction nominale.
- Répartir les interventions entre leaders, followers et instants différents.
- Conserver les mêmes états initiaux pour les deux branches d'un fork.
- Ne pas fournir `desired_offset` comme entrée au modèle si le but est de
  découvrir les relations à partir des trajectoires.
- Réaliser les splits train/validation/test au niveau des runs ou des seeds,
  jamais au niveau de fenêtres issues d'un même run.

Une formation parfaitement immobile à l'équilibre ne contient presque aucune
excitation et reste difficilement identifiable. Les accélérations du leader,
le vent, les erreurs initiales et les perturbations contrôlées sont donc des
éléments informatifs du dataset, pas seulement du bruit.

### Fréquence des nouvelles traces

La configuration actuelle utilise un pas physique exact de `1/240 s` et un
contrôle à 80 Hz. Un dataset nouvellement généré contient donc normalement 80
lignes de learning trace par seconde. Pour obtenir une grille d'apprentissage
à 10 Hz, utiliser `downsample: 8` dans la configuration du modèle.

L'ancien dataset `all_formations_forks_60s` avait été produit avec une ancienne
horloge et contient environ 60 lignes/s ; il requiert encore `downsample: 6`.
Il ne faut pas réutiliser automatiquement ce stride pour le nouveau dataset.

## 11. Emplacement de l'implémentation

- `swarm/swarm.py` : topologie, attraction, amortissement et construction de la cible ;
- `entities/uav.py` : application graduelle, répulsion locale, PID et ablations ;
- `simulator/simulator_manager.py` : lecture du YAML et metadata par run ;
- `simulator/relational_ground_truth_logger.py` : CSV et matrices par arête ;
- `simulator/learning_trace_logger.py` : tenseurs synchronisés par nœud.
