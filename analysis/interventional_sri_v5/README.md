# Interventional SRI v5

Cette version combine la pipeline complète de SRI avec les forks physiques du
simulateur. Elle est séparée des versions précédentes afin de conserver la v4
binaire comme expérience d'identifiabilité minimale.

## Commandes

Entraînement complet :

```bash
python -m analysis.interventional_sri_v5.train \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_v5
```

Smoke test :

```bash
python -m analysis.interventional_sri_v5.train \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_v5_smoke \
  --epochs 1 \
  --device cpu
```

Réévaluation :

```bash
python -m analysis.interventional_sri_v5.evaluate \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_v5
```

## 1. Entrées

```text
history_states       [B,N,L,6]  positions et vitesses avant le fork
history_interventions[B,N,L,4]  interventions passées, nulles par défaut
intervention_input   [B,N,H,4]  force courante et indicateur active
baseline_future      [B,N,H,6]  branche physique sans intervention
intervention_future  [B,N,H,6]  branche physique avec intervention
paired               [B]        indique l'existence du fork physique
```

L'intervention courante n'entre jamais dans l'encodeur. Le posterior décrit
donc la structure présente juste avant l'action et ne peut pas lire la réponse
qu'il doit expliquer.

## 2. GNN et attention de strength

À chaque instant historique, un MLP partagé calcule une représentation de
chaque nœud. Pour toute paire dirigée `sender -> receiver`, un deuxième MLP
construit :

\[
h_{ji,\mathrm{gnn}}^t
=\operatorname{MLP}_{edge}([h_j^t,h_i^t]).
\]

Les relations ayant le même receiver sont comparées par attention multi-tête :

\[
h_{ji,\mathrm{str}}^t
=\operatorname{MHA}(h_{ji,\mathrm{gnn}}^t).
\]

La strength est bornée dans `[strength_floor,1]`. Ce plancher empêche
l'intensité de fermer silencieusement toutes les relations pendant que la
probabilité d'existence reste arbitraire.

## 3. Prior forward et posterior backward de SRI

La v5 infère deux variables catégorielles dynamiques :

\[
z_{G,ji}^t\in\{0,1,2\},\qquad
z_{A,i}^t\in\{0,1,2\}.
\]

- `z_G=0` signifie aucune interaction directe ;
- `z_G=1,2` sont deux mécanismes relationnels non supervisés ;
- `z_A` représente un mode de dynamique locale du nœud.

Deux LSTM forward apprennent les priors causaux :

\[
p_G^t=p(z_G^t\mid X^{1:t},z_G^{1:t-1}),\qquad
p_A^t=p(z_A^t\mid X^{1:t},z_A^{1:t-1}).
\]

Deux LSTM reverse utilisent le reste de l'historique pendant l'entraînement et
produisent :

\[
q_G^t=q(z_G^t\mid X^{1:L}),\qquad
q_A^t=q(z_A^t\mid X^{1:L}).
\]

Les quatre distributions sont catégorielles. `z_G` et `z_A` sont échantillonnés
par Gumbel-Softmax straight-through. Pour les instants futurs, seule la partie
forward est utilisée.

L'existence binaire évaluée contre le simulateur marginalise les types actifs :

\[
P(edge)=1-P(z_G=0)=P(z_G=1)+P(z_G=2).
\]

Les deux types actifs ne doivent pas être interprétés comme des labels connus
avant une analyse de leurs MLP et de leurs activations. Le simulateur fournit
une vérité binaire de chaîne, pas des labels répulsion/alignement comme le
dataset original de SRI.

## 4. Décodeur SRI typé

Chaque type actif possède son propre MLP :

\[
m_{j\leftarrow i}^t
=s_{ji}^t\sum_{k=1}^{2}z_{G,ji,k}^t
g_k(h_i^t,h_j^t).
\]

Le type zéro ne possède aucun MLP et produit exactement un message nul. Les
messages entrants sont agrégés par receiver. Un GRU local reçoit ensuite :

```text
état propre du nœud
+ messages graphe
+ embedding de z_A
+ intervention externe u
+ contexte éventuel
```

Le chemin local ne voit jamais directement les états des autres drones. Toute
propagation d'une intervention vers un nœud non ciblé doit traverser les
messages du graphe.

## 5. Reconstruction historique et rollout futur

Contrairement aux versions qui entraînent uniquement le dernier posterior, la
v5 reconstruit chaque transition de l'historique en teacher forcing :

\[
\hat X^{t+1}=D(X^t,z_G^t,z_A^t).
\]

Le futur est ensuite prédit autorégressivement. Après chaque pas, le prior
forward recalcule `z_G` et `z_A` uniquement à partir des états déjà prédits.

## 6. Double rollout interventionnel

Au fork, les branches partagent exactement :

- l'état observé initial ;
- le posterior `q_G^L` ;
- le posterior `q_A^L` ;
- la mémoire du décodeur ;
- les paramètres ;
- les mêmes bruits Gumbel futurs.

Elles diffèrent seulement par `u` :

\[
\hat X^0=D(X_L,z_G^L,z_A^L,U=0),
\qquad
\hat X^I=D(X_L,z_G^L,z_A^L,U=I).
\]

L'effet causal prédit est :

\[
\widehat{\Delta X}=\hat X^I-\hat X^0.
\]

Les graphes futurs peuvent diverger après le premier pas, mais seulement parce
que l'intervention a déjà modifié un état prédit. Il n'y a donc pas de fuite du
futur dans le posterior initial.

## 7. Loss

La partie SRI est un ELBO dynamique :

\[
\mathcal L_{SRI}
=\lambda_h\mathcal L_{history}
+\mathcal L_{baseline}
+\beta_G KL(q_G\Vert p_G)
+\beta_A KL(q_A\Vert p_A).
\]

Notre extension ajoute :

\[
\mathcal L_I,
\quad
\mathcal L_{\Delta},
\quad
\mathcal L_{non-target},
\quad
\mathcal L_{necessity},
\quad
\mathcal L_{contrast}.
\]

`necessity` exige que le graphe complet batte le graphe nul pour les nœuds non
ciblés réellement affectés. `contrast` exige qu'il batte une permutation du
même graphe. Les batches contiennent 50 % de forks et 50 % de fenêtres
nominales. Le KL est réchauffé progressivement et la température Gumbel décroît
de 1.0 à 0.3.

## 8. Optimisations d'entraînement sans approximation

Les deux branches d'un même mode (`full`, `zero` ou `shuffled`) sont
concaténées sur l'axe batch et déroulées dans une seule boucle récurrente. Les
états initiaux et les bruits Gumbel restent partagés, puis les résultats sont
séparés en branches baseline et intervention. Cette vectorisation ne change ni
les prédictions, ni les losses, ni le nombre d'ablations.

Pendant les epochs, les rollouts ne construisent que les sorties nécessaires à
la loss. Les historiques complets de messages, graphes et node-states restent
activés par défaut pour l'évaluation détaillée. Enfin, les métriques sont
accumulées sur le device et transférées une seule fois à la fin de l'epoch afin
d'éviter une synchronisation MPS/CUDA pour chaque scalaire.

## 9. Fichiers d'évaluation supplémentaires

En plus des matrices et courbes génériques :

```text
v5_latent_diagnostics.json
v5_latent_probabilities.npz
```

Ils contiennent l'utilisation des types d'arête, les états de nœud, les
entropies, la moyenne et l'écart-type de `P(edge)` et de la strength. Une seule
classe active proche de 100 %, un écart-type de `P(edge)` presque nul ou une
entropie presque nulle sont explicitement signalés comme symptômes possibles
d'effondrement.
