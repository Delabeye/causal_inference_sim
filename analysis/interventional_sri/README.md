# Interventional SRI

Ce dossier contient la version exécutable de notre modèle théorique. Le
backbone est inspiré de SRI/dNRI, mais la cible relationnelle est volontairement
plus simple : une arête dirigée existe ou non, et possède une intensité
continue. Le nouvel élément causal est l'entraînement sur deux futurs physiques
issus du même checkpoint.

## 1. Exemple d'apprentissage

Après génération des runs avec forks :

```bash
python -m analysis.interventional_sri.train \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_dynamic_v3
```

Pour vérifier rapidement la pipeline :

```bash
python -m analysis.interventional_sri.train \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_dynamic_v3_smoke \
  --epochs 2
```

Les paramètres de données, d'architecture et de loss sont centralisés dans
`config.py`.

## 2. Échantillon d'entrée

Chaque échantillon est construit à partir d'un manifeste
`run_<id>_counterfactual_forks.json` ou d'une trace nominale.

Les tenseurs ont les formes suivantes :

```text
history_states       [B, N, L, 6]
intervention_input   [B, N, H, 4]
baseline_future      [B, N, H, 6]
intervention_future  [B, N, H, 6]
paired               [B]
```

Les six variables d'état sont :

```text
x, y, z, vx, vy, vz
```

Les quatre variables d'intervention sont :

```text
force_x, force_y, force_z, active
```

Avec les valeurs par défaut, les traces de contrôle synchronisées à 80 Hz sont
sous-échantillonnées à 10 Hz : `L=20` représente 2 s d'historique et `H=80`
représente 8 s
de rollout.

Le chargeur garde ensemble les runs appariés lors du découpage
train/validation/test. La normalisation est ajustée uniquement sur le split
d'entraînement.

## 3. Encodeur

L'encodeur reçoit uniquement les informations disponibles avant
l'intervention courante :

\[
H_t=X_{t-L+1:t}.
\]

L'API accepte aussi un contexte historique et les interventions passées :

\[
q_\phi(Z_t,S_t\mid H_t,C_t,I_{<t}).
\]

Le descripteur de l'intervention courante `intervention_input` n'est jamais
transmis à l'encodeur.

### Encodage temporel des nœuds

Un GRU partagé transforme l'historique de chaque drone en représentation
temporelle :

\[
h_i=\operatorname{GRU}_\phi(H_i).
\]

### Passage node-to-edge

Pour chaque paire ordonnée `sender i -> receiver j` :

\[
e_{ji}=\operatorname{MLP}_{edge}([h_i,h_j]).
\]

### Attention entre relations

Les arêtes candidates ayant le même receiver communiquent par self-attention.
Cela permet par exemple aux candidats `0->2`, `1->2` et `3->2` d'être comparés
avant de décider lequel explique le mouvement de `drone_2`.

## 4. Sortie et posterior

L'encodeur possède deux têtes.

### Existence

\[
q_\phi(Z_{ji}=k\mid H_t)
=\operatorname{Categorical}(\pi_{ji}),
\qquad k\in\{0,1\}.
\]

`0` signifie aucune interaction directe et `1` une interaction directe. Un
échantillon différentiable est obtenu par Gumbel-Softmax pendant
l'entraînement.

### Intensité

\[
S_{ji}=\operatorname{sigmoid}
\left(f_{strength}(e_{ji})\right)\in[0,1].
\]

L'adjacence effective est :

\[
A_{ji}=Z_{ji}S_{ji}.
\]

Le code expose à la fois l'échantillon straight-through, la moyenne du
posterior et l'arête effectivement choisie par le mode courant du décodeur :

```text
existence_probability
existence_sample
strength
effective_edge_sample
effective_edge_mean = P(edge) * strength
baseline/intervention_graph_decoder_edge
```

## 5. Décodeur

Le décodeur est un GNN récurrent partagé entre les deux branches. Pour chaque
arête :

\[
m_{j\leftarrow i}^t
=A_{ji}\,g_\theta(h_i^t,h_j^t).
\]

Les messages entrants sont sommés :

\[
m_j^t=\sum_{i\ne j}m_{j\leftarrow i}^t.
\]

La mise à jour du nœud reçoit son état, les messages, le contexte et
l'intervention :

\[
h_j^{t+1}=\operatorname{GRU}_\theta
(x_j^t,m_j^t,u_j^t,c_j^t,h_j^t),
\]

\[
\hat x_j^{t+1}=x_j^t+f_{out}(h_j^{t+1}).
\]

L'intensité possède donc un rôle explicite : elle multiplie réellement le
message relationnel dans le décodeur.

### Prior causal dynamique du graphe

Après avoir prédit l'état suivant, un GRU d'arête met à jour le graphe sans
consulter la future trajectoire réelle :

\[
h_{ji,G}^{t+1}=\operatorname{GRU}_G\left(
[\hat x_j^{t+1},\hat x_i^{t+1},P_{ji}^{t},S_{ji}^{t}],
h_{ji,G}^{t}
\right).
\]

Les mises à jour sont résiduelles pour éviter que la matrice change
brutalement :

\[
P_{ji}^{t+1}=\operatorname{Softmax}\left(
\log P_{ji}^{t}+\alpha f_G(h_{ji,G}^{t+1})
\right),
\]

\[
S_{ji}^{t+1}=\operatorname{sigmoid}\left(
\operatorname{logit}(S_{ji}^{t})+
\alpha f_S(h_{ji,G}^{t+1})
\right).
\]

Le graphe moyen au pas suivant est donc :

\[
\bar A_{ji}^{t+1}=P(Z_{ji}^{t+1}=1)S_{ji}^{t+1}.
\]

### Curriculum soft vers discret

Au début de l'entraînement, le décodeur utilise l'espérance continue

\[
A_{soft}=P(Z=1)S,
\]

qui donne des gradients stables à l'existence et à l'intensité. Après le warm-up,
il passe progressivement à l'échantillon straight-through

\[
A_{dec}=(1-\rho)A_{soft}+\rho A_{ST},
\qquad \rho:0\rightarrow1.
\]

La température Gumbel diminue simultanément de `1.0` à `0.3`. Les messages
relationnels sont bornés par `LayerNorm + tanh`; leur MLP ne peut donc pas
annuler une petite intensité en produisant des messages arbitrairement grands.

## 6. Double rollout

L'encodeur n'est exécuté qu'une fois. Les deux décodeurs sont en réalité deux
appels au même module avec :

```text
même échantillon de graphe initial
même état initial
même mémoire GRU initiale
même mémoire de prior de graphe initiale
mêmes bruits Gumbel couplés
mêmes paramètres
```

Branche nominale :

\[
\hat X^0_{t+1:t+H}=D_\theta(X_t,A^0_{t:t+H-1},U=0).
\]

Branche intervention :

\[
\hat X^I_{t+1:t+H}=D_\theta(X_t,A^I_{t:t+H-1},U_t).
\]

L'effet prédit est :

\[
\widehat{\Delta X}=\hat X^I-\hat X^0.
\]

Le décodeur ne contient pas de dropout afin que deux rollouts recevant les
mêmes interventions soient rigoureusement identiques, y compris pendant
l'entraînement. Les graphes commencent identiquement, puis peuvent diverger
uniquement lorsque les états prédits des deux branches divergent.

## 7. Loss

Tous les échantillons, y compris les runs sans perturbation, contribuent à la
loss nominale :

\[
\mathcal L_0
=\frac{1}{2\sigma_x^2}
\|X^0-\hat X^0\|^2.
\]

Seuls les échantillons possédant un fork contribuent aux deux termes suivants :

\[
\mathcal L_I
=\frac{1}{2\sigma_x^2}
\|X^I-\hat X^I\|^2,
\]

\[
\mathcal L_\Delta
=\|(\hat X^I-\hat X^0)-(X^I-X^0)\|^2.
\]

Le masque `paired` empêche les runs nominaux d'être traités comme des
interventions nulles dans la loss causale.

Le posterior d'existence est régularisé vers un prior sparse :

\[
\mathcal L_{KL}
=D_{KL}(q_\phi(Z\mid H)\|p(Z)).
\]

L'adjacence effective reste mesurée comme diagnostic :

\[
\mathcal D_S=\mathbb E[P(Z=1)S].
\]

mais son poids est nul par défaut (`lambda_strength=0`). La sparsité agit sur
`P(edge)` via le KL, jamais directement sur `P(edge)S`; sinon l'optimum facile
consiste à faire s'effondrer l'intensité.

La mise à jour dynamique est légèrement régularisée par :

\[
\mathcal L_{smooth}
=\mathbb E\left[|A^{t+1}-A^t|\right].
\]

Enfin, sur les nœuds non ciblés mais réellement affectés par un fork, une loss
de nécessité impose au graphe complet de battre la prédiction sans graphe. Sans
arêtes, l'intervention ne peut atteindre que le drone ciblé et l'effet prédit
sur les autres est exactement nul :

\[
\mathcal L_{necessary}
=\max\left(0,m+
\frac{\mathcal E_{graph}(\Delta X)}
{\mathcal E_{zero}(\Delta X)+\epsilon}-1\right).
\]

La marge par défaut demande une amélioration relative d'au moins 5 %, ce qui
évite que ce terme devienne négligeable devant la NLL de trajectoire.

La loss complète est :

\[
\mathcal L
=\mathcal L_0
+\lambda_I\mathcal L_I
+\lambda_\Delta\mathcal L_\Delta
+\beta\mathcal L_{KL}
+\lambda_{smooth}\mathcal L_{smooth}
+\lambda_{necessary}\mathcal L_{necessary}.
\]

Les termes intervention et effet sont normalisés uniquement par le nombre de
forks valides. Ils ne sont donc pas artificiellement réduits lorsque le batch
contient beaucoup de runs sans perturbation.

## 8. Fichiers

```text
model.py   encodeur, posterior, décodeur, double rollout et loss
data.py    lecture des NPZ/manifeste et création des paires parent/fork
config.py  hyperparamètres modifiables
train.py   entraînement, validation, test et sauvegarde du checkpoint
evaluate.py évaluation reproductible du checkpoint sauvegardé
reporting.py métriques physiques, comparaisons et figures d'évaluation
```

## 9. Évaluation visible

Après l'entraînement, ou pour réévaluer un checkpoint existant :

```bash
python -m analysis.interventional_sri.evaluate \
  --log-dir datasets/all_formations_forks_60s \
  --output-dir causal_out/interventional_sri_dynamic_v3
```

L'évaluateur écrit dans le dossier de sortie :

- `evaluation_summary.md` et `evaluation_report.json` ;
- `training_curves.png` ;
- `horizon_errors.png`, en mètres et mètres par seconde sur les huit secondes ;
- `counterfactual_examples.png`, avec les trajectoires parent/fork réelles et
  prédites ;
- `decoder_graph_ablation.png` et `decoder_graph_mode_metrics.csv`, comparant
  le même checkpoint avec `soft`, `hard`, `zero` et `shuffled` ;
- `edge_threshold_calibration.json`, contenant le seuil discret appris sur la
  validation ;
- `interaction_graphs.png`, comparant existence, intensité et graphes du
  simulateur ;
- `run_mean_graphs/run_<id>_mean_graphs.png`, une moyenne séparée pour chaque
  run ;
- `run_graph_evolution/run_<id>_graph_evolution.png`, l'évolution des douze
  arêtes dirigées aux snapshots évalués ;
- `snapshot_graph_matrices.npz`, les matrices non moyennées à chaque snapshot ;
- `rollout_graph_matrices.npz`, les 80 matrices causales prédites dans chaque
  branche baseline/intervention ;
- `rollout_graph_evolution_examples.png`, la divergence des graphes au cours
  des huit secondes ;
- `formation_comparison.png` et `intervention_target_comparison.png` ;
- les valeurs numériques détaillées dans les fichiers CSV.

La hiérarchie d'interprétation est donc `snapshot -> évolution causale dans le
rollout -> moyenne du run -> moyenne de formation -> moyenne globale`. Les
matrices futures sont produites par le prior causal sur les états prédits ; ce
ne sont pas des posteriors recalculés avec la future trajectoire réelle.

Le mode `soft` est la prédiction principale. Le mode `hard` mesure le coût de
la discrétisation. Les modes `zero` et `shuffled` vérifient que le décodeur se
sert réellement de la structure. Le rapport fournit notamment
`RMSE_zero - RMSE_soft` sur l'effet des nœuds non ciblés : cette valeur doit être
positive. Le seuil binaire de `P(edge)` est sélectionné sur le split de
validation en maximisant le F1 structurel, puis seulement appliqué au test.

La baseline « zero effect » prédit que l'intervention ne change rien :

\[
\widehat X^I-\widehat X^0=0.
\]

Le modèle causal n'est utile que si son erreur d'effet est inférieure à celle
de cette baseline. Les comparaisons dNRI/SRI/RiTINI ne sont déclarées valides
que lorsque ces modèles sont évalués sur exactement les mêmes fenêtres parent
et fork.

Le modèle est un SRI inspiré et non une reproduction exacte de l'article : il
remplace les catégories physiques `none/repulsion/alignment` par
`no-edge/edge`, et rend l'intensité directement opératoire dans le message du
décodeur. L'intervention explicite, le fork physique et la loss différentielle
sont les ajouts causaux propres à ce projet.
