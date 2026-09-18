# Checkpoint : matrice de formation et influence inter-drones

Date : 2026-07-29

## Point de départ

Dans le NRI/DRI actuel, une variable latente binaire

\[
z_{ij}^{t}\in\{0,1\}
\]

répond principalement à la question : « une relation de \(i\) vers \(j\)
existe-t-elle ? ». La probabilité

\[
\pi_{ij}^{t}=P(z_{ij}^{t}=1\mid X_{\leq t})
\]

mesure la confiance du modèle dans l'existence de l'arête. Elle ne mesure pas
directement l'intensité physique ou causale de l'influence.

## Limite identifiée

Une interaction inter-drone possède potentiellement :

- une direction ;
- une intensité ;
- un signe (attraction ou répulsion) ;
- une durée ou un retard ;
- un type (suivi, évitement, collision, communication).

Une variable binaire ne permet pas de distinguer une influence faible d'une
influence forte. De plus, si toutes les formations utilisent le même leader et
les mêmes followers, leur topologie binaire peut être identique malgré des
offsets géométriques différents.

## Recommandation retenue

Séparer deux objets :

1. Une matrice de topologie nominale de formation :

\[
A_{ij}^{\mathrm{form}}\in\{0,1\}.
\]

2. Une matrice dynamique d'influence :

\[
W_{ij}^{t}\in[-1,1]
\]

ou \(W_{ij}^{t}\in[0,1]\) si le signe n'est pas nécessaire.

Le message transmis dans le décodeur peut alors prendre la forme :

\[
m_{ij}^{t}
=
A_{ij}^{\mathrm{form}}W_{ij}^{t}
f_\theta(x_i^t,x_j^t).
\]

Une variante entièrement apprise sépare une porte d'existence et un poids :

\[
g_{ij}^{t}\sim\operatorname{Bernoulli}(\pi_{ij}^{t}),
\qquad
w_{ij}^{t}\sim\mathcal N(\mu_{ij}^{t},\sigma_{ij}^{t\,2}),
\]

\[
m_{ij}^{t}
=
g_{ij}^{t}w_{ij}^{t}f_\theta(x_i^t,x_j^t).
\]

## Interprétation causale à conserver

Une matrice apprise uniquement avec la loss de trajectoire indique surtout
quelles relations sont utiles à la prédiction. Elle ne constitue pas encore une
preuve d'influence causale.

Pour valider causalement \(W_{ij}^{t}\), comparer des runs contrefactuelles
strictement appariées, avec et sans intervention sur le drone \(i\) :

\[
\operatorname{Effect}_{i\rightarrow j}
=
\left\|
x_j^{\mathrm{perturbé}}
-
x_j^{\mathrm{baseline}}
\right\|.
\]

Les deux runs doivent être identiques avant l'intervention : mêmes seeds
effectifs, waypoints, état initial, formation et environnement.

## Matrice causale interventionnelle ajoutée

La référence causale est désormais calculée séparément du graphe NRI par
`analysis/causal_intervention_matrix.py`. Pour une intervention de force
\(\delta_j\) sur le drone sender \(j\), elle estime à l'horizon \(h\) :

\[
C_{j\rightarrow i}(h)
=
\frac{
\left\|x_i^{\mathrm{perturbé}}(t_{fin}+h)
-x_i^{\mathrm{baseline}}(t_{fin}+h)\right\|_2
}{\|\delta_j\|_2}.
\]

La convention des fichiers est \(C[receiver, sender]\). La diagonale est
réservée à l'effet direct de la manipulation. Elle reste dans la table
détaillée pour vérifier que l'intervention a eu un effet, mais elle est mise à
zéro dans les matrices inter-drones. Une paire jamais testée reste `NaN` : une
absence d'expérience ne doit pas être confondue avec un effet nul.

Par défaut :

- les runs doivent être identiques avant l'intervention à \(10^{-9}\) près ;
- les effets après le premier crash du sender ou du receiver sont censurés ;
- les horizons sont mesurés depuis la fin de l'intervention ;
- la médiane est utilisée pour agréger plusieurs interventions ;
- des matrices séparées sont produites par split et par identité du leader.

Une matrice est également exportée pour chaque paire baseline/perturbée dans
`causal_out/interventional_causal_matrix/pairs/`. Comme une paire ne contient
qu'une intervention, elle n'identifie que la colonne du sender manipulé. Les
autres colonnes restent `NaN`. La matrice `all` est une synthèse secondaire et
ne remplace pas ces résultats expérience par expérience.

Cette matrice mesure un effet total sous l'intervention du simulateur. Elle ne
doit pas être confondue avec une différence entre deux graphes NRI, une
ablation interne du décodeur, une causalité de Granger ou une probabilité
d'existence d'arête.

## Utilisation dans le modèle

La première utilisation est une évaluation tenue à part. Avec le NRI binaire
actuel, comparer \(P_\theta(edge)\) au support causal obtenu par seuillage de
\(C\), et comparer également son classement à celui des valeurs continues de
\(C\). Si une tête d'intensité \(S_\theta\) est ajoutée plus tard, on pourra
alors comparer \(P_\theta(edge)S_\theta\) directement aux intensités de \(C\).

Une future variante d'entraînement pourra utiliser uniquement la matrice du
split train avec une loss de classification, de valeur ou de ranking :

\[
\mathcal L
=\mathcal L_{prediction}
+\lambda_C\mathcal L_{causal}(A_\theta^{\mathrm{effective}},C_{train}).
\]

Les matrices validation/test ne doivent jamais entrer dans cette loss.
