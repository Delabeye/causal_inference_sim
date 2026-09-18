# SRI-inspired binary relational inference

This implementation reproduces the main architectural ideas of SRI without
using its repulsion/alignment edge ontology. Every directed pair has exactly
two categorical states:

```text
0 = no-edge
1 = edge
```

The encoder combines a relation GNN, receiver-local multi-head attention and
forward/reverse LSTMs. It infers a dynamic binary posterior, a causal dynamic
prior and a scalar strength conditional on the edge. The effective decoder
weight is:

```text
A_ij(t) = z_ij,edge(t) * strength_ij(t)
```

When the sampled type is `no-edge`, the inter-drone message is exactly zero.
Strength is not a third edge type.

Training is unsupervised with respect to the simulator graph. The loss uses
trajectory reconstruction, temporal posterior/prior KL, a sparse binary prior,
graph smoothness and a small effective-strength penalty. Simulator structural
and active matrices are used only for held-out evaluation.

The default experiment uses trajectories only: waypoint targets, offsets and
other controller context are not model inputs.

Train from the project root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.sri_binary.train
```

Evaluate an existing checkpoint:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.sri_binary.evaluate
```

Important outputs:

```text
best_model.pt
manifest.json
evaluation_report.json
training_history.csv
training_curves.png
test_sri_binary_graph.csv
test_run_<id>_sri_binary_graph.png
```
