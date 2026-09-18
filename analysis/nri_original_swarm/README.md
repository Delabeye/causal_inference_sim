# Original NRI baseline for the UAV logs

This directory adapts the public MIT-licensed `tkipf/nri` pipeline to the
simulator CSV files. It is deliberately isolated from the existing NRI/DRI
experiments.

## What is faithful to original NRI

- one latent graph for each input sequence;
- factor-graph MLP encoder (`node -> edge -> node -> edge`);
- two categorical edge types;
- Gumbel-Softmax samples;
- recurrent interaction-network decoder with a GRU-style hidden state;
- delta-state prediction;
- Gaussian NLL plus categorical KL;
- no adjacency BCE and no formation loss;
- ground-truth edges are used only for metrics.

The optimized objective is:

```text
loss = gaussian_nll(predicted_states, true_states)
       + kl_weight * categorical_kl(q_edges, prior)
```

`relations` is not referenced by either loss term.
`kl_weight=1` with a uniform prior recovers the original baseline objective.
The leader-switch experiment uses a sparse `(no-edge, edge)=(0.85, 0.15)`
prior and a stronger KL weight; its output directory is separate from the
uniform-prior baseline.

## Necessary adaptations

- UAV states have six dimensions: 3D position and 3D velocity.
- The encoder conditions the latent graph on a separate exogenous context
  tensor: navigation intent, desired formation offset, wind, obstacle geometry,
  and controlled external force.
- The recurrent decoder injects this local context into its GRU-style gates;
  context never passes through inter-drone messages and is not reconstructed.
- A follower's logged `target_*` already contains leader information. It is
  therefore masked, while `desired_offset_*` is retained. This prevents the
  decoder from bypassing the latent graph through a precomputed follower target.
- Long runs are converted to fixed-length sequences.
- The recurrent decoder retains a node memory across the complete sequence and
  uses ground truth inputs every `prediction_steps` during training.
- normalization is fitted on training runs only.
- train/validation/test are split by run, never by window.
- the diagonal is excluded and edges are ordered as matrix entries
  `A[receiver, sender]`.

Original NRI assumes one static physical graph per trajectory. The default
`ground_truth_mode="structural"` therefore evaluates the fixed
leader-to-follower controller graph. Set it to `"active"` to aggregate the
time-varying `interaction_active` column inside each sequence; this is useful,
but it is no longer the exact static-graph setting of the original paper.

## Configure and train

Edit `config.py`, especially:

```python
log_dir
output_dir
drone_names
context_feature_columns
downsample
seq_len
ground_truth_mode
epochs
batch_size
```

Then, from the project root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.nri_original_swarm.train
```

To evaluate the existing `best_model.pt` and regenerate the test plots without
training again:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.nri_original_swarm.evaluate
```

If `plan.csv` contains a `split` column, those run-level splits are used.
Otherwise, run/seed groups are deterministically split. Explicit lists in
`config.py` take priority over both methods.

## Outputs

The configured output directory receives:

```text
best_model.pt
manifest.json
training_history.csv
training_curves.png
final_report.json
test_window_edges.csv
test_run_graphs.csv
test_run_<id>_graph.png
test_trajectories.csv
test_run_<id>_window_<n>_trajectories.png
evaluation_report.json
```

Each trajectory figure compares the real test trajectory with:

- `One-step prediction`: the decoder starts from the real state at every step;
- `Autoregressive rollout`: the decoder sees only the first state, then feeds
  each prediction back as its next input.

The CSV stores both predictions and the truth in physical units. Position RMSE
in metres and velocity RMSE in metres per second are included in the reports.
The original NRI encoder still uses the complete test window to infer its one
latent graph, so this evaluates trajectory reconstruction/rollout conditioned
on that window rather than strict online forecasting from past data only.
The rollout also receives the recorded exogenous context at every future step.
It is consequently a scenario-conditioned prediction. For deployment with
unknown future wind or interventions, those channels must be forecast, fixed by
the experiment, or held at their last observed value.

`edge_accuracy` is the direct type accuracy used by original NRI.
`edge_accuracy_permutation` also reports the best binary label permutation,
because unsupervised latent type numbers can be exchanged.

For `ground_truth_mode="active"`, every selected run must contain
`run_<id>_interactions.csv`, unless `require_ground_truth=False`. Missing truth
is then represented internally by `-1` and never enters the loss.
