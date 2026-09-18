# RiTINI swarm baseline

This module adapts the public
[KrishnaswamyLab/RiTINI](https://github.com/KrishnaswamyLab/RiTINI)
architecture to the UAV tensors and run-level data splits in this repository.
It is a clean PyTorch implementation rather than a copy of the upstream code.

The retained RiTINI ingredients are:

- an LSTM that encodes a recent history for every node;
- learned temporal/lag attention;
- directed multi-head graph attention with `A[receiver, sender]` semantics;
- a nonlinear Graph Neural ODE vector field;
- fixed-step Euler or RK4 integration;
- trajectory reconstruction, graph-prior, attention-entropy and temporal graph
  smoothness losses;
- dynamic attention matrices as the inferred interaction graph.

The upstream package depends on `torch-geometric` and `torchdiffeq`. This
adaptation implements dense graph attention and fixed-step integration with
stock PyTorch, so the existing project environment is sufficient.

By default, the permissive graph is complete and `lambda_prior = 0`. Simulator
interaction matrices are therefore used only for held-out graph evaluation.
Set `prior_mode = "structural_mean"` and `lambda_prior > 0` in `config.py` only
when intentionally running an oracle-prior experiment. The prior is then built
from the training split only.

RiTINI-style perturbation training means that nominal and perturbed runs are
both reconstructed. It does **not** optimize the paired contrast
`(X_intervention - X_baseline)`; that belongs to the proposed causal extension,
not this baseline.

Train from the project root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.ritini_swarm.train
```

Evaluate an existing checkpoint:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.ritini_swarm.evaluate
```

Outputs are written under `causal_out/ritini_swarm/`:

```text
best_model.pt
evaluation_report.json
manifest.json
training_history.csv
training_curves.png
test_dynamic_attention.csv
```

The original RiTINI repository is distributed under Yale's non-commercial
license. Consult its license before combining or distributing upstream source
code for commercial use.
