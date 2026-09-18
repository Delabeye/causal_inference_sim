# Conditional Granger baseline

This package now exposes one graph only: a directed conditional-Granger matrix.
There is no separate LTI interaction matrix, trajectory rollout, or simulator
ground-truth matrix in this baseline.

## Method

A ridge VAR(p) is the regression used internally:

```text
X[t+1] = sum_k A[k] X[t-k] + B U[t] + c.
```

For each possible sender `i`, a restricted VAR is fitted without any lagged
state from `i`. For every receiver `j`, the score is:

```text
raw(i -> j) = log(SSE_restricted_j / SSE_full_j)
G[j, i] = 1 - exp(-max(raw(i -> j), 0))
```

The matrix convention is always:

```text
G[receiver, sender] = directed score sender -> receiver
```

The diagonal is zero because this baseline measures inter-drone Granger
influence. The controls (targets, wind, and external forces) are conditioning
variables when `use_controls = True`; they are not represented as graph nodes.

## Run

Edit `config.py`, then fit the baseline:

```bash
python -m analysis.granger_baseline.train
```

Recreate matrices from the saved VAR without fitting it again:

```bash
python -m analysis.granger_baseline.evaluate
```

## Outputs

The canonical downstream artifact is trained without validation/test leakage:

```text
causal_out/granger_baseline/granger_baseline_matrix.npz
```

It contains `matrix`, `drone_names`, `matrix_convention`, `score_definition`,
and `granger_lags`. CSV, NPY, and PNG versions are saved next to it. Validation,
test, and per-test-run Granger matrices are diagnostics only. The fitted ridge
VAR coefficients are stored in `granger_model.npz` because they are required to
recompute Granger scores, but they are not exported as another interaction
matrix.

This is predictive Granger causality. It can later be used as a fixed prior,
regularization target, or comparison baseline in a causality-informed neural
model, but it is not by itself proof of physical causation.

## Multi-horizon position- and velocity-change variant

The multi-horizon command fits two separate targets:
`velocity[t+h] - velocity[t]` and `position[t+h] - position[t]`, at
approximately 0.5, 1 and 2 seconds. It saves one directed matrix per target and
horizon, plus one combined matrix per target containing the maximum score
across horizons. Thresholds are selected on validation only.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m analysis.granger_baseline.train_multihorizon
```

Its configuration is in `multihorizon_config.py`; its default output directory
is `causal_out/granger_multihorizon_velocity_delta_nri_confirm`. The original
velocity filenames are retained for compatibility; position artifacts contain
`position_delta` or `position_combined` in their names.
