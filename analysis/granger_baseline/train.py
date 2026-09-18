"""Fit and evaluate one conditional-Granger matrix baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .config import CONFIG, Config
from .data import PreparedRuns, RunData, prepare_runs
from .model import GrangerVAR, Standardizer, fit_granger_var


MATRIX_CONVENTION = "matrix[receiver, sender] means sender -> receiver"
SCORE_DEFINITION = (
    "max(0, 1 - exp(-max(log(SSE_restricted/SSE_full), 0)))"
)


def _jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def _fit_standardizers(
    data: PreparedRuns, config: Config
) -> tuple[Standardizer, Standardizer | None]:
    state_arrays = [run.states.reshape(len(run.times), -1) for run in data.train]
    state_scaler = Standardizer.fit(state_arrays)
    if not config.use_controls:
        return state_scaler, None
    control_arrays = [run.controls.reshape(len(run.times), -1) for run in data.train]
    return state_scaler, Standardizer.fit(control_arrays)


def _scaled_run(
    run: RunData,
    state_scaler: Standardizer,
    control_scaler: Standardizer | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    states = state_scaler.transform(run.states.reshape(len(run.times), -1))
    controls = (
        control_scaler.transform(run.controls.reshape(len(run.times), -1))
        if control_scaler is not None
        else None
    )
    return states, controls


def _scaled_split(
    runs: list[RunData],
    state_scaler: Standardizer,
    control_scaler: Standardizer | None,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    return [_scaled_run(run, state_scaler, control_scaler) for run in runs]


def _save_matrix_csv(path: Path, matrix: np.ndarray, names: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["receiver\\sender", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *map(float, row)])


def _save_matrix_artifacts(
    output_dir: Path,
    stem: str,
    matrix: np.ndarray,
    names: tuple[str, ...],
    title: str,
    lags: int,
) -> None:
    _save_matrix_csv(output_dir / f"{stem}.csv", matrix, names)
    np.save(output_dir / f"{stem}.npy", matrix)
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        matrix=matrix,
        drone_names=np.asarray(names),
        matrix_convention=np.asarray(MATRIX_CONVENTION),
        score_definition=np.asarray(SCORE_DEFINITION),
        granger_lags=np.asarray(lags),
    )
    _plot_matrix(output_dir / f"{stem}.png", matrix, names, title)


def _plot_matrix(
    path: Path,
    matrix: np.ndarray,
    names: tuple[str, ...],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(5.4, 4.7), constrained_layout=True)
    color_limit = max(0.05, float(np.nanmax(matrix, initial=0.0)))
    image = axis.imshow(matrix, vmin=0.0, vmax=color_limit, cmap="viridis")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set(xlabel="Sender i", ylabel="Receiver j", title=title)
    for receiver in range(len(names)):
        for sender in range(len(names)):
            value = matrix[receiver, sender]
            axis.text(
                sender,
                receiver,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.45 * color_limit else "black",
            )
    figure.colorbar(image, ax=axis, label="Conditional Granger score")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _matrix_summary(matrix: np.ndarray) -> dict[str, float]:
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {
        "mean_off_diagonal_score": float(off_diagonal.mean()),
        "max_off_diagonal_score": float(off_diagonal.max(initial=0.0)),
        "nonzero_edge_count": int(np.count_nonzero(off_diagonal > 0.0)),
    }


def _off_diagonal(values: np.ndarray) -> np.ndarray:
    mask = ~np.eye(values.shape[-1], dtype=bool)
    return values[mask]


def _binary_graph_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= float(threshold)).astype(np.int64)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positive_count = int((labels == 1).sum())
    if positive_count:
        order = np.argsort(-scores, kind="stable")
        ordered_labels = labels[order]
        cumulative_positive = np.cumsum(ordered_labels == 1)
        ranks = np.arange(1, len(labels) + 1)
        average_precision = float(
            (
                cumulative_positive
                / ranks
                * (ordered_labels == 1)
            ).sum()
            / positive_count
        )
    else:
        average_precision = 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((true_positive + true_negative) / max(len(labels), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": average_precision,
        "positive_edges": positive_count,
        "predicted_positive_edges": int((predictions == 1).sum()),
        "labeled_edges": int(len(labels)),
    }


def _select_validation_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = np.concatenate(
        [candidates, [float(candidates.max(initial=0.0) + 1e-12)]]
    )
    evaluated = [
        _binary_graph_metrics(labels, scores, threshold)
        for threshold in candidates
    ]
    best = max(
        evaluated,
        key=lambda item: (item["f1"], item["precision"], item["threshold"]),
    )
    return float(best["threshold"]), best


def _nri_run_level_metrics(
    comparison_dir: Path | None, expected_run_ids: set[int]
) -> dict[str, float | int] | None:
    if comparison_dir is None:
        return None
    path = Path(comparison_dir) / "test_run_graphs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing NRI run-level comparison file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"run_id", "mean_p_edge", "predicted_type", "true_type"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks columns {sorted(required)}")
    actual_run_ids = {int(row["run_id"]) for row in rows}
    if actual_run_ids != expected_run_ids:
        raise ValueError(
            "NRI and Granger test run ids differ: "
            f"only_nri={sorted(actual_run_ids - expected_run_ids)}, "
            f"only_granger={sorted(expected_run_ids - actual_run_ids)}"
        )
    labels = np.asarray([int(row["true_type"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["mean_p_edge"]) for row in rows])
    return _binary_graph_metrics(labels, scores, threshold=0.5)


def _save_model(
    path: Path,
    model: GrangerVAR,
    state_scaler: Standardizer,
    control_scaler: Standardizer | None,
) -> None:
    np.savez_compressed(
        path,
        full_coefficients=model.full_coefficients,
        restricted_coefficients=np.stack(model.restricted_coefficients),
        restricted_columns=np.stack(model.restricted_columns),
        state_mean=state_scaler.mean,
        state_scale=state_scaler.scale,
        control_mean=(control_scaler.mean if control_scaler is not None else np.empty(0)),
        control_scale=(control_scaler.scale if control_scaler is not None else np.empty(0)),
    )


def _load_model(
    path: Path, config: Config
) -> tuple[GrangerVAR, Standardizer, Standardizer | None]:
    with np.load(path, allow_pickle=False) as archive:
        state_scaler = Standardizer(archive["state_mean"], archive["state_scale"])
        control_scaler = (
            Standardizer(archive["control_mean"], archive["control_scale"])
            if archive["control_mean"].size
            else None
        )
        state_dim = len(config.drone_names) * len(config.state_columns)
        control_dim = (
            len(config.drone_names) * len(config.control_columns)
            if control_scaler is not None
            else 0
        )
        model = GrangerVAR(
            full_coefficients=archive["full_coefficients"],
            restricted_coefficients=list(archive["restricted_coefficients"]),
            restricted_columns=list(archive["restricted_columns"].astype(bool)),
            lags=config.granger_lags,
            state_dim=state_dim,
            control_dim=control_dim,
            node_count=len(config.drone_names),
            state_per_node=len(config.state_columns),
        )
    return model, state_scaler, control_scaler


def evaluate_model(
    model: GrangerVAR,
    state_scaler: Standardizer,
    control_scaler: Standardizer | None,
    data: PreparedRuns,
    config: Config,
    output_dir: Path,
) -> dict:
    scaled_splits = {
        split: _scaled_split(getattr(data, split), state_scaler, control_scaler)
        for split in ("train", "val", "test")
    }
    matrices = {
        split: model.effect_matrix_for_runs(runs)
        for split, runs in scaled_splits.items()
    }
    run_matrices_by_split = {
        split: {
            run.run_id: model.effect_matrix(*scaled)
            for run, scaled in zip(getattr(data, split), scaled_splits[split])
        }
        for split in ("train", "val", "test")
    }

    graph_values = {}
    for split in ("val", "test"):
        labels = []
        scores = []
        for run in getattr(data, split):
            labels.append(_off_diagonal(run.structural_relations))
            scores.append(_off_diagonal(run_matrices_by_split[split][run.run_id]))
        graph_values[split] = (np.concatenate(labels), np.concatenate(scores))
    threshold, validation_graph_metrics = _select_validation_threshold(
        *graph_values["val"]
    )
    test_graph_metrics = _binary_graph_metrics(
        *graph_values["test"], threshold
    )
    nri_graph_metrics = _nri_run_level_metrics(
        config.comparison_nri_dir,
        {run.run_id for run in data.test},
    )

    # This train-only matrix is the leakage-free artifact to reuse downstream.
    _save_matrix_artifacts(
        output_dir,
        "granger_baseline_matrix",
        matrices["train"],
        config.drone_names,
        f"Granger baseline - train - VAR({config.granger_lags})",
        config.granger_lags,
    )
    for split in ("val", "test"):
        _save_matrix_artifacts(
            output_dir,
            f"granger_{split}_matrix",
            matrices[split],
            config.drone_names,
            f"Granger diagnostic - {split} - VAR({config.granger_lags})",
            config.granger_lags,
        )

    run_rows: list[dict] = []
    run_matrices: dict[int, np.ndarray] = run_matrices_by_split["test"]
    plotted_ids = {
        run.run_id for run in data.test[: config.max_saved_test_runs]
    }
    for run in data.test:
        matrix = run_matrices[run.run_id]
        for receiver, receiver_name in enumerate(config.drone_names):
            for sender, sender_name in enumerate(config.drone_names):
                if receiver == sender:
                    continue
                run_rows.append(
                    {
                        "run_id": run.run_id,
                        "formation": run.formation,
                        "intervention_target": run.intervention_target,
                        "paired_baseline_run_id": run.paired_baseline_run_id,
                        "sender": sender_name,
                        "receiver": receiver_name,
                        "granger_score": float(matrix[receiver, sender]),
                    }
                )
        if run.run_id in plotted_ids:
            _save_matrix_artifacts(
                output_dir,
                f"test_run_{run.run_id}_granger_matrix",
                matrix,
                config.drone_names,
                f"Run {run.run_id} - conditional Granger",
                config.granger_lags,
            )

    if run_rows:
        with (output_dir / "test_run_granger_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)

    paired_rows: list[dict] = []
    for run in data.test:
        baseline_id = run.paired_baseline_run_id
        if baseline_id is None or baseline_id not in run_matrices:
            continue
        delta = run_matrices[run.run_id] - run_matrices[baseline_id]
        for receiver, receiver_name in enumerate(config.drone_names):
            for sender, sender_name in enumerate(config.drone_names):
                if receiver == sender:
                    continue
                paired_rows.append(
                    {
                        "baseline_run_id": baseline_id,
                        "perturbed_run_id": run.run_id,
                        "formation": run.formation,
                        "intervention_target": run.intervention_target,
                        "sender": sender_name,
                        "receiver": receiver_name,
                        "granger_score_delta": float(delta[receiver, sender]),
                    }
                )
    if paired_rows:
        with (output_dir / "paired_test_granger_differences.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

    report = {
        "method": f"conditional ridge VAR({config.granger_lags})",
        "baseline_source": "train split only",
        "score_definition": SCORE_DEFINITION,
        "matrix_convention": MATRIX_CONVENTION,
        "split_summary": {
            split: _matrix_summary(matrix) for split, matrix in matrices.items()
        },
        "structural_graph_comparison": {
            "threshold_selection": "maximum F1 on validation; tie-break by precision then sparsity",
            "validation": validation_graph_metrics,
            "test": test_graph_metrics,
        },
        "warning": (
            "Granger measures directed conditional predictability. A causal "
            "interpretation additionally requires assumptions or interventions."
        ),
    }
    if nri_graph_metrics is not None:
        comparison_rows = [
            {"model": "conditional_ridge_var", **test_graph_metrics},
            {"model": "nri_original_swarm_confirm", **nri_graph_metrics},
        ]
        comparison_path = output_dir / "comparison_with_nri_run_level.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        report["comparison_with_nri"] = {
            "granularity": "one directed graph score per test run and off-diagonal edge",
            "same_test_run_ids": True,
            "nri": nri_graph_metrics,
            "granger": test_graph_metrics,
            "artifact": str(comparison_path.resolve()),
        }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _config_from_manifest(manifest: dict, output_dir: Path) -> Config:
    valid = {item.name for item in fields(Config)}
    values = {key: value for key, value in manifest["config"].items() if key in valid}
    for key in ("log_dir", "output_dir"):
        values[key] = Path(values[key])
    for key in ("drone_names", "state_columns", "control_columns"):
        values[key] = tuple(values[key])
    for split in ("train", "val", "test"):
        values[f"{split}_run_ids"] = list(manifest["split_run_ids"][split])
    values["output_dir"] = output_dir
    values["max_saved_test_runs"] = CONFIG.max_saved_test_runs
    return Config(**values)


def evaluate_saved(output_dir: Path | None = None) -> dict:
    output_dir = Path(output_dir or CONFIG.output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    config = _config_from_manifest(manifest, output_dir)
    data = prepare_runs(config)
    model, state_scaler, control_scaler = _load_model(
        output_dir / "granger_model.npz", config
    )
    return evaluate_model(
        model, state_scaler, control_scaler, data, config, output_dir
    )


def train(config: Config = CONFIG) -> dict:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_runs(config)
    state_scaler, control_scaler = _fit_standardizers(data, config)
    scaled_train = _scaled_split(data.train, state_scaler, control_scaler)
    model = fit_granger_var(
        scaled_train,
        config.granger_lags,
        config.granger_ridge,
        len(config.drone_names),
        len(config.state_columns),
    )
    _save_model(output_dir / "granger_model.npz", model, state_scaler, control_scaler)
    manifest = {
        "config": _jsonable_config(config),
        "split_run_ids": data.split_run_ids,
        "skipped_runs": data.skipped_runs,
        "method": {
            "name": "conditional Granger",
            "internal_regression": f"ridge VAR({config.granger_lags})",
            "score": SCORE_DEFINITION,
            "baseline_artifact": "granger_baseline_matrix.npz",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return evaluate_model(
        model, state_scaler, control_scaler, data, config, output_dir
    )


if __name__ == "__main__":
    train()
