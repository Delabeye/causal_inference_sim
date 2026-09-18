"""Train multi-horizon Granger models on future position and velocity changes."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from .data import PreparedRuns, RunData, prepare_runs
from .model import Standardizer
from .multihorizon_config import CONFIG, Config
from .multihorizon_model import (
    MultiHorizonDeltaGranger,
    fit_multi_horizon_position_granger,
    fit_multi_horizon_velocity_granger,
)
from .train import (
    MATRIX_CONVENTION,
    SCORE_DEFINITION,
    _binary_graph_metrics,
    _fit_standardizers,
    _jsonable_config,
    _matrix_summary,
    _nri_run_level_metrics,
    _off_diagonal,
    _save_matrix_artifacts,
    _scaled_split,
    _select_validation_threshold,
)


def _combine(matrices: dict[int, np.ndarray]) -> np.ndarray:
    return np.maximum.reduce([matrices[horizon] for horizon in sorted(matrices)])


def _graph_values(
    runs: list[RunData],
    run_matrices: dict[int, dict[int, np.ndarray]],
    horizon: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for run in runs:
        labels.append(_off_diagonal(run.structural_relations))
        matrix = (
            _combine(run_matrices[run.run_id])
            if horizon is None
            else run_matrices[run.run_id][horizon]
        )
        scores.append(_off_diagonal(matrix))
    return np.concatenate(labels), np.concatenate(scores)


def _save_model(
    path: Path,
    model: MultiHorizonDeltaGranger,
    state_scaler: Standardizer,
    control_scaler: Standardizer | None,
    target: str,
) -> None:
    horizons = model.horizons
    np.savez_compressed(
        path,
        horizons=np.asarray(horizons, dtype=np.int64),
        full_coefficients=np.stack(
            [model.regressors[h].full_coefficients for h in horizons]
        ),
        restricted_coefficients=np.stack(
            [model.regressors[h].restricted_coefficients for h in horizons]
        ),
        restricted_columns=np.stack(
            [model.regressors[h].restricted_columns for h in horizons]
        ),
        state_mean=state_scaler.mean,
        state_scale=state_scaler.scale,
        control_mean=(
            control_scaler.mean if control_scaler is not None else np.empty(0)
        ),
        control_scale=(
            control_scaler.scale if control_scaler is not None else np.empty(0)
        ),
        target=np.asarray(target),
        matrix_convention=np.asarray(MATRIX_CONVENTION),
        score_definition=np.asarray(SCORE_DEFINITION),
    )


def _save_test_rows(
    output_dir: Path,
    runs: list[RunData],
    matrices: dict[int, dict[int, np.ndarray]],
    names: tuple[str, ...],
    lags: int,
    target_name: str,
) -> None:
    rows = []
    for run in runs:
        combined = _combine(matrices[run.run_id])
        for horizon, matrix in sorted(matrices[run.run_id].items()):
            for receiver, receiver_name in enumerate(names):
                for sender, sender_name in enumerate(names):
                    if receiver == sender:
                        continue
                    rows.append(
                        {
                            "run_id": run.run_id,
                            "paired_baseline_run_id": run.paired_baseline_run_id,
                            "horizon_steps": horizon,
                            "sender": sender_name,
                            "receiver": receiver_name,
                            f"granger_{target_name}_delta_score": float(
                                matrix[receiver, sender]
                            ),
                            f"combined_{target_name}_max_score": float(
                                combined[receiver, sender]
                            ),
                        }
                    )
        _save_matrix_artifacts(
            output_dir,
            (
                f"test_run_{run.run_id}_multihorizon_combined_matrix"
                if target_name == "velocity"
                else f"test_run_{run.run_id}_multihorizon_{target_name}_combined_matrix"
            ),
            combined,
            names,
            f"Run {run.run_id} - multi-horizon Granger on {target_name} changes",
            lags,
        )
    if rows:
        filename = (
            "test_run_multihorizon_scores.csv"
            if target_name == "velocity"
            else f"test_run_multihorizon_{target_name}_scores.csv"
        )
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _save_paired_differences(
    output_dir: Path,
    runs: list[RunData],
    matrices: dict[int, dict[int, np.ndarray]],
    names: tuple[str, ...],
    target_name: str,
) -> None:
    combined = {run_id: _combine(value) for run_id, value in matrices.items()}
    rows = []
    for run in runs:
        baseline_id = run.paired_baseline_run_id
        if baseline_id is None or baseline_id not in combined:
            continue
        difference = combined[run.run_id] - combined[baseline_id]
        for receiver, receiver_name in enumerate(names):
            for sender, sender_name in enumerate(names):
                if receiver == sender:
                    continue
                rows.append(
                    {
                        "baseline_run_id": baseline_id,
                        "perturbed_run_id": run.run_id,
                        "sender": sender_name,
                        "receiver": receiver_name,
                        f"combined_{target_name}_score_difference": float(
                            difference[receiver, sender]
                        ),
                    }
                )
    if rows:
        filename = (
            "paired_test_multihorizon_differences.csv"
            if target_name == "velocity"
            else f"paired_test_multihorizon_{target_name}_differences.csv"
        )
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _evaluate_target(
    target_name: str,
    model: MultiHorizonDeltaGranger,
    data: PreparedRuns,
    scaled_splits: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
    horizon_seconds: dict[str, float],
    output_dir: Path,
    config: Config,
) -> dict:
    pooled = {
        split: model.effect_matrices_for_runs(values)
        for split, values in scaled_splits.items()
    }
    run_matrices = {
        split: {
            run.run_id: model.effect_matrices(*scaled)
            for run, scaled in zip(getattr(data, split), scaled_splits[split])
        }
        for split in ("val", "test")
    }

    horizon_metrics = {}
    for horizon in model.horizons:
        val_values = _graph_values(data.val, run_matrices["val"], horizon)
        test_values = _graph_values(data.test, run_matrices["test"], horizon)
        threshold, val_metrics = _select_validation_threshold(*val_values)
        horizon_metrics[str(horizon)] = {
            "seconds": horizon_seconds[str(horizon)],
            "validation": val_metrics,
            "test": _binary_graph_metrics(*test_values, threshold),
        }
        for split in ("train", "val", "test"):
            _save_matrix_artifacts(
                output_dir,
                f"h{horizon}_{target_name}_delta_{split}_matrix",
                pooled[split][horizon],
                config.drone_names,
                (
                    f"{target_name.capitalize()}-change Granger - {split} - "
                    f"h={horizon_seconds[str(horizon)]:.3f}s"
                ),
                config.granger_lags,
            )

    combined_pooled = {split: _combine(values) for split, values in pooled.items()}
    combined_val = _graph_values(data.val, run_matrices["val"], None)
    combined_test = _graph_values(data.test, run_matrices["test"], None)
    combined_threshold, combined_val_metrics = _select_validation_threshold(
        *combined_val
    )
    combined_test_metrics = _binary_graph_metrics(
        *combined_test, combined_threshold
    )
    for split, matrix in combined_pooled.items():
        suffix = (
            f"multihorizon_combined_{split}_matrix"
            if target_name == "velocity"
            else f"multihorizon_{target_name}_combined_{split}_matrix"
        )
        _save_matrix_artifacts(
            output_dir,
            suffix,
            matrix,
            config.drone_names,
            (
                f"Multi-horizon {target_name}-change Granger - {split} - "
                "max horizons"
            ),
            config.granger_lags,
        )

    _save_test_rows(
        output_dir,
        data.test,
        run_matrices["test"],
        config.drone_names,
        config.granger_lags,
        target_name,
    )
    _save_paired_differences(
        output_dir,
        data.test,
        run_matrices["test"],
        config.drone_names,
        target_name,
    )
    return {
        "split_summary_combined": {
            split: _matrix_summary(matrix)
            for split, matrix in combined_pooled.items()
        },
        "per_horizon_structural_metrics": horizon_metrics,
        "combined_structural_metrics": {
            "threshold_selection": "maximum validation F1",
            "validation": combined_val_metrics,
            "test": combined_test_metrics,
        },
    }


def train(config: Config = CONFIG) -> dict:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data: PreparedRuns = prepare_runs(config)
    state_scaler, control_scaler = _fit_standardizers(data, config)
    scaled_splits = {
        split: _scaled_split(getattr(data, split), state_scaler, control_scaler)
        for split in ("train", "val", "test")
    }
    fit_kwargs = {
        "runs": scaled_splits["train"],
        "lags": config.granger_lags,
        "horizons": config.horizon_steps,
        "ridge": config.granger_ridge,
        "node_count": len(config.drone_names),
        "state_per_node": len(config.state_columns),
    }
    models = {
        "velocity": fit_multi_horizon_velocity_granger(**fit_kwargs),
        "position": fit_multi_horizon_position_granger(**fit_kwargs),
    }
    for target_name, model in models.items():
        _save_model(
            output_dir / f"granger_multihorizon_{target_name}_model.npz",
            model,
            state_scaler,
            control_scaler,
            f"{target_name}_delta_xyz",
        )

    sample_dt = float(
        np.median(np.concatenate([np.diff(run.times) for run in data.train]))
    )
    horizon_seconds = {
        str(horizon): float(horizon * sample_dt)
        for horizon in models["velocity"].horizons
    }
    target_results = {
        target_name: _evaluate_target(
            target_name,
            model,
            data,
            scaled_splits,
            horizon_seconds,
            output_dir,
            config,
        )
        for target_name, model in models.items()
    }

    nri_metrics = _nri_run_level_metrics(
        config.comparison_nri_dir, {run.run_id for run in data.test}
    )
    if nri_metrics is not None:
        comparison_rows = [
            {
                "model": f"granger_multihorizon_{target_name}_delta",
                **result["combined_structural_metrics"]["test"],
            }
            for target_name, result in target_results.items()
        ]
        comparison_rows.append(
            {"model": "nri_original_swarm_confirm", **nri_metrics}
        )
        with (output_dir / "comparison_with_nri_run_level.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)

    manifest = {
        "config": _jsonable_config(config),
        "split_run_ids": data.split_run_ids,
        "targets": {
            "velocity": "v_receiver[t+h] - v_receiver[t] in standardized coordinates",
            "position": "p_receiver[t+h] - p_receiver[t] in standardized coordinates",
        },
        "horizon_seconds": horizon_seconds,
        "matrix_convention": MATRIX_CONVENTION,
        "score_definition": SCORE_DEFINITION,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "method": (
            "conditional Ridge VAR history with multi-horizon position-delta "
            "and velocity-delta targets"
        ),
        "targets": ["delta_p_xyz", "delta_v_xyz"],
        "lags": config.granger_lags,
        "horizon_steps": list(models["velocity"].horizons),
        "horizon_seconds": horizon_seconds,
        "horizon_aggregation": "maximum score per directed edge",
        "matrix_convention": MATRIX_CONVENTION,
        "results_by_target": target_results,
        "comparison_with_nri": {
            "same_test_run_ids": True,
            "granger_velocity": target_results["velocity"][
                "combined_structural_metrics"
            ]["test"],
            "granger_position": target_results["position"][
                "combined_structural_metrics"
            ]["test"],
            "nri": nri_metrics,
        },
        "warning": (
            "These scores measure incremental predictability of future position "
            "or velocity changes, not physical intervention effects."
        ),
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
