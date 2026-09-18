#!/usr/bin/env python3
"""Compare controller-ablation matrices from a deterministic run pair.

The matrices follow ``A[receiver, sender]``. This script visualizes the direct
controller effect in the baseline and perturbed runs; the external perturbation
itself is not added to either adjacency matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROL_CHANNELS = ("target_delta_norm", "rpm_delta_norm")


def _read_plan_metadata(log_dir: Path, run_id: int) -> dict[str, Any]:
    path = log_dir / "plan.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["run_id"])) == int(run_id):
                return row
    return {}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    return float(value)


def load_control_run(log_dir: Path, run_id: int) -> dict[str, Any]:
    path = log_dir / f"run_{int(run_id)}_ground_truth_matrices.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate the run with the relational logger enabled."
        )
    with np.load(path) as payload:
        missing = {
            "times",
            "drone_names",
            "counterfactual_valid",
            *CONTROL_CHANNELS,
        } - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing channels: {sorted(missing)}")
        return {
            "path": path,
            "times": np.asarray(payload["times"], dtype=float),
            "names": [str(name) for name in payload["drone_names"].tolist()],
            "valid": np.asarray(payload["counterfactual_valid"], dtype=float),
            **{
                channel: np.asarray(payload[channel], dtype=float)
                for channel in CONTROL_CHANNELS
            },
        }


def validate_pair(baseline: dict[str, Any], perturbed: dict[str, Any]) -> None:
    if baseline["names"] != perturbed["names"]:
        raise ValueError("Baseline and perturbed runs use different drone ordering.")
    if baseline["times"].shape != perturbed["times"].shape or not np.allclose(
        baseline["times"], perturbed["times"], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "The pair does not have identical timestamps. Run the paired "
            "determinism checker before comparing controller matrices."
        )
    expected = (
        baseline["times"].size,
        len(baseline["names"]),
        len(baseline["names"]),
    )
    for run in (baseline, perturbed):
        if run["valid"].shape != expected:
            raise ValueError(f"Invalid counterfactual_valid shape: {run['valid'].shape}")
        for channel in CONTROL_CHANNELS:
            if run[channel].shape != expected:
                raise ValueError(f"Invalid {channel} shape: {run[channel].shape}")


def phase_masks(
    times: np.ndarray,
    event_time: float,
    event_duration: float,
    phase_window: float,
) -> dict[str, np.ndarray]:
    event_end = event_time + event_duration
    return {
        "before": (times >= event_time - phase_window) & (times < event_time),
        "during": (times >= event_time) & (times < event_end),
        "after": (times >= event_end) & (times < event_end + phase_window),
    }


def masked_channel(run: dict[str, Any], channel: str) -> np.ndarray:
    values = np.asarray(run[channel], dtype=float).copy()
    valid = np.asarray(run["valid"], dtype=float) > 0.5
    values[~valid] = np.nan
    diagonal = np.arange(values.shape[1])
    values[:, diagonal, diagonal] = 0.0
    return values


def aggregate_matrix(values: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    selected = values[mask]
    if selected.shape[0] == 0:
        raise ValueError("A requested phase contains no matrix samples.")
    with np.errstate(invalid="ignore"):
        if method == "mean":
            result = np.nanmean(selected, axis=0)
        elif method == "median":
            result = np.nanmedian(selected, axis=0)
        elif method == "max":
            result = np.nanmax(selected, axis=0)
        else:
            raise ValueError(f"Unknown aggregation '{method}'.")
    np.fill_diagonal(result, 0.0)
    return result


def _finite_limit(matrices: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = np.concatenate(
        [matrix[np.isfinite(matrix)].reshape(-1) for matrix in matrices]
    )
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.percentile(positive, percentile)), 1e-12)


def _annotate_matrix(axis, matrix: np.ndarray, value_limit: float) -> None:
    size = matrix.shape[0]
    for receiver in range(size):
        for sender in range(size):
            value = matrix[receiver, sender]
            if not np.isfinite(value):
                label = "NA"
            elif abs(value) >= 100.0:
                label = f"{value:.0f}"
            elif abs(value) >= 1.0:
                label = f"{value:.2f}"
            else:
                label = f"{value:.3f}"
            color = "white" if abs(value) > 0.55 * value_limit else "black"
            axis.text(sender, receiver, label, ha="center", va="center", fontsize=7, color=color)


def plot_phase_matrices(
    *,
    baseline_values: np.ndarray,
    perturbed_values: np.ndarray,
    masks: dict[str, np.ndarray],
    names: list[str],
    channel: str,
    aggregation: str,
    baseline_run_id: int,
    perturbed_run_id: int,
    output_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    phase_matrices: dict[str, dict[str, np.ndarray]] = {}
    absolute_matrices = []
    difference_matrices = []
    for phase, mask in masks.items():
        baseline_matrix = aggregate_matrix(baseline_values, mask, aggregation)
        perturbed_matrix = aggregate_matrix(perturbed_values, mask, aggregation)
        difference = perturbed_matrix - baseline_matrix
        phase_matrices[phase] = {
            "baseline": baseline_matrix,
            "perturbed": perturbed_matrix,
            "difference": difference,
        }
        absolute_matrices.extend([baseline_matrix, perturbed_matrix])
        difference_matrices.append(difference)

    value_limit = _finite_limit(absolute_matrices)
    difference_limit = _finite_limit([np.abs(matrix) for matrix in difference_matrices])
    figure, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    phase_labels = {"before": "Avant", "during": "Pendant", "after": "Après"}
    image = None
    difference_image = None
    for row, phase in enumerate(("before", "during", "after")):
        matrices = phase_matrices[phase]
        for column, key in enumerate(("baseline", "perturbed", "difference")):
            axis = axes[row, column]
            matrix = matrices[key]
            if key == "difference":
                image_here = axis.imshow(
                    matrix,
                    cmap="coolwarm",
                    vmin=-difference_limit,
                    vmax=difference_limit,
                )
                difference_image = image_here
                annotation_limit = difference_limit
            else:
                image_here = axis.imshow(
                    matrix, cmap="viridis", vmin=0.0, vmax=value_limit
                )
                image = image_here
                annotation_limit = value_limit
            _annotate_matrix(axis, matrix, annotation_limit)
            axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
            axis.set_yticks(range(len(names)), names)
            axis.set_xlabel("Sender")
            axis.set_ylabel("Receiver")
            if row == 0:
                title = {
                    "baseline": f"Run {baseline_run_id} — base",
                    "perturbed": f"Run {perturbed_run_id} — perturbé",
                    "difference": "Perturbé − base",
                }[key]
                axis.set_title(title)
            if column == 0:
                axis.text(
                    -0.42,
                    0.5,
                    phase_labels[phase],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
    label = {
        "target_delta_norm": "Effet sur la cible PID (m)",
        "rpm_delta_norm": "Effet sur la commande moteurs (norme RPM)",
    }[channel]
    figure.suptitle(f"Matrices de contrôle — {label}", fontsize=15)
    if image is not None:
        figure.colorbar(image, ax=axes[:, :2], shrink=0.72, label=label)
    if difference_image is not None:
        figure.colorbar(
            difference_image,
            ax=axes[:, 2],
            shrink=0.72,
            label="Différence signée",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return phase_matrices


def _off_diagonal_mean(values: np.ndarray) -> np.ndarray:
    count = values.shape[1]
    mask = ~np.eye(count, dtype=bool)
    return np.nanmean(values[:, mask], axis=1)


def plot_difference_timeline(
    times: np.ndarray,
    baseline: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    event_time: float,
    event_duration: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for axis, channel in zip(axes, CONTROL_CHANNELS):
        base_mean = _off_diagonal_mean(baseline[channel])
        perturbed_mean = _off_diagonal_mean(perturbed[channel])
        difference_mean = _off_diagonal_mean(
            np.abs(perturbed[channel] - baseline[channel])
        )
        axis.plot(times, base_mean, label="Base", linewidth=1.3)
        axis.plot(times, perturbed_mean, label="Perturbé", linewidth=1.3)
        axis.plot(times, difference_mean, label="|Perturbé − base|", linewidth=1.5)
        axis.axvspan(
            event_time,
            event_time + event_duration,
            color="tab:red",
            alpha=0.14,
            label="Intervention",
        )
        axis.grid(alpha=0.25)
        axis.set_ylabel(
            "Delta cible (m)"
            if channel == "target_delta_norm"
            else "Delta commande (RPM)"
        )
        axis.legend(loc="upper left", ncols=4, fontsize=8)
    axes[-1].set_xlabel("Temps simulé (s)")
    figure.suptitle("Évolution temporelle des effets directs du contrôleur")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def export_phase_tables(
    phase_matrices: dict[str, dict[str, np.ndarray]],
    names: list[str],
    channel: str,
    output_dir: Path,
) -> None:
    for phase, matrices in phase_matrices.items():
        for kind, matrix in matrices.items():
            pd.DataFrame(matrix, index=names, columns=names).to_csv(
                output_dir / f"{channel}_{phase}_{kind}.csv",
                index_label="receiver\\sender",
            )


def edge_summary(
    *,
    times: np.ndarray,
    baseline: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    names: list[str],
    aggregation: str,
) -> pd.DataFrame:
    rows = []
    for channel in CONTROL_CHANNELS:
        difference = perturbed[channel] - baseline[channel]
        for receiver, receiver_name in enumerate(names):
            for sender, sender_name in enumerate(names):
                if receiver == sender:
                    continue
                row: dict[str, Any] = {
                    "channel": channel,
                    "sender": sender_name,
                    "receiver": receiver_name,
                }
                for phase, mask in masks.items():
                    base = aggregate_matrix(baseline[channel], mask, aggregation)
                    pert = aggregate_matrix(perturbed[channel], mask, aggregation)
                    row[f"baseline_{phase}"] = float(base[receiver, sender])
                    row[f"perturbed_{phase}"] = float(pert[receiver, sender])
                    row[f"difference_{phase}"] = float(
                        pert[receiver, sender] - base[receiver, sender]
                    )
                edge_difference = np.abs(difference[:, receiver, sender])
                if np.any(np.isfinite(edge_difference)):
                    max_index = int(np.nanargmax(edge_difference))
                    row["max_abs_difference"] = float(edge_difference[max_index])
                    row["time_of_max_difference"] = float(times[max_index])
                else:
                    row["max_abs_difference"] = np.nan
                    row["time_of_max_difference"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["channel", "max_abs_difference"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize baseline and perturbed controller-ablation matrices."
    )
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=int, required=True)
    parser.add_argument("--perturbed-run", type=int, required=True)
    parser.add_argument("--event-time", type=float, default=None)
    parser.add_argument("--event-duration", type=float, default=None)
    parser.add_argument("--phase-window", type=float, default=2.0)
    parser.add_argument("--aggregation", choices=["mean", "median", "max"], default="mean")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline = load_control_run(args.logs_dir, args.baseline_run)
    perturbed = load_control_run(args.logs_dir, args.perturbed_run)
    validate_pair(baseline, perturbed)
    metadata = _read_plan_metadata(args.logs_dir, args.perturbed_run)
    event_time = args.event_time
    if event_time is None:
        event_time = _optional_float(metadata.get("intervention_start_time"))
    event_duration = args.event_duration
    if event_duration is None:
        event_duration = _optional_float(metadata.get("intervention_duration"))
    if event_time is None or event_duration is None:
        raise ValueError(
            "Provide --event-time and --event-duration, or include them in plan.csv."
        )
    if args.phase_window <= 0.0 or event_duration <= 0.0:
        raise ValueError("Phase window and event duration must be positive.")

    output_dir = args.output_dir or (
        args.logs_dir
        / f"control_matrix_comparison_{args.baseline_run}_vs_{args.perturbed_run}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = phase_masks(
        baseline["times"], event_time, event_duration, args.phase_window
    )
    masked_baseline = {
        channel: masked_channel(baseline, channel) for channel in CONTROL_CHANNELS
    }
    masked_perturbed = {
        channel: masked_channel(perturbed, channel) for channel in CONTROL_CHANNELS
    }

    outputs = {}
    for channel in CONTROL_CHANNELS:
        plot_path = output_dir / f"{channel}_phase_matrices.png"
        matrices = plot_phase_matrices(
            baseline_values=masked_baseline[channel],
            perturbed_values=masked_perturbed[channel],
            masks=masks,
            names=baseline["names"],
            channel=channel,
            aggregation=args.aggregation,
            baseline_run_id=args.baseline_run,
            perturbed_run_id=args.perturbed_run,
            output_path=plot_path,
        )
        export_phase_tables(matrices, baseline["names"], channel, output_dir)
        outputs[channel] = str(plot_path.resolve())

    timeline_path = output_dir / "control_matrix_difference_timeline.png"
    plot_difference_timeline(
        baseline["times"],
        masked_baseline,
        masked_perturbed,
        event_time,
        event_duration,
        timeline_path,
    )
    edge_table = edge_summary(
        times=baseline["times"],
        baseline=masked_baseline,
        perturbed=masked_perturbed,
        masks=masks,
        names=baseline["names"],
        aggregation=args.aggregation,
    )
    edge_path = output_dir / "control_edge_changes.csv"
    edge_table.to_csv(edge_path, index=False)

    pre_mask = baseline["times"] < event_time
    pre_differences = {
        channel: float(
            np.nanmax(
                np.abs(
                    masked_perturbed[channel][pre_mask]
                    - masked_baseline[channel][pre_mask]
                )
            )
        )
        for channel in CONTROL_CHANNELS
    }
    report = {
        "matrix_convention": "row=receiver,column=sender",
        "baseline_run_id": args.baseline_run,
        "perturbed_run_id": args.perturbed_run,
        "event_time": event_time,
        "event_duration": event_duration,
        "phase_window": args.phase_window,
        "aggregation": args.aggregation,
        "drone_names": baseline["names"],
        "preintervention_max_abs_difference": pre_differences,
        "preintervention_matrices_identical": all(
            value <= 1e-12 for value in pre_differences.values()
        ),
        "plots": {**outputs, "timeline": str(timeline_path.resolve())},
        "edge_table": str(edge_path.resolve()),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
