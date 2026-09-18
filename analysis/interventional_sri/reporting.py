"""Human-readable evaluation artifacts for the interventional SRI model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def collect_predictions(
    model,
    loader,
    rel_rec,
    rel_send,
    device,
    decoder_graph_mode: str = "soft",
) -> dict[str, np.ndarray]:
    """Run a deterministic pass with one explicit decoder graph mode."""

    model.eval()
    values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "history",
            "baseline_target",
            "intervention_target",
            "intervention_input",
            "paired",
            "run_id",
            "snapshot_time",
            "baseline_prediction",
            "intervention_prediction",
            "existence_probability",
            "strength",
            "effective_edge_mean",
            "baseline_graph_probability",
            "intervention_graph_probability",
            "baseline_graph_strength",
            "intervention_graph_strength",
            "baseline_graph_effective_mean",
            "intervention_graph_effective_mean",
        )
    }
    with torch.no_grad():
        for batch in loader:
            history = batch["history_states"].to(device)
            intervention_input = batch["intervention_input"].to(device)
            output = model(
                history,
                intervention_input,
                rel_rec,
                rel_send,
                decoder_graph_mode=decoder_graph_mode,
            )
            tensors = {
                "history": history,
                "baseline_target": batch["baseline_future"],
                "intervention_target": batch["intervention_future"],
                "intervention_input": batch["intervention_input"],
                "paired": batch["paired"],
                "run_id": batch["run_id"],
                "snapshot_time": batch["snapshot_time"],
                "baseline_prediction": output["baseline_prediction"],
                "intervention_prediction": output["intervention_prediction"],
                "existence_probability": output["existence_probability"][..., 1],
                "strength": output["strength"],
                "effective_edge_mean": output["effective_edge_mean"],
                "baseline_graph_probability": output[
                    "baseline_graph_probability"
                ][..., 1],
                "intervention_graph_probability": output[
                    "intervention_graph_probability"
                ][..., 1],
                "baseline_graph_strength": output["baseline_graph_strength"],
                "intervention_graph_strength": output[
                    "intervention_graph_strength"
                ],
                "baseline_graph_effective_mean": output[
                    "baseline_graph_effective_mean"
                ],
                "intervention_graph_effective_mean": output[
                    "intervention_graph_effective_mean"
                ],
            }
            for name, tensor in tensors.items():
                values[name].append(tensor.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def _state_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Position/velocity errors in physical units for ``[S,N,T,6]`` arrays."""

    if not len(prediction):
        return {}
    position_error = prediction[..., :3] - target[..., :3]
    velocity_error = prediction[..., 3:6] - target[..., 3:6]
    position_norm = np.linalg.norm(position_error, axis=-1)
    velocity_norm = np.linalg.norm(velocity_error, axis=-1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "position_ade_m": float(position_norm.mean()),
        "position_fde_m": float(position_norm[..., -1].mean()),
        "velocity_ade_mps": float(velocity_norm.mean()),
        "velocity_fde_mps": float(velocity_norm[..., -1].mean()),
    }


def _effect_metrics(
    baseline_prediction: np.ndarray,
    intervention_prediction: np.ndarray,
    baseline_target: np.ndarray,
    intervention_target: np.ndarray,
) -> dict[str, float]:
    return _state_metrics(
        intervention_prediction - baseline_prediction,
        intervention_target - baseline_target,
    )


def _horizon_errors(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = prediction - target
    return (
        np.linalg.norm(error[..., :3], axis=-1).mean(axis=(0, 1)),
        np.linalg.norm(error[..., 3:6], axis=-1).mean(axis=(0, 1)),
    )


def _plan_metadata(log_dir: Path) -> dict[int, dict[str, str]]:
    path = log_dir / "plan.csv"
    if not path.exists():
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("run_id", "")).strip()
            if text:
                result[int(float(text))] = {
                    "formation": str(row.get("formation", "unknown") or "unknown"),
                    "planned_target": str(
                        row.get("intervention_target", "none") or "none"
                    ),
                }
    return result


def _target_labels(inputs: np.ndarray, paired: np.ndarray, names: tuple[str, ...]) -> list[str]:
    magnitude = np.abs(inputs[..., :3]).sum(axis=(2, 3)) + inputs[..., 3].sum(axis=2)
    indices = magnitude.argmax(axis=1)
    return [names[index] if bool(is_paired) else "none" for index, is_paired in zip(indices, paired)]


def _sample_rows(
    arrays: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
    formations: list[str],
    targets: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(arrays["run_id"])):
        baseline = _state_metrics(
            physical["baseline_prediction"][index : index + 1],
            physical["baseline_target"][index : index + 1],
        )
        row: dict[str, Any] = {
            "run_id": int(arrays["run_id"][index]),
            "snapshot_time_s": float(arrays["snapshot_time"][index]),
            "formation": formations[index],
            "intervention_target": targets[index],
            "paired": int(arrays["paired"][index]),
            **{f"baseline_{key}": value for key, value in baseline.items()},
        }
        if arrays["paired"][index]:
            intervention = _state_metrics(
                physical["intervention_prediction"][index : index + 1],
                physical["intervention_target"][index : index + 1],
            )
            effect = _effect_metrics(
                physical["baseline_prediction"][index : index + 1],
                physical["intervention_prediction"][index : index + 1],
                physical["baseline_target"][index : index + 1],
                physical["intervention_target"][index : index + 1],
            )
            row.update({f"intervention_{key}": value for key, value in intervention.items()})
            row.update({f"effect_{key}": value for key, value in effect.items()})
        rows.append(row)
    return rows


def _group_metrics(
    arrays: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
    formations: list[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    paired = arrays["paired"].astype(bool)
    labels = np.asarray(formations)
    for formation in sorted(set(formations)):
        selected = labels == formation
        selected_paired = selected & paired
        report: dict[str, Any] = {
            "samples": int(selected.sum()),
            "paired_samples": int(selected_paired.sum()),
            "baseline": _state_metrics(
                physical["baseline_prediction"][selected],
                physical["baseline_target"][selected],
            ),
        }
        if selected_paired.any():
            report["intervention"] = _state_metrics(
                physical["intervention_prediction"][selected_paired],
                physical["intervention_target"][selected_paired],
            )
            report["effect"] = _effect_metrics(
                physical["baseline_prediction"][selected_paired],
                physical["intervention_prediction"][selected_paired],
                physical["baseline_target"][selected_paired],
                physical["intervention_target"][selected_paired],
            )
        result[formation] = report
    return result


def _target_metrics(
    arrays: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
    targets: list[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    paired = arrays["paired"].astype(bool)
    labels = np.asarray(targets)
    for target in sorted(set(labels[paired].tolist())):
        selected = paired & (labels == target)
        result[target] = {
            "paired_samples": int(selected.sum()),
            "baseline": _state_metrics(
                physical["baseline_prediction"][selected],
                physical["baseline_target"][selected],
            ),
            "intervention": _state_metrics(
                physical["intervention_prediction"][selected],
                physical["intervention_target"][selected],
            ),
            "effect": _effect_metrics(
                physical["baseline_prediction"][selected],
                physical["intervention_prediction"][selected],
                physical["baseline_target"][selected],
                physical["intervention_target"][selected],
            ),
        }
    return result


def _run_metrics(
    arrays: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Aggregate trajectory and causal-effect errors without mixing runs."""

    result = {}
    paired = arrays["paired"].astype(bool)
    run_ids = arrays["run_id"].astype(int)
    for run_id in sorted(set(run_ids.tolist())):
        selected = run_ids == run_id
        selected_paired = selected & paired
        metrics: dict[str, Any] = {
            "samples": int(selected.sum()),
            "paired_samples": int(selected_paired.sum()),
            "baseline": _state_metrics(
                physical["baseline_prediction"][selected],
                physical["baseline_target"][selected],
            ),
        }
        if selected_paired.any():
            metrics["intervention"] = _state_metrics(
                physical["intervention_prediction"][selected_paired],
                physical["intervention_target"][selected_paired],
            )
            metrics["effect"] = _effect_metrics(
                physical["baseline_prediction"][selected_paired],
                physical["intervention_prediction"][selected_paired],
                physical["baseline_target"][selected_paired],
                physical["intervention_target"][selected_paired],
            )
        result[str(run_id)] = metrics
    return result


def _binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    target, score = target[valid], score[valid]
    if not len(target):
        return {key: None for key in ("accuracy", "precision", "recall", "f1", "auroc", "average_precision")}
    prediction = score >= threshold
    tp = int(np.sum(prediction & target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    tn = int(np.sum(~prediction & ~target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives, negatives = int(target.sum()), int((~target).sum())
    auroc = None
    if positives and negatives:
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        ranks = np.arange(1, len(score) + 1, dtype=float)
        start = 0
        while start < len(score):
            end = start + 1
            while end < len(score) and sorted_score[end] == sorted_score[start]:
                end += 1
            ranks[start:end] = ranks[start:end].mean()
            start = end
        original_ranks = np.empty_like(ranks)
        original_ranks[order] = ranks
        auroc = float(
            (original_ranks[target].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    average_precision = None
    if positives:
        descending = np.argsort(-score, kind="mergesort")
        ordered_target = target[descending]
        cumulative = np.cumsum(ordered_target)
        positive_positions = np.flatnonzero(ordered_target)
        average_precision = float(
            np.mean(cumulative[positive_positions] / (positive_positions + 1))
        )
    return {
        "accuracy": (tp + tn) / len(target),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "average_precision": average_precision,
    }


def _edge_vector(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def _edge_matrix(vector: np.ndarray, nodes: int) -> np.ndarray:
    matrix = np.zeros((nodes, nodes), dtype=float)
    matrix[~np.eye(nodes, dtype=bool)] = vector
    return matrix


def _graph_ground_truth(
    log_dir: Path,
    arrays: dict[str, np.ndarray],
    drone_names: tuple[str, ...],
    threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    predicted_probability = arrays["existence_probability"]
    predicted_strength = arrays["strength"]
    predicted_effective = arrays["effective_edge_mean"]
    truth = {name: [] for name in ("structural", "active", "target_delta_norm", "valid")}
    rows = []
    cached: dict[int, dict[str, np.ndarray]] = {}
    pairs = [
        (sender, receiver)
        for receiver in drone_names
        for sender in drone_names
        if sender != receiver
    ]
    usable_indices = []
    for sample_index, (run_value, snapshot) in enumerate(
        zip(arrays["run_id"], arrays["snapshot_time"])
    ):
        run_id = int(run_value)
        path = log_dir / f"run_{run_id}_ground_truth_matrices.npz"
        if not path.exists():
            continue
        if run_id not in cached:
            with np.load(path, allow_pickle=False) as archive:
                names = [str(value) for value in archive["drone_names"].tolist()]
                order = [names.index(name) for name in drone_names]
                cached[run_id] = {
                    "times": np.asarray(archive["times"], dtype=float),
                    **{
                        name: np.asarray(archive[name], dtype=float)[:, order][:, :, order]
                        for name in ("structural", "active", "target_delta_norm", "counterfactual_valid")
                    },
                }
        archive = cached[run_id]
        time_index = int(np.argmin(np.abs(archive["times"] - float(snapshot))))
        vectors = {
            "structural": _edge_vector(archive["structural"][time_index]),
            "active": _edge_vector(archive["active"][time_index]),
            "target_delta_norm": _edge_vector(archive["target_delta_norm"][time_index]),
            "valid": _edge_vector(archive["counterfactual_valid"][time_index]),
        }
        for name, vector in vectors.items():
            truth[name].append(vector)
        usable_indices.append(sample_index)
        for edge_index, (sender, receiver) in enumerate(pairs):
            rows.append(
                {
                    "run_id": run_id,
                    "snapshot_time_s": float(snapshot),
                    "sender": sender,
                    "receiver": receiver,
                    "p_edge": float(predicted_probability[sample_index, edge_index]),
                    "strength": float(predicted_strength[sample_index, edge_index]),
                    "effective_edge_mean": float(predicted_effective[sample_index, edge_index]),
                    "structural_target": float(vectors["structural"][edge_index]),
                    "active_target": float(vectors["active"][edge_index]),
                    "target_delta_norm": float(vectors["target_delta_norm"][edge_index]),
                    "counterfactual_valid": float(vectors["valid"][edge_index]),
                }
            )
    if not usable_indices:
        return {"available": False, "samples": 0}, rows, {}
    truth_arrays = {name: np.stack(parts) for name, parts in truth.items()}
    indices = np.asarray(usable_indices, dtype=int)
    p_edge = predicted_probability[indices]
    effective = predicted_effective[indices]
    valid = truth_arrays["valid"] > 0.5
    continuous_valid = valid & np.isfinite(truth_arrays["target_delta_norm"])
    correlation = None
    if continuous_valid.sum() > 1:
        x = effective[continuous_valid]
        y = truth_arrays["target_delta_norm"][continuous_valid]
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    report = {
        "available": True,
        "samples": len(usable_indices),
        "structural_existence": _binary_metrics(
            truth_arrays["structural"].reshape(-1) > 0.5,
            p_edge.reshape(-1),
            threshold,
        ),
        "active_existence": _binary_metrics(
            truth_arrays["active"].reshape(-1) > 0.5,
            p_edge.reshape(-1),
            threshold,
        ),
        "decision_threshold": float(threshold),
        "effective_vs_target_delta_pearson": correlation,
    }
    means = {
        "p_edge": p_edge.mean(axis=0),
        "strength": predicted_strength[indices].mean(axis=0),
        "effective": effective.mean(axis=0),
        "structural": truth_arrays["structural"].mean(axis=0),
        "active": truth_arrays["active"].mean(axis=0),
    }
    return report, rows, means


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_horizon(path: Path, times: np.ndarray, horizon: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for prefix, label in (("baseline", "Baseline"), ("intervention", "Intervention"), ("effect", "Causal effect")):
        axes[0].plot(times, horizon[f"{prefix}_position"], label=label, linewidth=2)
        axes[1].plot(times, horizon[f"{prefix}_velocity"], label=label, linewidth=2)
    axes[0].plot(
        times,
        horizon["zero_effect_position"],
        ":",
        label="Zero-effect baseline",
        linewidth=2,
    )
    axes[1].plot(
        times,
        horizon["zero_effect_velocity"],
        ":",
        label="Zero-effect baseline",
        linewidth=2,
    )
    axes[0].set(title="Position error over rollout", xlabel="Horizon [s]", ylabel="Mean Euclidean error [m]")
    axes[1].set(title="Velocity error over rollout", xlabel="Horizon [s]", ylabel="Mean Euclidean error [m/s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_formations(path: Path, groups: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(groups)
    x = np.arange(len(labels))
    width = 0.25
    baseline = [groups[name]["baseline"]["position_ade_m"] for name in labels]
    intervention = [groups[name].get("intervention", {}).get("position_ade_m", np.nan) for name in labels]
    effect = [groups[name].get("effect", {}).get("position_ade_m", np.nan) for name in labels]
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.8), constrained_layout=True)
    axis.bar(x - width, baseline, width, label="Baseline")
    axis.bar(x, intervention, width, label="Intervention")
    axis.bar(x + width, effect, width, label="Causal effect")
    axis.set(title="Held-out error by formation", xlabel="Formation", ylabel="Position ADE [m]", xticks=x, xticklabels=labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_targets(path: Path, groups: dict[str, dict[str, Any]]) -> None:
    if not groups:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(groups)
    values = [groups[name]["effect"]["position_ade_m"] for name in labels]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.5), constrained_layout=True)
    bars = axis.bar(labels, values)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(
        title="Causal-effect error by intervened drone",
        xlabel="Intervention target",
        ylabel="Effect position ADE [m]",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_graph(path: Path, means: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    if not means:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("p_edge", "strength", "effective", "structural", "active")
    titles = ("P(edge)", "Strength", "P(edge) × strength", "Structural target", "Active target")
    figure, axes = plt.subplots(1, len(keys), figsize=(18, 4), constrained_layout=True)
    for axis, key, title in zip(axes, keys, titles):
        matrix = _edge_matrix(means[key], len(names))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set(xlabel="Sender", ylabel="Receiver", title=title)
        for receiver in range(len(names)):
            for sender in range(len(names)):
                axis.text(sender, receiver, f"{matrix[receiver, sender]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[receiver, sender] < 0.45 else "black")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Mean score")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_graph_evolution(
    path: Path,
    times: np.ndarray,
    values: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    run_id: int,
) -> None:
    """Plot one row per directed edge and one column per inferred snapshot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("p_edge", "effective", "structural", "active")
    titles = ("P(edge)", "P(edge) × strength", "Structural target", "Active target")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    edge_labels = [f"{sender}→{receiver}" for sender, receiver in pairs]
    for axis, key, title in zip(axes.ravel(), keys, titles):
        image = axis.imshow(
            values[key].T,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_yticks(range(len(edge_labels)), edge_labels, fontsize=7)
        if len(times) <= 8:
            tick_indices = np.arange(len(times))
        else:
            tick_indices = np.unique(
                np.linspace(0, len(times) - 1, 8, dtype=int)
            )
        axis.set_xticks(tick_indices, [f"{times[index]:.1f}" for index in tick_indices])
        axis.set(title=title, xlabel="Snapshot time [s]", ylabel="Directed edge")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="Score")
    figure.suptitle(f"Run {run_id}: inferred graph evolution across snapshots")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_run_graph_artifacts(
    output_dir: Path,
    graph_rows: list[dict[str, Any]],
    names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int]:
    """Save snapshot tensors, run means and one evolution figure per run."""

    if not graph_rows:
        return [], 0
    pair_order = [
        (sender, receiver)
        for receiver in names
        for sender in names
        if sender != receiver
    ]
    pair_index = {pair: index for index, pair in enumerate(pair_order)}
    grouped: dict[int, dict[float, list[dict[str, Any]]]] = {}
    for row in graph_rows:
        grouped.setdefault(int(row["run_id"]), {}).setdefault(
            float(row["snapshot_time_s"]), []
        ).append(row)

    mean_dir = output_dir / "run_mean_graphs"
    evolution_dir = output_dir / "run_graph_evolution"
    mean_dir.mkdir(parents=True, exist_ok=True)
    evolution_dir.mkdir(parents=True, exist_ok=True)
    for stale in mean_dir.glob("run_*_mean_graphs.png"):
        stale.unlink()
    for stale in evolution_dir.glob("run_*_graph_evolution.png"):
        stale.unlink()
    keys = ("p_edge", "strength", "effective_edge_mean", "structural_target", "active_target")
    compact_names = {
        "p_edge": "p_edge",
        "strength": "strength",
        "effective_edge_mean": "effective",
        "structural_target": "structural",
        "active_target": "active",
    }
    mean_rows = []
    snapshot_run_ids = []
    snapshot_times = []
    snapshot_values = {name: [] for name in compact_names.values()}
    for run_id, by_time in sorted(grouped.items()):
        times = np.asarray(sorted(by_time), dtype=float)
        values = {
            compact_names[key]: np.zeros((len(times), len(pair_order)), dtype=np.float32)
            for key in keys
        }
        for time_index, timestamp in enumerate(times):
            for row in by_time[float(timestamp)]:
                edge_index = pair_index[(str(row["sender"]), str(row["receiver"]))]
                for source, destination in compact_names.items():
                    values[destination][time_index, edge_index] = float(row[source])
        means = {name: matrix.mean(axis=0) for name, matrix in values.items()}
        _plot_graph(
            mean_dir / f"run_{run_id}_mean_graphs.png",
            means,
            names,
        )
        _plot_graph_evolution(
            evolution_dir / f"run_{run_id}_graph_evolution.png",
            times,
            values,
            pair_order,
            run_id,
        )
        for edge_index, (sender, receiver) in enumerate(pair_order):
            mean_rows.append(
                {
                    "run_id": run_id,
                    "sender": sender,
                    "receiver": receiver,
                    "snapshots": len(times),
                    **{
                        f"mean_{name}": float(vector[edge_index])
                        for name, vector in means.items()
                    },
                }
            )
        snapshot_run_ids.extend([run_id] * len(times))
        snapshot_times.extend(times.tolist())
        for name, matrix in values.items():
            snapshot_values[name].append(matrix)

    np.savez_compressed(
        output_dir / "snapshot_graph_matrices.npz",
        run_ids=np.asarray(snapshot_run_ids, dtype=np.int64),
        snapshot_times=np.asarray(snapshot_times, dtype=np.float64),
        edge_senders=np.asarray([pair[0] for pair in pair_order]),
        edge_receivers=np.asarray([pair[1] for pair in pair_order]),
        **{
            name: np.concatenate(parts, axis=0)
            for name, parts in snapshot_values.items()
        },
    )
    return mean_rows, len(grouped)


def _plot_rollout_graph_examples(
    path: Path,
    arrays: dict[str, np.ndarray],
    formations: list[str],
    names: tuple[str, ...],
    horizon_times: np.ndarray,
) -> None:
    """Show branch-specific graph evolution inside representative rollouts."""

    paired_indices = np.flatnonzero(arrays["paired"])
    if not len(paired_indices):
        return
    selected = []
    seen = set()
    for index in paired_indices:
        if formations[index] not in seen:
            selected.append(int(index))
            seen.add(formations[index])
    pairs = [
        (sender, receiver)
        for receiver in names
        for sender in names
        if sender != receiver
    ]
    edge_labels = [f"{sender}→{receiver}" for sender, receiver in pairs]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(15, 3.8 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, index in enumerate(selected):
        baseline = arrays["baseline_graph_probability"][index].T
        intervention = arrays["intervention_graph_probability"][index].T
        difference = (
            arrays["intervention_graph_effective_mean"][index]
            - arrays["baseline_graph_effective_mean"][index]
        ).T
        maximum = max(float(np.abs(difference).max()), 1e-6)
        panels = (
            (baseline, "Baseline P(edge)", "viridis", None),
            (intervention, "Intervention P(edge)", "viridis", None),
            (
                difference,
                "Intervention − baseline effective graph",
                "coolwarm",
                TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
            ),
        )
        for column, (matrix, title, cmap, norm) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                vmin=0.0 if norm is None else None,
                vmax=1.0 if norm is None else None,
                cmap=cmap,
                norm=norm,
            )
            axis.set_yticks(range(len(edge_labels)), edge_labels, fontsize=7)
            ticks = np.unique(
                np.linspace(0, len(horizon_times) - 1, 6, dtype=int)
            )
            axis.set_xticks(
                ticks, [f"{horizon_times[tick]:.1f}" for tick in ticks]
            )
            axis.set(
                title=title,
                xlabel="Rollout horizon [s]",
                ylabel="Directed edge",
            )
            figure.colorbar(image, ax=axis, shrink=0.75)
        axes[row, 0].text(
            -0.22,
            1.12,
            f"Run {int(arrays['run_id'][index])} · {formations[index]}",
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
    figure.suptitle("Dynamic graph prior inside paired 8 s rollouts")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_rollout_graph_artifacts(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    formations: list[str],
    names: tuple[str, ...],
    horizon_times: np.ndarray,
) -> dict[str, float]:
    pairs = [
        (sender, receiver)
        for receiver in names
        for sender in names
        if sender != receiver
    ]
    np.savez_compressed(
        output_dir / "rollout_graph_matrices.npz",
        run_ids=arrays["run_id"].astype(np.int64),
        snapshot_times=arrays["snapshot_time"].astype(np.float64),
        rollout_horizon_times=horizon_times.astype(np.float64),
        edge_senders=np.asarray([pair[0] for pair in pairs]),
        edge_receivers=np.asarray([pair[1] for pair in pairs]),
        baseline_probability=arrays["baseline_graph_probability"],
        intervention_probability=arrays["intervention_graph_probability"],
        baseline_strength=arrays["baseline_graph_strength"],
        intervention_strength=arrays["intervention_graph_strength"],
        baseline_effective=arrays["baseline_graph_effective_mean"],
        intervention_effective=arrays["intervention_graph_effective_mean"],
    )
    _plot_rollout_graph_examples(
        output_dir / "rollout_graph_evolution_examples.png",
        arrays,
        formations,
        names,
        horizon_times,
    )
    paired = arrays["paired"].astype(bool)
    paired_difference = np.abs(
        arrays["intervention_graph_effective_mean"][paired]
        - arrays["baseline_graph_effective_mean"][paired]
    )
    nominal_difference = np.abs(
        arrays["intervention_graph_effective_mean"][~paired]
        - arrays["baseline_graph_effective_mean"][~paired]
    )
    return {
        "paired_mean_absolute_graph_divergence": float(paired_difference.mean()),
        "paired_final_absolute_graph_divergence": float(
            paired_difference[:, -1].mean()
        ),
        "zero_intervention_graph_divergence": float(
            nominal_difference.mean() if nominal_difference.size else 0.0
        ),
    }


def _plot_examples(
    path: Path,
    arrays: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
    formations: list[str],
    targets: list[str],
    names: tuple[str, ...],
    times: np.ndarray,
) -> None:
    paired_indices = np.flatnonzero(arrays["paired"])
    if not len(paired_indices):
        return
    selected = []
    seen = set()
    for index in paired_indices:
        if formations[index] not in seen:
            selected.append(int(index))
            seen.add(formations[index])
        if len(selected) == 4:
            break
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(selected), 2, figsize=(12, 3.8 * len(selected)), squeeze=False, constrained_layout=True)
    for row, index in enumerate(selected):
        target_name = targets[index]
        node = names.index(target_name) if target_name in names else 0
        start = physical["history"][index, node, -1]
        baseline_truth = physical["baseline_target"][index, node]
        baseline_prediction = physical["baseline_prediction"][index, node]
        intervention_truth = physical["intervention_target"][index, node]
        intervention_prediction = physical["intervention_prediction"][index, node]
        trajectory_axis, effect_axis = axes[row]
        trajectory_axis.plot(baseline_truth[:, 0], baseline_truth[:, 1], label="Baseline truth", linewidth=2)
        trajectory_axis.plot(baseline_prediction[:, 0], baseline_prediction[:, 1], "--", label="Baseline prediction")
        trajectory_axis.plot(intervention_truth[:, 0], intervention_truth[:, 1], label="Intervention truth", linewidth=2)
        trajectory_axis.plot(intervention_prediction[:, 0], intervention_prediction[:, 1], "--", label="Intervention prediction")
        trajectory_axis.scatter([start[0]], [start[1]], marker="o", color="black", s=25, label="Snapshot")
        trajectory_axis.set(title=f"Run {int(arrays['run_id'][index])} · {formations[index]} · {target_name}", xlabel="x [m]", ylabel="y [m]")
        trajectory_axis.axis("equal")
        trajectory_axis.grid(alpha=0.25)
        trajectory_axis.legend(fontsize=8)
        true_effect = np.linalg.norm(intervention_truth[:, :3] - baseline_truth[:, :3], axis=-1)
        predicted_effect = np.linalg.norm(intervention_prediction[:, :3] - baseline_prediction[:, :3], axis=-1)
        effect_axis.plot(times, true_effect, label="True effect", linewidth=2)
        effect_axis.plot(times, predicted_effect, "--", label="Predicted effect", linewidth=2)
        effect_axis.set(title="Intervention displacement", xlabel="Horizon [s]", ylabel="|Xᶦ − X⁰| [m]")
        effect_axis.grid(alpha=0.25)
        effect_axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_training_history(path: Path, history_path: Path) -> None:
    if not history_path.exists():
        return
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([float(row["epoch"]) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for split in ("train", "val"):
        axes[0].plot(epochs, [float(row[f"{split}_loss"]) for row in rows], label=split)
        axes[1].plot(epochs, [float(row[f"{split}_effect_rmse"]) for row in rows], label=split)
        axes[2].plot(epochs, [float(row[f"{split}_edge_probability"]) for row in rows], label=f"{split} P(edge)")
        axes[2].plot(epochs, [float(row[f"{split}_strength"]) for row in rows], "--", label=f"{split} strength")
        if f"{split}_probability_weighted_strength" in rows[0]:
            axes[2].plot(
                epochs,
                [
                    float(row[f"{split}_probability_weighted_strength"])
                    for row in rows
                ],
                ":",
                label=f"{split} weighted strength",
            )
    axes[0].set(title="Objective", xlabel="Epoch", ylabel="Loss")
    axes[0].set_yscale("log")
    axes[1].set(title="Normalized causal-effect error", xlabel="Epoch", ylabel="Effect RMSE")
    axes[2].set(title="Graph posterior", xlabel="Epoch", ylabel="Mean score")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def calibrate_edge_threshold(
    model,
    loader,
    rel_rec,
    rel_send,
    log_dir: Path,
    drone_names: tuple[str, ...],
    device,
) -> dict[str, Any]:
    """Choose the hard edge threshold on validation data, never on test data."""

    arrays = collect_predictions(
        model, loader, rel_rec, rel_send, device, decoder_graph_mode="soft"
    )
    _, rows, _ = _graph_ground_truth(Path(log_dir), arrays, drone_names)
    if not rows:
        return {
            "available": False,
            "threshold": 0.5,
            "reason": "No validation structural graph target was available.",
        }
    target = np.asarray([row["structural_target"] > 0.5 for row in rows])
    score = np.asarray([row["p_edge"] for row in rows], dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                score[np.isfinite(score)],
                np.asarray([0.5]),
            ]
        )
    )
    best_threshold = 0.5
    best_metrics = _binary_metrics(target, score, best_threshold)
    for threshold in candidates:
        metrics = _binary_metrics(target, score, float(threshold))
        current = (
            float(metrics["f1"] or 0.0),
            float(metrics["accuracy"] or 0.0),
            -abs(float(threshold) - 0.5),
        )
        best = (
            float(best_metrics["f1"] or 0.0),
            float(best_metrics["accuracy"] or 0.0),
            -abs(float(best_threshold) - 0.5),
        )
        if current > best:
            best_threshold = float(threshold)
            best_metrics = metrics
    return {
        "available": True,
        "target": "structural_existence",
        "selection_split": "validation",
        "criterion": "maximum_f1_then_accuracy",
        "threshold": best_threshold,
        "samples": len(rows),
        "metrics": best_metrics,
    }


def _decoder_mode_metrics(
    arrays: dict[str, np.ndarray], data
) -> dict[str, Any]:
    physical = {
        name: data.normalization.inverse(arrays[name])
        for name in (
            "baseline_target",
            "intervention_target",
            "baseline_prediction",
            "intervention_prediction",
        )
    }
    paired = arrays["paired"].astype(bool)
    effect = _effect_metrics(
        physical["baseline_prediction"][paired],
        physical["intervention_prediction"][paired],
        physical["baseline_target"][paired],
        physical["intervention_target"][paired],
    )
    target_nodes = np.abs(arrays["intervention_input"]).sum(axis=(2, 3)) > 1e-8
    propagation_nodes = paired[:, None] & ~target_nodes
    non_target_effect = _effect_metrics(
        physical["baseline_prediction"][propagation_nodes],
        physical["intervention_prediction"][propagation_nodes],
        physical["baseline_target"][propagation_nodes],
        physical["intervention_target"][propagation_nodes],
    )
    return {
        "baseline": _state_metrics(
            physical["baseline_prediction"], physical["baseline_target"]
        ),
        "intervention": _state_metrics(
            physical["intervention_prediction"][paired],
            physical["intervention_target"][paired],
        ),
        "effect": effect,
        "non_target_effect": non_target_effect,
    }


def _plot_decoder_graph_ablation(
    path: Path, mode_metrics: dict[str, dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(mode_metrics)
    all_nodes = [
        mode_metrics[name]["effect"]["position_rmse_m"] for name in labels
    ]
    non_targets = [
        mode_metrics[name]["non_target_effect"]["position_rmse_m"]
        for name in labels
    ]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar(positions - width / 2, all_nodes, width, label="All nodes")
    axis.bar(positions + width / 2, non_targets, width, label="Non-target nodes")
    axis.set_xticks(positions, labels)
    axis.set(
        title="Does the decoder use the inferred graph?",
        xlabel="Decoder graph mode",
        ylabel="Causal-effect position RMSE [m]",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_evaluation_artifacts(
    model,
    loader,
    rel_rec,
    rel_send,
    data,
    config,
    device,
    edge_threshold: float = 0.5,
    primary_graph_mode: str = "soft",
    graph_is_dynamic: bool = True,
) -> dict[str, Any]:
    """Evaluate in simulator units and write CSV/PNG artifacts."""

    if primary_graph_mode not in {"soft", "hard"}:
        raise ValueError("primary_graph_mode must be soft or hard.")

    output_dir = Path(config.output_dir)
    log_dir = Path(config.log_dir)
    # Every mode is evaluated from the exact same checkpoint.  Dynamic v3 uses
    # soft decoding as its primary estimate; the fixed-binary v4 deliberately
    # uses hard decoding as its primary estimate.
    model.set_edge_decision_threshold(float(edge_threshold))
    decoder_mode_arrays = {
        mode: collect_predictions(
            model,
            loader,
            rel_rec,
            rel_send,
            device,
            decoder_graph_mode=mode,
        )
        for mode in ("soft", "hard", "zero", "shuffled")
    }
    arrays = decoder_mode_arrays[primary_graph_mode]
    decoder_mode_metrics = {
        mode: _decoder_mode_metrics(mode_arrays, data)
        for mode, mode_arrays in decoder_mode_arrays.items()
    }
    primary_effect_rmse = decoder_mode_metrics[primary_graph_mode]["effect"][
        "position_rmse_m"
    ]
    zero_effect_rmse = decoder_mode_metrics["zero"]["effect"][
        "position_rmse_m"
    ]
    primary_non_target_rmse = decoder_mode_metrics[primary_graph_mode]["non_target_effect"][
        "position_rmse_m"
    ]
    zero_non_target_rmse = decoder_mode_metrics["zero"]["non_target_effect"][
        "position_rmse_m"
    ]
    graph_utilization = {
        "primary_graph_mode": primary_graph_mode,
        "effect_position_rmse_gap_m": zero_effect_rmse - primary_effect_rmse,
        "non_target_effect_position_rmse_gap_m": (
            zero_non_target_rmse - primary_non_target_rmse
        ),
        "non_target_hard_vs_shuffled_gap_m": (
            decoder_mode_metrics["shuffled"]["non_target_effect"]["position_rmse_m"]
            - decoder_mode_metrics["hard"]["non_target_effect"]["position_rmse_m"]
        ),
        "interpretation": (
            f"Positive gaps mean the inferred {primary_graph_mode} graph improves prediction over "
            "a decoder whose relational messages are forced to zero."
        ),
    }
    initial_probability = arrays["existence_probability"]
    initial_strength = arrays["strength"]
    probability_sum = max(float(initial_probability.sum()), 1e-12)
    posterior_diagnostics = {
        "mean_edge_probability": float(initial_probability.mean()),
        "mean_strength_all_candidate_edges": float(initial_strength.mean()),
        "probability_weighted_conditional_strength": float(
            (initial_probability * initial_strength).sum() / probability_sum
        ),
        "mean_effective_edge": float(
            (initial_probability * initial_strength).mean()
        ),
        "note": (
            "Strength is fixed to one in this experiment; P(edge) is therefore "
            "the only learned relational score."
            if np.allclose(initial_strength, 1.0)
            else "The probability-weighted value is the relevant collapse diagnostic; "
            "strength on an edge whose existence probability is near zero has no "
            "operational effect."
        ),
    }
    physical = {
        name: data.normalization.inverse(arrays[name])
        for name in (
            "history",
            "baseline_target",
            "intervention_target",
            "baseline_prediction",
            "intervention_prediction",
        )
    }
    paired = arrays["paired"].astype(bool)
    metadata = _plan_metadata(log_dir)
    formations = [metadata.get(int(run_id), {}).get("formation", "unknown") for run_id in arrays["run_id"]]
    targets = _target_labels(arrays["intervention_input"], paired, config.drone_names)

    physical_metrics: dict[str, Any] = {
        "baseline": _state_metrics(physical["baseline_prediction"], physical["baseline_target"]),
        "intervention": _state_metrics(physical["intervention_prediction"][paired], physical["intervention_target"][paired]),
        "effect": _effect_metrics(
            physical["baseline_prediction"][paired],
            physical["intervention_prediction"][paired],
            physical["baseline_target"][paired],
            physical["intervention_target"][paired],
        ),
    }
    true_effect = (
        physical["intervention_target"][paired]
        - physical["baseline_target"][paired]
    )
    zero_effect_metrics = _state_metrics(np.zeros_like(true_effect), true_effect)
    physical_metrics["zero_effect_baseline"] = zero_effect_metrics
    zero_rmse = zero_effect_metrics["position_rmse_m"]
    physical_metrics["effect_improvement_vs_zero_percent"] = (
        100.0
        * (zero_rmse - physical_metrics["effect"]["position_rmse_m"])
        / max(zero_rmse, 1e-12)
    )
    baseline_position, baseline_velocity = _horizon_errors(physical["baseline_prediction"], physical["baseline_target"])
    intervention_position, intervention_velocity = _horizon_errors(physical["intervention_prediction"][paired], physical["intervention_target"][paired])
    effect_position, effect_velocity = _horizon_errors(
        physical["intervention_prediction"][paired] - physical["baseline_prediction"][paired],
        physical["intervention_target"][paired] - physical["baseline_target"][paired],
    )
    zero_effect_position, zero_effect_velocity = _horizon_errors(
        np.zeros_like(true_effect), true_effect
    )
    horizon = {
        "baseline_position": baseline_position,
        "baseline_velocity": baseline_velocity,
        "intervention_position": intervention_position,
        "intervention_velocity": intervention_velocity,
        "effect_position": effect_position,
        "effect_velocity": effect_velocity,
        "zero_effect_position": zero_effect_position,
        "zero_effect_velocity": zero_effect_velocity,
    }
    raw_dt = 1.0 / 60.0
    traces = sorted(log_dir.glob("run_*_learning_trace.npz"))[:3]
    observed_dt = []
    for path in traces:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive["times"]) > 1:
                observed_dt.append(float(np.median(np.diff(archive["times"]))))
    if observed_dt:
        raw_dt = float(np.median(observed_dt))
    horizon_times = (np.arange(config.rollout_steps) + 1) * config.downsample * raw_dt

    groups = _group_metrics(arrays, physical, formations)
    target_groups = _target_metrics(arrays, physical, targets)
    run_groups = _run_metrics(arrays, physical)
    graph_report, graph_rows, graph_means = _graph_ground_truth(
        log_dir, arrays, config.drone_names, threshold=edge_threshold
    )
    sample_rows = _sample_rows(arrays, physical, formations, targets)
    horizon_rows = [
        {
            "step": index + 1,
            "horizon_s": float(horizon_times[index]),
            "baseline_position_error_m": float(baseline_position[index]),
            "intervention_position_error_m": float(intervention_position[index]),
            "effect_position_error_m": float(effect_position[index]),
            "baseline_velocity_error_mps": float(baseline_velocity[index]),
            "intervention_velocity_error_mps": float(intervention_velocity[index]),
            "effect_velocity_error_mps": float(effect_velocity[index]),
            "zero_effect_position_error_m": float(zero_effect_position[index]),
            "zero_effect_velocity_error_mps": float(zero_effect_velocity[index]),
        }
        for index in range(config.rollout_steps)
    ]
    formation_rows = []
    for formation, metrics in groups.items():
        row: dict[str, Any] = {"formation": formation, "samples": metrics["samples"], "paired_samples": metrics["paired_samples"]}
        for section in ("baseline", "intervention", "effect"):
            for key, value in metrics.get(section, {}).items():
                row[f"{section}_{key}"] = value
        formation_rows.append(row)
    target_rows = []
    for target, metrics in target_groups.items():
        row = {"intervention_target": target, "paired_samples": metrics["paired_samples"]}
        for section in ("baseline", "intervention", "effect"):
            for key, value in metrics[section].items():
                row[f"{section}_{key}"] = value
        target_rows.append(row)
    run_rows = []
    for run_id, metrics in run_groups.items():
        row = {
            "run_id": int(run_id),
            "samples": metrics["samples"],
            "paired_samples": metrics["paired_samples"],
        }
        for section in ("baseline", "intervention", "effect"):
            for key, value in metrics.get(section, {}).items():
                row[f"{section}_{key}"] = value
        run_rows.append(row)

    _write_csv(output_dir / "test_sample_metrics.csv", sample_rows)
    _write_csv(output_dir / "horizon_metrics.csv", horizon_rows)
    _write_csv(output_dir / "formation_metrics.csv", formation_rows)
    _write_csv(output_dir / "intervention_target_metrics.csv", target_rows)
    _write_csv(output_dir / "run_metrics.csv", run_rows)
    decoder_mode_rows = []
    for mode, metrics in decoder_mode_metrics.items():
        row: dict[str, Any] = {"decoder_graph_mode": mode}
        for section, values in metrics.items():
            for key, value in values.items():
                row[f"{section}_{key}"] = value
        decoder_mode_rows.append(row)
    _write_csv(output_dir / "decoder_graph_mode_metrics.csv", decoder_mode_rows)
    _write_csv(output_dir / "graph_edge_predictions.csv", graph_rows)
    run_graph_rows, graph_run_count = _write_run_graph_artifacts(
        output_dir, graph_rows, config.drone_names
    )
    _write_csv(output_dir / "run_graph_means.csv", run_graph_rows)
    dynamic_graph_report = _write_rollout_graph_artifacts(
        output_dir,
        arrays,
        formations,
        config.drone_names,
        horizon_times,
    )
    _plot_horizon(output_dir / "horizon_errors.png", horizon_times, horizon)
    _plot_formations(output_dir / "formation_comparison.png", groups)
    _plot_targets(output_dir / "intervention_target_comparison.png", target_groups)
    _plot_graph(output_dir / "interaction_graphs.png", graph_means, config.drone_names)
    _plot_examples(output_dir / "counterfactual_examples.png", arrays, physical, formations, targets, config.drone_names, horizon_times)
    _plot_training_history(output_dir / "training_curves.png", output_dir / "training_history.csv")
    _plot_decoder_graph_ablation(
        output_dir / "decoder_graph_ablation.png", decoder_mode_metrics
    )

    report = {
        "units": {"position": "metres", "velocity": "metres_per_second"},
        "physical_metrics": physical_metrics,
        "by_formation": groups,
        "by_intervention_target": target_groups,
        "by_run": run_groups,
        "graph_ground_truth": graph_report,
        "dynamic_graph_rollout": dynamic_graph_report,
        "decoder_graph_modes": decoder_mode_metrics,
        "graph_utilization": graph_utilization,
        "posterior_diagnostics": posterior_diagnostics,
        "edge_decision_threshold": float(edge_threshold),
        "primary_graph_mode": primary_graph_mode,
        "graph_evaluation_levels": {
            "snapshot": (
                "One inferred matrix per history window at its snapshot time; "
                "stored in snapshot_graph_matrices.npz and graph_edge_predictions.csv."
            ),
            "run_mean": (
                "Mean of all snapshot matrices belonging to the same run; "
                "stored in run_graph_means.csv and run_mean_graphs/."
            ),
            "between_snapshot_evolution": (
                "Evolution across the discrete evaluated snapshots of each run; "
                "stored in run_graph_evolution/."
            ),
            "inside_rollout_evolution": (
                "The causal recurrent prior predicts one branch-specific graph at "
                "each rollout step; stored in rollout_graph_matrices.npz."
                if graph_is_dynamic
                else "The snapshot graph is held fixed and repeated at every rollout "
                "step; stored in rollout_graph_matrices.npz."
            ),
            "rollout_interpretation": (
                "Future matrices are causal prior predictions based on autoregressive "
                "states, not posteriors re-encoded from future ground truth."
                if graph_is_dynamic
                else "There is no future graph inference: baseline and intervention "
                "use exactly the same pre-intervention binary topology."
            ),
            "runs_with_graph_ground_truth": graph_run_count,
        },
        "coverage_warnings": [
            f"No paired intervention sample for formation '{name}' in the held-out split."
            for name, metrics in groups.items()
            if metrics["paired_samples"] == 0
        ],
        "ablation_status": {
            "zero_intervention_input": {
                "equivalent_to": "zero-effect baseline",
                "metrics": zero_effect_metrics,
                "effect_position_rmse_improvement_percent": physical_metrics[
                    "effect_improvement_vs_zero_percent"
                ],
            },
            "same_checkpoint_graph_modes": [
                "soft",
                "hard",
                "zero",
                "shuffled",
            ],
            "requires_retraining": [
                "without causal-effect loss",
                "without relation attention",
            ] + (["without strength"] if not np.allclose(initial_strength, 1.0) else []),
        },
        "baseline_comparison_status": {
            "available": False,
            "reason": (
                "Existing dNRI/SRI/RiTINI reports use different windows or targets. "
                "A fair comparison requires evaluating every model on these exact "
                "held-out 8 s parent/fork windows."
            ),
        },
        "artifacts": [
            "training_curves.png",
            "horizon_errors.png",
            "formation_comparison.png",
            "intervention_target_comparison.png",
            "counterfactual_examples.png",
            "decoder_graph_ablation.png",
            "interaction_graphs.png",
            "test_sample_metrics.csv",
            "horizon_metrics.csv",
            "formation_metrics.csv",
            "intervention_target_metrics.csv",
            "graph_edge_predictions.csv",
            "snapshot_graph_matrices.npz",
            "run_metrics.csv",
            "decoder_graph_mode_metrics.csv",
            "edge_threshold_calibration.json",
            "run_graph_means.csv",
            "run_mean_graphs/",
            "run_graph_evolution/",
            "rollout_graph_matrices.npz",
            "rollout_graph_evolution_examples.png",
        ],
    }
    summary = [
        "# Interventional SRI evaluation\n",
        f"Held-out samples: {len(arrays['run_id'])} ({int(paired.sum())} paired forks).\n",
        "| Quantity | Baseline | Intervention | Causal effect |",
        "|---|---:|---:|---:|",
        "| Position RMSE [m] | "
        f"{physical_metrics['baseline']['position_rmse_m']:.4f} | "
        f"{physical_metrics['intervention']['position_rmse_m']:.4f} | "
        f"{physical_metrics['effect']['position_rmse_m']:.4f} |",
        "| Position ADE [m] | "
        f"{physical_metrics['baseline']['position_ade_m']:.4f} | "
        f"{physical_metrics['intervention']['position_ade_m']:.4f} | "
        f"{physical_metrics['effect']['position_ade_m']:.4f} |",
        "| Position FDE [m] | "
        f"{physical_metrics['baseline']['position_fde_m']:.4f} | "
        f"{physical_metrics['intervention']['position_fde_m']:.4f} | "
        f"{physical_metrics['effect']['position_fde_m']:.4f} |",
        "| Velocity RMSE [m/s] | "
        f"{physical_metrics['baseline']['velocity_rmse_mps']:.4f} | "
        f"{physical_metrics['intervention']['velocity_rmse_mps']:.4f} | "
        f"{physical_metrics['effect']['velocity_rmse_mps']:.4f} |\n",
        f"Zero-effect baseline position RMSE: {zero_effect_metrics['position_rmse_m']:.4f} m.",
        "Improvement of the learned causal effect over zero effect: "
        f"{physical_metrics['effect_improvement_vs_zero_percent']:.2f}%.",
        "The causal-effect columns compare `(predicted intervention - predicted baseline)` with the physical fork difference.",
        f"Graph utilization gap (no graph - {primary_graph_mode} graph), non-target effect position RMSE: "
        f"{graph_utilization['non_target_effect_position_rmse_gap_m']:.4f} m. "
        "A positive value means the graph is useful.",
    ]
    (output_dir / "evaluation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    report["artifacts"].append("evaluation_summary.md")
    return report


__all__ = [
    "calibrate_edge_threshold",
    "collect_predictions",
    "create_evaluation_artifacts",
]
