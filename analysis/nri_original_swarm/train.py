"""Train the original NRI objective on UAV logs and evaluate hidden edge labels."""

from __future__ import annotations

import os

# The local conda environment loads two OpenMP runtimes through scientific libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import CONFIG, Config
from .data import PreparedData, edge_pairs, prepare_data
from .model import (
    MLPEncoder,
    RNNDecoder,
    categorical_kl,
    gaussian_nll,
    relation_matrices,
)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def binary_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray):
    if labels.size == 0:
        return {
            "edge_accuracy": None,
            "edge_accuracy_permutation": None,
            "edge_precision": None,
            "edge_recall": None,
            "edge_f1": None,
            "edge_average_precision": None,
            "labeled_edges": 0,
        }
    labels = labels.astype(np.int64)
    predictions = predictions.astype(np.int64)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = float((predictions == labels).mean())
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
            (cumulative_positive / ranks * (ordered_labels == 1)).sum() / positive_count
        )
    else:
        average_precision = None
    return {
        "edge_accuracy": accuracy,
        "edge_accuracy_permutation": max(accuracy, 1.0 - accuracy),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "edge_average_precision": average_precision,
        "labeled_edges": int(labels.size),
    }


def run_epoch(
    encoder: MLPEncoder,
    decoder: RNNDecoder,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    optimizer=None,
):
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "kl": 0.0,
        "weighted_kl": 0.0,
        "mse": 0.0,
        "samples": 0,
    }
    all_labels = []
    all_predictions = []
    all_scores = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            states = batch["states"].to(device)
            contexts = batch["contexts"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = encoder(states, rel_rec, rel_send, contexts)
            probabilities = torch.softmax(logits, dim=-1)
            edge_sample = F.gumbel_softmax(
                logits,
                tau=config.gumbel_temperature,
                hard=config.hard_gumbel_train if training else True,
                dim=-1,
            )
            prediction = decoder(
                states,
                edge_sample,
                rel_rec,
                rel_send,
                config.prediction_steps if training else 1,
                contexts=contexts,
            )
            target = states[:, :, 1:, :]
            nll = gaussian_nll(prediction, target, config.output_variance)
            kl = categorical_kl(
                probabilities,
                len(config.drone_names),
                prior=config.prior,
            )
            weighted_kl = config.kl_weight * kl
            loss = nll + weighted_kl
            if training:
                loss.backward()
                optimizer.step()

            batch_size = states.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["nll"] += float(nll.detach().cpu()) * batch_size
            totals["kl"] += float(kl.detach().cpu()) * batch_size
            totals["weighted_kl"] += float(weighted_kl.detach().cpu()) * batch_size
            totals["mse"] += float(F.mse_loss(prediction, target).detach().cpu()) * batch_size
            totals["samples"] += batch_size

            truth_mask = batch["has_truth"].bool()
            if truth_mask.any():
                labels = batch["relations"][truth_mask].reshape(-1).numpy()
                selected_prob = probabilities.detach().cpu()[truth_mask]
                predicted = selected_prob.argmax(dim=-1).reshape(-1).numpy()
                scores = selected_prob[..., 1].reshape(-1).numpy()
                all_labels.append(labels)
                all_predictions.append(predicted)
                all_scores.append(scores)

    count = max(totals.pop("samples"), 1)
    metrics = {key: value / count for key, value in totals.items()}
    labels = np.concatenate(all_labels) if all_labels else np.empty(0)
    predictions = np.concatenate(all_predictions) if all_predictions else np.empty(0)
    scores = np.concatenate(all_scores) if all_scores else np.empty(0)
    metrics.update(binary_metrics(labels, predictions, scores))
    return metrics


def make_loaders(data: PreparedData, config: Config):
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    return {
        "train": DataLoader(data.train, shuffle=True, generator=generator, **common),
        "val": DataLoader(data.val, shuffle=False, **common),
        "test": DataLoader(data.test, shuffle=False, **common),
    }


def save_history(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    columns = list(history[0])
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="NLL + KL", title="NRI objective")
        axes[0].legend()
        axes[1].plot(epochs, [row["train_mse"] for row in history], label="train")
        axes[1].plot(epochs, [row["val_mse"] for row in history], label="validation")
        axes[1].set(xlabel="Epoch", ylabel="MSE", title="Trajectory reconstruction")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass


def export_test_edges(
    encoder: MLPEncoder,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    output_dir: Path,
):
    encoder.eval()
    pairs = edge_pairs(config.drone_names)
    rows = []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            contexts = batch["contexts"].to(device)
            probabilities = torch.softmax(
                encoder(states, rel_rec, rel_send, contexts), dim=-1
            ).cpu()
            predictions = probabilities.argmax(dim=-1)
            for item in range(states.shape[0]):
                has_truth = bool(batch["has_truth"][item])
                for edge_index, (sender, receiver) in enumerate(pairs):
                    rows.append(
                        {
                            "run_id": int(batch["run_id"][item]),
                            "window_start": float(batch["start_time"][item]),
                            "window_end": float(batch["end_time"][item]),
                            "sender": sender,
                            "receiver": receiver,
                            "p_no_edge": float(probabilities[item, edge_index, 0]),
                            "p_edge": float(probabilities[item, edge_index, 1]),
                            "predicted_type": int(predictions[item, edge_index]),
                            "true_type": int(batch["relations"][item, edge_index]) if has_truth else -1,
                        }
                    )
    edge_path = output_dir / "test_window_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run_rows = []
    for run_id in sorted({row["run_id"] for row in rows}):
        run_data = [row for row in rows if row["run_id"] == run_id]
        for sender, receiver in pairs:
            selected = [
                row for row in run_data
                if row["sender"] == sender and row["receiver"] == receiver
            ]
            truths = [row["true_type"] for row in selected if row["true_type"] >= 0]
            run_rows.append(
                {
                    "run_id": run_id,
                    "sender": sender,
                    "receiver": receiver,
                    "mean_p_edge": float(np.mean([row["p_edge"] for row in selected])),
                    "predicted_type": int(np.mean([row["p_edge"] for row in selected]) >= 0.5),
                    "true_type": int(round(np.mean(truths))) if truths else -1,
                    "windows": len(selected),
                }
            )
    with (output_dir / "test_run_graphs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    _save_heatmaps(run_rows, config, output_dir)


def _save_heatmaps(run_rows: list[dict], config: Config, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = config.drone_names
    for run_id in sorted({row["run_id"] for row in run_rows})[: config.max_saved_run_heatmaps]:
        matrix = np.zeros((len(names), len(names)), dtype=float)
        for row in run_rows:
            if row["run_id"] == run_id:
                matrix[names.index(row["receiver"]), names.index(row["sender"])] = row["mean_p_edge"]
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_xlabel("Sender")
        axis.set_ylabel("Receiver")
        axis.set_title(f"NRI inferred graph, test run {run_id}")
        for receiver in range(len(names)):
            for sender in range(len(names)):
                axis.text(sender, receiver, f"{matrix[receiver, sender]:.2f}", ha="center", va="center", color="white" if matrix[receiver, sender] < 0.45 else "black")
        figure.colorbar(image, ax=axis, label="P(edge)")
        figure.tight_layout()
        figure.savefig(output_dir / f"test_run_{run_id}_graph.png", dpi=160)
        plt.close(figure)


def export_test_trajectories(
    encoder: MLPEncoder,
    decoder: RNNDecoder,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    data: PreparedData,
    config: Config,
    device: torch.device,
    output_dir: Path,
):
    """Export one-step and full autoregressive predictions in physical units."""
    encoder.eval()
    decoder.eval()
    rows = []
    windows_seen: dict[int, int] = {}
    plotted_runs: set[int] = set()
    squared_errors = {
        "one_step_position": 0.0,
        "rollout_position": 0.0,
        "one_step_velocity": 0.0,
        "rollout_velocity": 0.0,
    }
    vector_count = 0

    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            contexts = batch["contexts"].to(device)
            logits = encoder(states, rel_rec, rel_send, contexts)
            hard_types = logits.argmax(dim=-1)
            hard_edges = F.one_hot(hard_types, num_classes=config.edge_types).to(states.dtype)
            one_step = decoder(
                states,
                hard_edges,
                rel_rec,
                rel_send,
                prediction_steps=1,
                contexts=contexts,
            )
            rollout = decoder(
                states,
                hard_edges,
                rel_rec,
                rel_send,
                prediction_steps=config.seq_len,
                contexts=contexts,
            )
            one_step = torch.cat([states[:, :, :1], one_step], dim=2)
            rollout = torch.cat([states[:, :, :1], rollout], dim=2)

            for item in range(states.shape[0]):
                run_id = int(batch["run_id"][item])
                window_index = windows_seen.get(run_id, 0)
                windows_seen[run_id] = window_index + 1
                times = batch["times"][item].numpy()
                truth = data.normalization.inverse(states[item].cpu().numpy())
                one = data.normalization.inverse(one_step[item].cpu().numpy())
                rolled = data.normalization.inverse(rollout[item].cpu().numpy())

                truth_eval = truth[:, 1:]
                one_eval = one[:, 1:]
                rollout_eval = rolled[:, 1:]
                squared_errors["one_step_position"] += float(
                    np.square(one_eval[..., :3] - truth_eval[..., :3]).sum()
                )
                squared_errors["rollout_position"] += float(
                    np.square(rollout_eval[..., :3] - truth_eval[..., :3]).sum()
                )
                squared_errors["one_step_velocity"] += float(
                    np.square(one_eval[..., 3:6] - truth_eval[..., 3:6]).sum()
                )
                squared_errors["rollout_velocity"] += float(
                    np.square(rollout_eval[..., 3:6] - truth_eval[..., 3:6]).sum()
                )
                vector_count += truth_eval.shape[0] * truth_eval.shape[1] * 3

                for drone_index, drone_name in enumerate(config.drone_names):
                    for step, timestamp in enumerate(times):
                        row = {
                            "run_id": run_id,
                            "window_index": window_index,
                            "time": float(timestamp),
                            "drone": drone_name,
                        }
                        for prefix, values in (
                            ("true", truth),
                            ("one_step", one),
                            ("rollout", rolled),
                        ):
                            vector = values[drone_index, step]
                            for feature_index, feature_name in enumerate(config.feature_columns):
                                row[f"{prefix}_{feature_name}"] = float(vector[feature_index])
                        rows.append(row)

                should_plot = (
                    window_index < config.trajectory_windows_per_run
                    and (
                        run_id in plotted_runs
                        or len(plotted_runs) < config.max_saved_trajectory_runs
                    )
                )
                if should_plot:
                    plotted_runs.add(run_id)
                    _plot_trajectory_window(
                        run_id,
                        window_index,
                        times,
                        truth,
                        one,
                        rolled,
                        config,
                        output_dir,
                    )

    if rows:
        path = output_dir / "test_trajectories.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    denominator = max(vector_count, 1)
    return {
        "one_step_position_rmse_m": float(
            np.sqrt(squared_errors["one_step_position"] / denominator)
        ),
        "rollout_position_rmse_m": float(
            np.sqrt(squared_errors["rollout_position"] / denominator)
        ),
        "one_step_velocity_rmse_mps": float(
            np.sqrt(squared_errors["one_step_velocity"] / denominator)
        ),
        "rollout_velocity_rmse_mps": float(
            np.sqrt(squared_errors["rollout_velocity"] / denominator)
        ),
        "windows": int(sum(windows_seen.values())),
    }


def _plot_trajectory_window(
    run_id: int,
    window_index: int,
    times: np.ndarray,
    truth: np.ndarray,
    one_step: np.ndarray,
    rollout: np.ndarray,
    config: Config,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    columns = 3
    rows = int(np.ceil(len(config.drone_names) / columns))
    figure = plt.figure(figsize=(5.2 * columns, 4.4 * rows))
    colors = plt.get_cmap("tab10")
    for drone_index, drone_name in enumerate(config.drone_names):
        axis = figure.add_subplot(rows, columns, drone_index + 1, projection="3d")
        axis.plot(
            truth[drone_index, :, 0],
            truth[drone_index, :, 1],
            truth[drone_index, :, 2],
            color="black",
            linewidth=2.0,
            label="Real trajectory",
        )
        axis.plot(
            one_step[drone_index, :, 0],
            one_step[drone_index, :, 1],
            one_step[drone_index, :, 2],
            color=colors(drone_index),
            linestyle=":",
            linewidth=1.5,
            label="One-step prediction",
        )
        axis.plot(
            rollout[drone_index, :, 0],
            rollout[drone_index, :, 1],
            rollout[drone_index, :, 2],
            color=colors(drone_index),
            linestyle="--",
            linewidth=2.0,
            label="Autoregressive rollout",
        )
        axis.scatter(*truth[drone_index, 0, :3], color="green", s=25, label="Start")
        axis.set_title(drone_name)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.legend(fontsize=7)
    figure.suptitle(
        f"Test run {run_id}, window {window_index}, "
        f"t={times[0]:.3f}-{times[-1]:.3f} s"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"test_run_{run_id}_window_{window_index}_trajectories.png",
        dpi=160,
    )
    plt.close(figure)


def train(config: Config = CONFIG) -> dict:
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing logs from {config.log_dir}")
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    print(
        "Windows: "
        f"train={len(data.train)}, val={len(data.val)}, test={len(data.test)}; "
        f"device={device}"
    )

    node_dims = len(config.feature_columns)
    encoder = MLPEncoder(
        config.seq_len,
        node_dims,
        config.encoder_hidden,
        config.edge_types,
        config.encoder_dropout,
        config.factor_graph,
        context_dims=len(config.context_feature_columns),
    ).to(device)
    decoder = RNNDecoder(
        node_dims,
        config.edge_types,
        config.decoder_hidden,
        config.decoder_dropout,
        config.skip_first_edge_type,
        context_dims=len(config.context_feature_columns),
    ).to(device)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )

    manifest = {
        "config": jsonable_config(config),
        "decoder_type": "recurrent_nri_gru_context_v1",
        "split_run_ids": data.split_run_ids,
        "normalization": data.normalization.as_dict(),
        "context_normalization": data.context_normalization.as_dict(),
        "skipped_runs": data.skipped_runs,
        "important": "Ground-truth relations are metrics only; they are absent from the ELBO.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history = []
    best_val = float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            encoder, decoder, loaders["train"], rel_rec, rel_send, config, device, optimizer
        )
        val_metrics = run_epoch(
            encoder, decoder, loaders["val"], rel_rec, rel_send, config, device
        )
        scheduler.step()
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6f} train_mse={train_metrics['mse']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f} "
            f"val_edge_acc={val_metrics['edge_accuracy']}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "decoder_type": "recurrent_nri_gru_context_v1",
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "manifest": manifest,
                },
                checkpoint_path,
            )
        save_history(history, output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    test_metrics = run_epoch(
        encoder, decoder, loaders["test"], rel_rec, rel_send, config, device
    )
    export_test_edges(
        encoder, loaders["test"], rel_rec, rel_send, config, device, output_dir
    )
    trajectory_metrics = export_test_trajectories(
        encoder,
        decoder,
        loaders["test"],
        rel_rec,
        rel_send,
        data,
        config,
        device,
        output_dir,
    )
    report = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "test": test_metrics,
        "test_trajectory_physical_units": trajectory_metrics,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
