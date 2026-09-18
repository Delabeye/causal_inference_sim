"""Train and evaluate the RiTINI swarm baseline."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.nri_ground_truth_continuous.data import PreparedData, prepare_data
from analysis.nri_ground_truth_continuous.train import make_loaders, select_device

from .config import CONFIG, Config
from .model import RiTINISwarm, complete_directed_mask, ritini_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def _edge_vector_to_matrix(values: torch.Tensor, node_count: int) -> torch.Tensor:
    matrix = values.new_zeros(node_count, node_count)
    matrix[complete_directed_mask(node_count, values.device)] = values
    return matrix


def build_prior(data: PreparedData, config: Config) -> torch.Tensor:
    """Build the prior without using validation or test information."""

    nodes = len(config.drone_names)
    mask = complete_directed_mask(nodes)
    if config.prior_mode == "complete":
        prior = torch.zeros(nodes, nodes)
        prior[mask] = 1.0
        return prior
    source = (
        data.train.structural_target
        if config.prior_mode == "structural_mean"
        else data.train.active_target
    )
    mean_edges = source.float().mean(dim=(0, 1))
    return _edge_vector_to_matrix(
        (mean_edges >= config.prior_threshold).float(), nodes
    )


def build_model(config: Config, device: torch.device) -> RiTINISwarm:
    return RiTINISwarm(
        state_dim=len(config.feature_columns),
        node_count=len(config.drone_names),
        history_steps=config.history_steps,
        latent_dim=config.latent_dim,
        attention_heads=config.attention_heads,
        field_hidden=config.field_hidden,
        dropout=config.dropout,
        negative_slope=config.negative_slope,
        ode_method=config.ode_method,
        ode_substeps=config.ode_substeps,
        integration_dt=config.integration_dt,
        use_observed_dt=config.use_observed_dt,
        time_scale=config.time_scale,
    ).to(device)


def _graph_labels(
    batch: dict[str, torch.Tensor], config: Config
) -> tuple[np.ndarray, np.ndarray]:
    start = config.history_steps - 1
    if config.evaluation_target == "structural":
        target = batch["structural_target"][:, start:].numpy()
        valid = np.ones_like(target, dtype=bool)
    elif config.evaluation_target == "active":
        target = batch["active_target"][:, start:].numpy()
        valid = np.ones_like(target, dtype=bool)
    else:
        target = batch["graph_target"][:, start:].numpy()
        valid = batch["graph_valid"][:, start:].bool().numpy()
    return target, valid


def _graph_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    if target.size == 0:
        return {
            "graph_mae": None,
            "graph_correlation": None,
            "edge_precision": None,
            "edge_recall": None,
            "edge_f1": None,
        }
    correlation = None
    if np.std(target) > 1e-12 and np.std(scores) > 1e-12:
        correlation = float(np.corrcoef(target, scores)[0, 1])
    labels = target >= 0.5
    predicted = scores >= threshold
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "graph_mae": float(np.mean(np.abs(scores - target))),
        "graph_correlation": correlation,
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
    }


def run_epoch(
    model: RiTINISwarm,
    loader: DataLoader,
    prior: torch.Tensor,
    config: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | None]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "position_mse": 0.0,
        "velocity_mse": 0.0,
        "prior_loss": 0.0,
        "attention_entropy": 0.0,
        "graph_smooth": 0.0,
        "samples": 0,
    }
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            states = batch["states"].to(device)
            times = batch["times"].to(device) if config.use_observed_dt else None
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(states, times)
            target_states = states[:, :, config.history_steps :]
            loss, parts = ritini_loss(
                output["prediction"],
                target_states,
                output["attention"],
                model.candidate_mask,
                prior,
                config.lambda_velocity,
                config.lambda_prior,
                config.lambda_entropy,
                config.lambda_graph_smooth,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch_size = int(states.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for name, value in parts.items():
                totals[name] += float(value.detach().cpu()) * batch_size
            totals["samples"] += batch_size
            graph_target, graph_valid = _graph_labels(batch, config)
            graph_score = output["edge_attention"].detach().cpu().numpy()
            targets.append(graph_target[graph_valid])
            scores.append(graph_score[graph_valid])

    samples = max(int(totals.pop("samples")), 1)
    metrics: dict[str, float | None] = {
        name: value / samples for name, value in totals.items()
    }
    flat_target = np.concatenate(targets) if targets else np.empty(0)
    flat_scores = np.concatenate(scores) if scores else np.empty(0)
    metrics.update(_graph_metrics(flat_target, flat_scores, config.attention_threshold))
    return metrics


def _save_history(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(
            epochs,
            [row[f"{split}_position_mse"] for row in history],
            label=split,
        )
        axes[2].plot(
            epochs,
            [row[f"{split}_edge_f1"] for row in history],
            label=split,
        )
    axes[0].set(title="Objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Trajectory", xlabel="Epoch", ylabel="Position MSE")
    axes[2].set(title="Graph evaluation only", xlabel="Epoch", ylabel="Edge F1")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(figure)


def export_attention_csv(
    model: RiTINISwarm,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> None:
    path = output_dir / "test_dynamic_attention.csv"
    rows = []
    model.eval()
    pairs = [
        (sender, receiver)
        for receiver in config.drone_names
        for sender in config.drone_names
        if sender != receiver
    ]
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            times_device = batch["times"].to(device) if config.use_observed_dt else None
            output = model(states, times_device)
            scores = output["edge_attention"].cpu().numpy()
            for item in range(states.shape[0]):
                run_id = int(batch["run_id"][item])
                timestamps = batch["times"][item, config.history_steps :].numpy()
                for step, timestamp in enumerate(timestamps):
                    for edge, (sender, receiver) in enumerate(pairs):
                        rows.append(
                            {
                                "run_id": run_id,
                                "time": float(timestamp),
                                "sender": sender,
                                "receiver": receiver,
                                "attention": float(scores[item, step, edge]),
                            }
                        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "time", "sender", "receiver", "attention"],
        )
        writer.writeheader()
        writer.writerows(rows)


def train(config: Config = CONFIG) -> dict[str, float | None]:
    config.validate()
    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = select_device(config.device)
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    prior = build_prior(data, config).to(device)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], prior, config, device, optimizer
        )
        val_metrics = run_epoch(model, loaders["val"], prior, config, device)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "prior_adjacency": prior.cpu(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "config": _jsonable_config(config),
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} "
                f"val_graph_corr={val_metrics['graph_correlation']}"
            )
    _save_history(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, loaders["test"], prior, config, device)
    export_attention_csv(model, loaders["test"], config, device, output_dir)
    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test": test_metrics,
        "split_run_ids": data.split_run_ids,
        "prior_mode": config.prior_mode,
        "prior_used_in_loss": bool(config.lambda_prior > 0),
        "reference": "https://github.com/KrishnaswamyLab/RiTINI",
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": _jsonable_config(config),
                "normalization": {
                    "minimum": data.normalization.minimum.tolist(),
                    "maximum": data.normalization.maximum.tolist(),
                },
                "prior_adjacency": prior.cpu().tolist(),
                "source_license": "RiTINI upstream is non-commercial; this is a clean adaptation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return test_metrics


def evaluate_saved(config: Config = CONFIG) -> dict[str, float | None]:
    config.validate()
    device = select_device(config.device)
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    checkpoint_path = Path(config.output_dir) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    prior = checkpoint["prior_adjacency"].to(device)
    metrics = run_epoch(model, loaders["test"], prior, config, device)
    export_attention_csv(model, loaders["test"], config, device, Path(config.output_dir))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
