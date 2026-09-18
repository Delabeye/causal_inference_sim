"""Train and evaluate the SRI-inspired binary dynamic graph model."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.nri_dynamic_categorical.train import (
    _classification_metrics,
    _matrix,
    _save_history,
    seed_everything,
)
from analysis.nri_ground_truth_continuous.data import PreparedData, prepare_data
from analysis.nri_ground_truth_continuous.train import make_loaders, select_device
from analysis.nri_original_swarm.data import edge_pairs
from analysis.nri_original_swarm.model import relation_matrices

from .config import CONFIG, Config
from .model import SRIBinaryModel, sri_binary_loss


def _jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def build_model(config: Config, device: torch.device) -> SRIBinaryModel:
    return SRIBinaryModel(
        state_dim=len(config.feature_columns),
        context_dim=len(config.context_feature_columns),
        encoder_hidden=config.encoder_hidden,
        attention_heads=config.attention_heads,
        message_hidden=config.message_hidden,
        decoder_hidden=config.decoder_hidden,
        context_hidden=config.context_hidden,
        dropout=config.dropout,
        temperature=config.gumbel_temperature,
        hard_gumbel=config.hard_gumbel,
    ).to(device)


def _target_and_valid(batch: dict, config: Config) -> tuple[np.ndarray, np.ndarray]:
    if config.evaluation_target == "structural":
        target = batch["structural_target"].numpy()
        valid = np.ones_like(target, dtype=bool)
    else:
        target = batch["active_target"].numpy()
        valid = batch["graph_valid"].bool().numpy()
    return target, valid


def run_epoch(
    model: SRIBinaryModel,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "negative_elbo": 0.0,
        "reconstruction_nll": 0.0,
        "temporal_kl": 0.0,
        "sparse_prior_kl": 0.0,
        "graph_smooth": 0.0,
        "strength_l1": 0.0,
        "context_mse": 0.0,
        "samples": 0,
    }
    labels: list[np.ndarray] = []
    prior_scores: list[np.ndarray] = []
    posterior_scores: list[np.ndarray] = []
    effective_scores: list[np.ndarray] = []
    gradient_context = torch.enable_grad() if training else torch.no_grad()
    with gradient_context:
        for batch in loader:
            states = batch["states"].to(device)
            contexts = batch["contexts"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                states,
                rel_rec,
                rel_send,
                contexts,
                reconstruction_steps=config.reconstruction_steps,
            )
            loss, parts = sri_binary_loss(
                prediction=output["prediction"],
                state_target=states[:, :, 1:],
                context_prediction=output["context_prediction"],
                context_target=contexts[:, :, 1:],
                posterior_probabilities=output["posterior_probabilities"],
                prior_probabilities=output["prior_probabilities"],
                posterior_strength=output["posterior_strength"],
                edge_prior=config.edge_prior,
                state_sigma=config.state_sigma,
                beta_kl=config.beta_kl,
                beta_sparse_prior=config.beta_sparse_prior,
                lambda_graph_smooth=config.lambda_graph_smooth,
                lambda_strength_l1=config.lambda_strength_l1,
                lambda_context=config.lambda_context,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch_size = int(states.shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            for key in parts:
                totals[key] += float(parts[key].detach()) * batch_size
            totals["samples"] += batch_size

            target, valid = _target_and_valid(batch, config)
            labels.append(target[valid])
            prior_scores.append(
                output["prior_probabilities"][..., 1].detach().cpu().numpy()[valid]
            )
            posterior_scores.append(
                output["posterior_probabilities"][..., 1]
                .detach()
                .cpu()
                .numpy()[valid]
            )
            effective_scores.append(
                output["soft_effective_edge"].detach().cpu().numpy()[valid]
            )

    count = max(totals.pop("samples"), 1)
    metrics = {key: value / count for key, value in totals.items()}
    # Keep the shared dynamic-NRI curve exporter compatible while reporting the
    # more precise SRI name as well.
    metrics["categorical_kl"] = metrics["temporal_kl"]
    target = np.concatenate(labels) if labels else np.empty(0)
    scores_by_name = {
        "prior": np.concatenate(prior_scores) if prior_scores else np.empty(0),
        "posterior": (
            np.concatenate(posterior_scores) if posterior_scores else np.empty(0)
        ),
        "effective": (
            np.concatenate(effective_scores) if effective_scores else np.empty(0)
        ),
    }
    for name, scores in scores_by_name.items():
        metrics.update(
            {
                f"{name}_{key}": value
                for key, value in _classification_metrics(target, scores).items()
            }
        )
    metrics["mean_posterior_edge_probability"] = float(
        scores_by_name["posterior"].mean()
    ) if scores_by_name["posterior"].size else None
    return metrics


def _plot_sri_graph(
    run_id: int,
    times: np.ndarray,
    probability: np.ndarray,
    effective: np.ndarray,
    target: np.ndarray,
    names: tuple[str, ...],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = (probability.mean(0), effective.mean(0), target.mean(0))
    titles = ("P(edge)", "P(edge) × strength", "Simulator target")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, title, vector in zip(axes, titles, values):
        matrix = _matrix(vector, len(names))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set(xlabel="Sender", ylabel="Receiver", title=title)
        for receiver in range(len(names)):
            for sender in range(len(names)):
                axis.text(
                    sender,
                    receiver,
                    f"{matrix[receiver, sender]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[receiver, sender] < 0.45 else "black",
                )
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Binary relation score")
    figure.suptitle(
        f"SRI binary run {run_id}, t={times.min():.1f}-{times.max():.1f} s"
    )
    figure.savefig(output_dir / f"test_run_{run_id}_sri_binary_graph.png", dpi=170)
    plt.close(figure)


def evaluate_test(
    model: SRIBinaryModel,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    data: PreparedData,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> dict:
    model.eval()
    rows = []
    per_run: dict[int, dict[str, list[np.ndarray]]] = {}
    one_squared = rollout_squared = 0.0
    one_count = rollout_count = 0
    pairs = edge_pairs(config.drone_names)
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            contexts = batch["contexts"].to(device)
            output = model(states, rel_rec, rel_send, contexts)
            forecast = model.forecast_after_context(
                states, contexts, config.observed_steps, rel_rec, rel_send
            )
            truth_physical = data.normalization.inverse(states.cpu().numpy())
            one_physical = data.normalization.inverse(output["prediction"].cpu().numpy())
            rollout_physical = data.normalization.inverse(
                forecast["prediction"].cpu().numpy()
            )
            one_squared += float(
                np.square(one_physical[..., :3] - truth_physical[:, :, 1:, :3]).sum()
            )
            rollout_squared += float(
                np.square(
                    rollout_physical[..., :3]
                    - truth_physical[:, :, config.observed_steps :, :3]
                ).sum()
            )
            one_count += int(one_physical[..., :3].size)
            rollout_count += int(rollout_physical[..., :3].size)

            prior_p = output["prior_probabilities"][..., 1].cpu().numpy()
            posterior_p = output["posterior_probabilities"][..., 1].cpu().numpy()
            prior_strength = output["prior_strength"].cpu().numpy()
            posterior_strength = output["posterior_strength"].cpu().numpy()
            posterior_effective = posterior_p * posterior_strength
            structural = batch["structural_target"].numpy()
            active = batch["active_target"].numpy()
            selected_target, valid = _target_and_valid(batch, config)
            for item, run_tensor in enumerate(batch["run_id"]):
                run_id = int(run_tensor)
                times = batch["times"][item].numpy()[:-1]
                destination = per_run.setdefault(
                    run_id, {"time": [], "p": [], "effective": [], "target": []}
                )
                destination["time"].append(times)
                destination["p"].append(posterior_p[item])
                destination["effective"].append(posterior_effective[item])
                destination["target"].append(selected_target[item])
                for t, timestamp in enumerate(times):
                    for edge_index, (sender, receiver) in enumerate(pairs):
                        rows.append(
                            {
                                "run_id": run_id,
                                "time": float(timestamp),
                                "sender": sender,
                                "receiver": receiver,
                                "prior_p_edge": float(prior_p[item, t, edge_index]),
                                "posterior_p_edge": float(
                                    posterior_p[item, t, edge_index]
                                ),
                                "prior_strength": float(
                                    prior_strength[item, t, edge_index]
                                ),
                                "posterior_strength": float(
                                    posterior_strength[item, t, edge_index]
                                ),
                                "posterior_effective_weight": float(
                                    posterior_effective[item, t, edge_index]
                                ),
                                "predicted_edge": int(
                                    posterior_p[item, t, edge_index] >= 0.5
                                ),
                                "ground_truth_structural": int(
                                    structural[item, t, edge_index]
                                ),
                                "ground_truth_active": int(active[item, t, edge_index]),
                                "ground_truth_valid": int(valid[item, t, edge_index]),
                            }
                        )

    if rows:
        with (output_dir / "test_sri_binary_graph.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for run_id in sorted(per_run)[: config.max_saved_test_runs]:
        values = per_run[run_id]
        times = np.concatenate(values["time"])
        order = np.argsort(times, kind="stable")
        _plot_sri_graph(
            run_id,
            times[order],
            np.concatenate(values["p"])[order],
            np.concatenate(values["effective"])[order],
            np.concatenate(values["target"])[order],
            config.drone_names,
            output_dir,
        )
    return {
        "one_step_position_rmse_m": float(np.sqrt(one_squared / max(one_count, 1))),
        "rollout_position_rmse_m": float(
            np.sqrt(rollout_squared / max(rollout_count, 1))
        ),
        "observed_steps": config.observed_steps,
        "forecast_steps": config.seq_len - config.observed_steps,
    }


def train(config: Config = CONFIG) -> dict:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = select_device(config.device)
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )
    best_loss = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], rel_rec, rel_send, config, device, optimizer
        )
        val_metrics = run_epoch(
            model, loaders["val"], rel_rec, rel_send, config, device, None
        )
        scheduler.step()
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "sri_binary_attention_lstm_v1",
                    "config": _jsonable_config(config),
                    "split_run_ids": data.split_run_ids,
                    "epoch": epoch,
                },
                config.output_dir / "best_model.pt",
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_p_edge={val_metrics['mean_posterior_edge_probability']:.3f} "
                f"val_f1={val_metrics['posterior_f1']:.3f}"
            )
    _save_history(history, config.output_dir)
    checkpoint = torch.load(
        config.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model, loaders["test"], rel_rec, rel_send, config, device, None
    )
    physical = evaluate_test(
        model, loaders["test"], rel_rec, rel_send, data, config, device, config.output_dir
    )
    report = {
        "checkpoint_epoch": best_epoch,
        "test": test_metrics,
        "physical_trajectory": physical,
        "split_run_ids": data.split_run_ids,
        "graph_labels_used_for_training": False,
        "edge_semantics": {"0": "no-edge", "1": "edge"},
        "evaluation_target": config.evaluation_target,
    }
    (config.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (config.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": _jsonable_config(config),
                "split_run_ids": data.split_run_ids,
                "skipped_runs": data.skipped_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def evaluate_saved(config: Config = CONFIG) -> dict:
    config.validate()
    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = select_device(config.device)
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    checkpoint_path = config.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "sri_binary_attention_lstm_v1":
        raise ValueError("The checkpoint is not an SRI binary model.")
    if checkpoint.get("split_run_ids") != data.split_run_ids:
        raise ValueError("Current dataset splits differ from the saved checkpoint.")
    if checkpoint.get("config") != _jsonable_config(config):
        raise ValueError("Current SRI configuration differs from the saved checkpoint.")
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model, loaders["test"], rel_rec, rel_send, config, device, None
    )
    physical = evaluate_test(
        model, loaders["test"], rel_rec, rel_send, data, config, device, config.output_dir
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test": test_metrics,
        "physical_trajectory": physical,
        "graph_labels_used_for_training": False,
        "edge_semantics": {"0": "no-edge", "1": "edge"},
        "evaluation_target": config.evaluation_target,
    }
    (config.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
