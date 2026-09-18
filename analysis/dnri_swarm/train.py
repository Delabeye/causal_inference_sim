"""Train and evaluate the dNRI swarm baseline.

Commands, from the repository root::

    python -m analysis.dnri_swarm.train train
    python -m analysis.dnri_swarm.train evaluate

The training objective is unsupervised with respect to the graph. Structural
edge labels are loaded only to report graph-recovery metrics.
"""

from __future__ import annotations

import os

# Some local scientific Python installations load libomp through both NumPy and
# PyTorch. Set this before importing either library so the documented project
# command does not abort on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from analysis.nri_original_swarm.data import PreparedData, edge_pairs, prepare_data
from analysis.nri_original_swarm.model import relation_matrices
from analysis.nri_original_swarm.train import (
    binary_metrics,
    make_loaders,
    seed_everything,
    select_device,
)

from .config import CONFIG, Config
from .model import DNRI, dnri_loss


def _jsonable_config(config: Config) -> dict:
    result = asdict(config)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
    return result


def _config_from_dict(raw: dict) -> Config:
    """Restore path/tuple fields using the current config as a type template."""
    defaults = Config()
    values = {}
    for key, value in raw.items():
        if not hasattr(defaults, key):
            continue
        default = getattr(defaults, key)
        if isinstance(default, Path):
            value = Path(value)
        elif isinstance(default, tuple) or (
            default is None and key == "prior" and value is not None
        ):
            value = tuple(value)
        values[key] = value
    return Config(**values)


def build_model(config: Config, device: torch.device) -> DNRI:
    return DNRI(
        node_dims=len(config.feature_columns),
        edge_types=config.edge_types,
        encoder_hidden=config.encoder_hidden,
        encoder_rnn_hidden=config.encoder_rnn_hidden,
        decoder_hidden=config.decoder_hidden,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
        encoder_rnn_layers=config.encoder_rnn_layers,
        encoder_head_hidden=config.encoder_head_hidden,
        encoder_head_layers=config.encoder_head_layers,
        prior_head_hidden=config.prior_head_hidden,
        prior_head_layers=config.prior_head_layers,
        skip_first_edge_type=config.skip_first_edge_type,
        gumbel_temperature=config.gumbel_temperature,
        hard_gumbel_train=config.hard_gumbel_train,
    ).to(device)


def _graph_arrays(
    batch: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_mask = batch["has_truth"].bool()
    if not truth_mask.any():
        empty = np.empty(0)
        return empty, empty, empty
    selected = probabilities.detach().cpu()[truth_mask]
    # Simulator ground truth is static within the baseline dataset. Repeat it
    # across dNRI time steps solely to score each inferred dynamic graph.
    labels = batch["relations"][truth_mask].unsqueeze(-1).expand(
        -1, -1, selected.shape[2]
    )
    return (
        labels.reshape(-1).numpy(),
        selected.argmax(dim=-1).reshape(-1).numpy(),
        selected[..., 1].reshape(-1).numpy(),
    )


def run_epoch(
    model: DNRI,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    graph_source: str = "posterior",
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "kl": 0.0,
        "uniform_kl": 0.0,
        "mse": 0.0,
        "posterior_entropy": 0.0,
        "prior_entropy": 0.0,
        "posterior_switch_rate": 0.0,
        "prior_switch_rate": 0.0,
    }
    sample_count = 0
    graph_values = {
        "posterior": {"labels": [], "predictions": [], "scores": []},
        "prior": {"labels": [], "predictions": [], "scores": []},
    }

    gradient_context = torch.enable_grad() if training else torch.no_grad()
    with gradient_context:
        for batch in loader:
            states = batch["states"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                states,
                rel_rec,
                rel_send,
                graph_source=graph_source,
                teacher_forcing=config.teacher_forcing,
            )
            target = states[:, :, 1:]
            losses = dnri_loss(
                output,
                target,
                variance=config.output_variance,
                num_nodes=len(config.drone_names),
                kl_weight=config.kl_weight,
                uniform_prior_weight=config.uniform_prior_weight,
            )
            if training:
                losses["loss"].backward()
                optimizer.step()

            posterior_logits = output["posterior_logits"]
            prior_logits = output["prior_logits"]
            prediction = output["prediction"]
            assert isinstance(posterior_logits, torch.Tensor)
            assert isinstance(prior_logits, torch.Tensor)
            assert isinstance(prediction, torch.Tensor)
            posterior = torch.softmax(posterior_logits, dim=-1)
            prior = torch.softmax(prior_logits, dim=-1)
            batch_size = states.shape[0]
            sample_count += batch_size
            for key in ("loss", "nll", "kl", "uniform_kl"):
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            totals["mse"] += float(
                F.mse_loss(prediction, target).detach().cpu()
            ) * batch_size
            eps = 1e-12
            totals["posterior_entropy"] += float(
                (-(posterior * torch.log(posterior + eps)).sum(-1).mean()).cpu()
            ) * batch_size
            totals["prior_entropy"] += float(
                (-(prior * torch.log(prior + eps)).sum(-1).mean()).cpu()
            ) * batch_size
            for name, probabilities in (("posterior", posterior), ("prior", prior)):
                types = probabilities.argmax(dim=-1)
                switch_rate = (
                    (types[:, :, 1:] != types[:, :, :-1]).float().mean()
                    if types.shape[2] > 1
                    else types.new_tensor(0.0, dtype=torch.float32)
                )
                totals[f"{name}_switch_rate"] += float(switch_rate.cpu()) * batch_size
                labels, predictions, scores = _graph_arrays(batch, probabilities)
                if labels.size:
                    graph_values[name]["labels"].append(labels)
                    graph_values[name]["predictions"].append(predictions)
                    graph_values[name]["scores"].append(scores)

    count = max(sample_count, 1)
    metrics = {key: value / count for key, value in totals.items()}
    for name, values in graph_values.items():
        labels = np.concatenate(values["labels"]) if values["labels"] else np.empty(0)
        predictions = (
            np.concatenate(values["predictions"])
            if values["predictions"]
            else np.empty(0)
        )
        scores = np.concatenate(values["scores"]) if values["scores"] else np.empty(0)
        for key, value in binary_metrics(labels, predictions, scores).items():
            metrics[f"{name}_{key}"] = value
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
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="negative ELBO", title="dNRI objective")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_mse"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_mse"] for row in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="normalized MSE", title="Reconstruction")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def _save_graph_heatmaps(run_rows: list[dict], config: Config, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = config.drone_names
    run_ids = sorted({row["run_id"] for row in run_rows})
    for run_id in run_ids[: config.max_saved_run_heatmaps]:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for axis, source in zip(axes, ("posterior", "prior")):
            matrix = np.zeros((len(names), len(names)), dtype=float)
            for row in run_rows:
                if row["run_id"] == run_id:
                    matrix[
                        names.index(row["receiver"]), names.index(row["sender"])
                    ] = row[f"mean_{source}_p_edge"]
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
            axis.set_yticks(range(len(names)), names)
            axis.set(xlabel="Sender", ylabel="Receiver", title=source.capitalize())
            for receiver in range(len(names)):
                for sender in range(len(names)):
                    axis.text(
                        sender,
                        receiver,
                        f"{matrix[receiver, sender]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if matrix[receiver, sender] < 0.45 else "black",
                    )
        figure.colorbar(image, ax=axes, label="P(edge)", shrink=0.85)
        figure.suptitle(f"dNRI mean dynamic graph — test run {run_id}")
        figure.subplots_adjust(left=0.07, right=0.91, bottom=0.18, top=0.87, wspace=0.28)
        figure.savefig(output_dir / f"test_run_{run_id}_graphs.png", dpi=160)
        plt.close(figure)


def export_dynamic_graphs(
    model: DNRI,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Export every inferred time-indexed edge plus run-mean adjacency matrices."""
    model.eval()
    pairs = edge_pairs(config.drone_names)
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            encoded = model.encode(states, rel_rec, rel_send)
            posterior = torch.softmax(encoded["posterior_logits"], dim=-1).cpu()
            prior = torch.softmax(encoded["prior_logits"], dim=-1).cpu()
            for item in range(states.shape[0]):
                has_truth = bool(batch["has_truth"][item])
                for edge_index, (sender, receiver) in enumerate(pairs):
                    for step in range(posterior.shape[2]):
                        rows.append(
                            {
                                "run_id": int(batch["run_id"][item]),
                                "window_start": float(batch["start_time"][item]),
                                "transition_index": step,
                                "time": float(batch["times"][item, step]),
                                "sender": sender,
                                "receiver": receiver,
                                "posterior_p_edge": float(
                                    posterior[item, edge_index, step, 1]
                                ),
                                "prior_p_edge": float(prior[item, edge_index, step, 1]),
                                "posterior_type": int(
                                    posterior[item, edge_index, step].argmax()
                                ),
                                "prior_type": int(prior[item, edge_index, step].argmax()),
                                "true_type": int(batch["relations"][item, edge_index])
                                if has_truth
                                else -1,
                            }
                        )
    if not rows:
        return {"dynamic_edge_rows": 0, "run_graph_rows": 0}
    with (output_dir / "test_dynamic_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        key = (row["run_id"], row["sender"], row["receiver"])
        grouped.setdefault(key, []).append(row)
    run_rows = []
    for (run_id, sender, receiver), selected in sorted(grouped.items()):
        truths = [row["true_type"] for row in selected if row["true_type"] >= 0]
        posterior_mean = float(np.mean([row["posterior_p_edge"] for row in selected]))
        prior_mean = float(np.mean([row["prior_p_edge"] for row in selected]))
        run_rows.append(
            {
                "run_id": run_id,
                "sender": sender,
                "receiver": receiver,
                "mean_posterior_p_edge": posterior_mean,
                "mean_prior_p_edge": prior_mean,
                "posterior_type": int(posterior_mean >= 0.5),
                "prior_type": int(prior_mean >= 0.5),
                "true_type": int(round(np.mean(truths))) if truths else -1,
                "dynamic_samples": len(selected),
            }
        )
    with (output_dir / "test_run_graphs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    _save_graph_heatmaps(run_rows, config, output_dir)
    return {"dynamic_edge_rows": len(rows), "run_graph_rows": len(run_rows)}


def evaluate_forecast(
    model: DNRI,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    data: PreparedData,
    config: Config,
    device: torch.device,
) -> dict:
    """Evaluate a genuine future rollout: future states never enter the encoder."""
    model.eval()
    normalized_squared_error = 0.0
    normalized_elements = 0
    physical_squared = np.zeros(len(config.feature_columns), dtype=np.float64)
    physical_count = 0
    per_horizon_squared = np.zeros(
        (config.seq_len - config.burn_in_steps, len(config.feature_columns)),
        dtype=np.float64,
    )
    per_horizon_count = 0
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            history = states[:, :, : config.burn_in_steps]
            target = states[:, :, config.burn_in_steps :]
            output = model.predict_future(
                history,
                target.shape[2],
                rel_rec,
                rel_send,
                hard=True,
                stochastic=False,
            )
            prediction = output["prediction"]
            error = prediction - target
            normalized_squared_error += float(torch.square(error).sum().cpu())
            normalized_elements += error.numel()

            predicted_physical = data.normalization.inverse(prediction.cpu().numpy())
            target_physical = data.normalization.inverse(target.cpu().numpy())
            physical_error = predicted_physical - target_physical
            physical_squared += np.square(physical_error).sum(axis=(0, 1, 2))
            physical_count += int(np.prod(physical_error.shape[:3]))
            per_horizon_squared += np.square(physical_error).sum(axis=(0, 1))
            per_horizon_count += int(np.prod(physical_error.shape[:2]))

    feature_rmse = np.sqrt(physical_squared / max(physical_count, 1))
    result = {
        "burn_in_steps": config.burn_in_steps,
        "horizon_steps": config.seq_len - config.burn_in_steps,
        "normalized_rollout_rmse": float(
            np.sqrt(normalized_squared_error / max(normalized_elements, 1))
        ),
        "feature_rmse": {
            feature: float(feature_rmse[index])
            for index, feature in enumerate(config.feature_columns)
        },
    }
    if len(config.feature_columns) >= 6:
        result["position_rmse_m"] = float(np.sqrt(np.mean(feature_rmse[:3] ** 2)))
        result["velocity_rmse_mps"] = float(np.sqrt(np.mean(feature_rmse[3:6] ** 2)))

    horizon_rmse = np.sqrt(per_horizon_squared / max(per_horizon_count, 1))
    result["per_horizon_feature_rmse"] = [
        {
            "horizon": step + 1,
            **{
                feature: float(horizon_rmse[step, index])
                for index, feature in enumerate(config.feature_columns)
            },
        }
        for step in range(horizon_rmse.shape[0])
    ]
    return result


def evaluate_model(
    model: DNRI,
    data: PreparedData,
    loaders: dict[str, DataLoader],
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> dict:
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device)
    posterior_metrics = run_epoch(
        model, loaders["test"], rel_rec, rel_send, config, device
    )
    prior_reconstruction_metrics = run_epoch(
        model,
        loaders["test"],
        rel_rec,
        rel_send,
        config,
        device,
        graph_source="prior",
    )
    forecast = evaluate_forecast(
        model, loaders["test"], rel_rec, rel_send, data, config, device
    )
    exports = export_dynamic_graphs(
        model, loaders["test"], rel_rec, rel_send, config, device, output_dir
    )
    report = {
        "posterior_reconstruction": posterior_metrics,
        "prior_reconstruction": prior_reconstruction_metrics,
        "causal_prior_forecast": forecast,
        "exports": exports,
        "graph_label_note": (
            "Static structural labels are repeated over time for graph metrics only; "
            "they are never used in the ELBO."
        ),
    }
    (output_dir / "test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train(config: Config = CONFIG) -> dict:
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dNRI windows from {config.log_dir}")
    data = prepare_data(config)
    loaders = make_loaders(data, config)
    print(
        f"Windows: train={len(data.train)}, val={len(data.val)}, "
        f"test={len(data.test)}; device={device}"
    )
    model = build_model(config, device)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )

    manifest = {
        "model": "dynamic_neural_relational_inference",
        "reference": "Graber and Schwing, Dynamic Neural Relational Inference, CVPR 2020",
        "config": _jsonable_config(config),
        "split_run_ids": data.split_run_ids,
        "normalization": data.normalization.as_dict(),
        "skipped_runs": data.skipped_runs,
        "important": "Ground-truth edges are metrics only and are absent from the ELBO.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    checkpoint_path = output_dir / "best_model.pt"
    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            rel_rec,
            rel_send,
            config,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            model, loaders["val"], rel_rec, rel_send, config, device
        )
        scheduler.step()
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:04d} train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_mse={val_metrics['mse']:.6f} "
            f"post_F1={val_metrics['posterior_edge_f1']:.3f} "
            f"prior_F1={val_metrics['prior_edge_f1']:.3f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": _jsonable_config(config),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "split_run_ids": data.split_run_ids,
                },
                checkpoint_path,
            )
    _save_history(history, output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    report = evaluate_model(model, data, loaders, config, device, output_dir)
    report["best_epoch"] = int(checkpoint["epoch"])
    report["best_validation"] = checkpoint["val_metrics"]
    (output_dir / "test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved dNRI checkpoint and evaluation to {output_dir}")
    return report


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
    device_name: str | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _config_from_dict(checkpoint["config"])
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)
    if log_dir is not None:
        config = replace(config, log_dir=log_dir)
    if device_name is not None:
        config = replace(config, device=device_name)
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = select_device(config.device)
    data = prepare_data(config)
    expected_splits = checkpoint.get("split_run_ids")
    if expected_splits is not None and data.split_run_ids != expected_splits:
        raise RuntimeError(
            "The current dataset split differs from the checkpoint split; "
            "evaluation would not be comparable."
        )
    loaders = make_loaders(data, config)
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = evaluate_model(model, data, loaders, config, device, destination)
    print(f"Saved dNRI evaluation to {destination}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "evaluate":
        checkpoint = args.checkpoint or (
            args.output_dir or Path(CONFIG.output_dir)
        ) / "best_model.pt"
        evaluate_checkpoint(
            checkpoint,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            device_name=args.device,
        )
        return
    config = CONFIG
    updates = {}
    for key in ("log_dir", "output_dir", "epochs", "device"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if updates:
        config = replace(config, **updates)
    train(config)


if __name__ == "__main__":
    main()
