"""Train interventional SRI on nominal runs and physical counterfactual forks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.nri_original_swarm.model import relation_matrices

from .config import CONFIG, Config
from .data import PreparedCounterfactualData, prepare_counterfactual_data
from .model import (
    InterventionalLossWeights,
    InterventionalSRIModel,
    interventional_sri_loss,
)
from .reporting import calibrate_edge_threshold, create_evaluation_artifacts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def _jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def build_model(config: Config, device: torch.device) -> InterventionalSRIModel:
    return InterventionalSRIModel(
        state_dim=config.state_dim,
        context_dim=config.context_dim,
        intervention_dim=config.intervention_dim,
        encoder_hidden=config.encoder_hidden,
        attention_heads=config.attention_heads,
        decoder_hidden=config.decoder_hidden,
        message_hidden=config.message_hidden,
        dynamic_graph_hidden=config.dynamic_graph_hidden,
        dynamic_graph_delta_scale=config.dynamic_graph_delta_scale,
        dropout=config.dropout,
        temperature=config.gumbel_temperature,
        hard_gumbel=config.hard_gumbel,
    ).to(device)


def make_loaders(
    data: PreparedCounterfactualData, config: Config
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            getattr(data, split),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
        )
        for split in ("train", "val", "test")
    }


def _loss_weights(config: Config) -> InterventionalLossWeights:
    return InterventionalLossWeights(
        lambda_intervention=config.lambda_intervention,
        lambda_effect=config.lambda_effect,
        beta_kl=config.beta_kl,
        lambda_strength=config.lambda_strength,
        lambda_graph_smoothness=config.lambda_graph_smoothness,
        lambda_graph_necessity=config.lambda_graph_necessity,
        graph_necessity_margin=config.graph_necessity_margin,
        graph_necessity_min_effect=config.graph_necessity_min_effect,
    )


def graph_training_schedule(config: Config, epoch: int) -> tuple[float, float]:
    """Return discrete blend and Gumbel temperature for a one-based epoch."""

    if epoch <= config.soft_graph_warmup_epochs:
        progress = 0.0
    else:
        progress = min(
            1.0,
            (epoch - config.soft_graph_warmup_epochs)
            / config.graph_discretization_epochs,
        )
    # Geometric temperature decay is smooth in log space.
    temperature = config.gumbel_temperature * (
        config.final_gumbel_temperature / config.gumbel_temperature
    ) ** progress
    return float(progress), float(temperature)


def _horizon_weights(config: Config, device: torch.device) -> torch.Tensor:
    weights = torch.pow(
        torch.tensor(config.horizon_decay, device=device),
        torch.arange(config.rollout_steps, device=device),
    )
    return weights / weights.mean()


def run_epoch(
    model: InterventionalSRIModel,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    decoder_graph_mode: str = "scheduled",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "baseline_nll": 0.0,
        "intervention_nll": 0.0,
        "effect_mse": 0.0,
        "existence_kl": 0.0,
        "effective_strength_l1": 0.0,
        "graph_smoothness": 0.0,
        "graph_necessity": 0.0,
        "graph_necessity_full_error": 0.0,
        "graph_necessity_no_graph_error": 0.0,
        "graph_necessity_nodes": 0.0,
        "paired_fraction": 0.0,
        "baseline_squared": 0.0,
        "baseline_values": 0,
        "intervention_squared": 0.0,
        "intervention_values": 0,
        "effect_squared": 0.0,
        "effect_values": 0,
        "edge_probability": 0.0,
        "strength": 0.0,
        "probability_weighted_strength": 0.0,
        "effective_edge": 0.0,
        "samples": 0,
    }
    horizon_weights = _horizon_weights(config, device)
    gradient_context = torch.enable_grad() if training else torch.no_grad()
    with gradient_context:
        for batch in loader:
            history = batch["history_states"].to(device)
            baseline_target = batch["baseline_future"].to(device)
            intervention_target = batch["intervention_future"].to(device)
            intervention_input = batch["intervention_input"].to(device)
            paired = batch["paired"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                history,
                intervention_input,
                rel_rec,
                rel_send,
                decoder_graph_mode=decoder_graph_mode,
            )
            loss, parts = interventional_sri_loss(
                output,
                baseline_target,
                intervention_target,
                paired,
                state_sigma=config.state_sigma,
                edge_prior_probability=config.edge_prior_probability,
                weights=_loss_weights(config),
                horizon_weights=horizon_weights,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

            batch_size = int(history.shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            for name, value in parts.items():
                totals[name] += float(value.detach()) * batch_size
            totals["samples"] += batch_size
            baseline_error = output["baseline_prediction"] - baseline_target
            totals["baseline_squared"] += float(
                baseline_error.detach().square().sum()
            )
            totals["baseline_values"] += baseline_error.numel()
            if bool(paired.any()):
                paired_prediction = output["intervention_prediction"][paired]
                paired_target = intervention_target[paired]
                paired_effect = output["effect_prediction"][paired]
                true_effect = (intervention_target - baseline_target)[paired]
                totals["intervention_squared"] += float(
                    (paired_prediction - paired_target).detach().square().sum()
                )
                totals["intervention_values"] += paired_target.numel()
                totals["effect_squared"] += float(
                    (paired_effect - true_effect).detach().square().sum()
                )
                totals["effect_values"] += true_effect.numel()
            totals["edge_probability"] += float(
                output["baseline_graph_probability"][..., 1].detach().mean()
            ) * batch_size
            totals["strength"] += float(
                output["baseline_graph_strength"].detach().mean()
            ) * batch_size
            probability = output["baseline_graph_probability"][..., 1].detach()
            strength = output["baseline_graph_strength"].detach()
            totals["probability_weighted_strength"] += float(
                (probability * strength).sum() / probability.sum().clamp_min(1e-8)
            ) * batch_size
            totals["effective_edge"] += float(
                (probability * strength).mean()
            ) * batch_size

    samples = max(int(totals.pop("samples")), 1)
    baseline_values = max(int(totals.pop("baseline_values")), 1)
    intervention_values = max(int(totals.pop("intervention_values")), 1)
    effect_values = max(int(totals.pop("effect_values")), 1)
    baseline_squared = totals.pop("baseline_squared")
    intervention_squared = totals.pop("intervention_squared")
    effect_squared = totals.pop("effect_squared")
    metrics = {name: value / samples for name, value in totals.items()}
    metrics.update(
        {
            "baseline_rmse": float(np.sqrt(baseline_squared / baseline_values)),
            "intervention_rmse": float(
                np.sqrt(intervention_squared / intervention_values)
            ),
            "effect_rmse": float(np.sqrt(effect_squared / effect_values)),
        }
    )
    return metrics


def _write_history(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(config: Config = CONFIG) -> dict:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = select_device(config.device)
    data = prepare_counterfactual_data(config)
    loaders = make_loaders(data, config)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        graph_blend, graph_temperature = graph_training_schedule(config, epoch)
        model.set_graph_schedule(
            discrete_blend=graph_blend,
            temperature=graph_temperature,
        )
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
                    "model_type": "interventional_sri_dynamic_graph_v3",
                    "model_state": model.state_dict(),
                    "config": _jsonable_config(config),
                    "normalization": data.normalization.as_dict(),
                    "split_run_ids": data.split_run_ids,
                    "epoch": epoch,
                    "decoder_graph_blend": graph_blend,
                    "gumbel_temperature": graph_temperature,
                },
                config.output_dir / "best_model.pt",
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} "
                f"val_effect_rmse={val_metrics['effect_rmse']:.5f} "
                f"p_edge={val_metrics['edge_probability']:.3f} "
                f"graph_blend={graph_blend:.2f} tau={graph_temperature:.3f}"
            )

    _write_history(history, config.output_dir)
    checkpoint = torch.load(
        config.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.set_graph_schedule(
        discrete_blend=float(checkpoint["decoder_graph_blend"]),
        temperature=float(checkpoint["gumbel_temperature"]),
    )
    test_metrics = run_epoch(
        model,
        loaders["test"],
        rel_rec,
        rel_send,
        config,
        device,
        None,
        decoder_graph_mode="soft",
    )
    edge_calibration = calibrate_edge_threshold(
        model,
        loaders["val"],
        rel_rec,
        rel_send,
        config.log_dir,
        config.drone_names,
        device,
    )
    (config.output_dir / "edge_threshold_calibration.json").write_text(
        json.dumps(edge_calibration, indent=2), encoding="utf-8"
    )
    report = {
        "model_type": checkpoint["model_type"],
        "checkpoint_epoch": best_epoch,
        "test": test_metrics,
        "split_run_ids": data.split_run_ids,
        "samples": {
            split: len(getattr(data, split)) for split in ("train", "val", "test")
        },
        "skipped_artifacts": data.skipped,
        "edge_semantics": {
            "existence": "0=no direct interaction, 1=direct interaction",
            "strength": "continuous conditional message gain in [0,1]",
            "matrix": "row=receiver,column=sender",
        },
        "current_intervention_visible_to_encoder": False,
        "paired_rollouts_share_initial_graph": True,
        "graph_evolves_causally_inside_rollout": True,
        "decoder_graph_curriculum": {
            "best_checkpoint_discrete_blend": float(
                checkpoint["decoder_graph_blend"]
            ),
            "best_checkpoint_temperature": float(
                checkpoint["gumbel_temperature"]
            ),
            "soft_graph_warmup_epochs": config.soft_graph_warmup_epochs,
            "graph_discretization_epochs": config.graph_discretization_epochs,
        },
        "edge_threshold_calibration": edge_calibration,
    }
    report.update(
        create_evaluation_artifacts(
            model,
            loaders["test"],
            rel_rec,
            rel_send,
            data,
            config,
            device,
            edge_threshold=float(edge_calibration["threshold"]),
        )
    )
    (config.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (config.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": _jsonable_config(config),
                "normalization": data.normalization.as_dict(),
                "split_run_ids": data.split_run_ids,
                "skipped_artifacts": data.skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=CONFIG.log_dir)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=CONFIG.device)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    train(
        replace(
            CONFIG,
            log_dir=arguments.log_dir,
            output_dir=arguments.output_dir,
            epochs=arguments.epochs,
            device=arguments.device,
        )
    )
