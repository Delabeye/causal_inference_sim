"""Train the full SRI + paired-intervention model (v5)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from analysis.interventional_sri.data import (
    PreparedCounterfactualData,
    prepare_counterfactual_data,
)
from analysis.interventional_sri.reporting import (
    calibrate_edge_threshold,
    create_evaluation_artifacts,
)
from analysis.nri_original_swarm.model import relation_matrices

from .config import CONFIG, Config
from .diagnostics import write_latent_diagnostics
from .model import (
    InterventionalSRIV5,
    SRIV5LossWeights,
    interventional_sri_v5_loss,
)


MODEL_TYPE = "interventional_sri_full_v5"


class BalancedPairedBatchSampler(Sampler[list[int]]):
    """Build batches containing equal numbers of forked and nominal samples."""

    def __init__(
        self,
        paired: torch.Tensor,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced batches require a positive even batch size.")
        flags = torch.as_tensor(paired, dtype=torch.bool).cpu()
        self.paired_indices = torch.nonzero(flags, as_tuple=False).flatten().tolist()
        self.nominal_indices = torch.nonzero(~flags, as_tuple=False).flatten().tolist()
        if not self.paired_indices or not self.nominal_indices:
            raise ValueError("Training requires both paired and nominal samples.")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return max(1, int(np.ceil(
            2 * max(len(self.paired_indices), len(self.nominal_indices))
            / self.batch_size
        )))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        half = self.batch_size // 2
        paired = self.paired_indices.copy()
        nominal = self.nominal_indices.copy()
        rng.shuffle(paired)
        rng.shuffle(nominal)
        for batch_index in range(len(self)):
            batch = [
                paired[(batch_index * half + offset) % len(paired)]
                for offset in range(half)
            ]
            batch.extend(
                nominal[(batch_index * half + offset) % len(nominal)]
                for offset in range(half)
            )
            rng.shuffle(batch)
            yield batch


def estimate_effect_scale(dataset, floor: float) -> torch.Tensor:
    """Estimate a per-state scale using paired training effects only."""

    paired = torch.as_tensor(dataset.paired, dtype=torch.bool)
    if not bool(paired.any()):
        raise ValueError("At least one paired sample is required for effect scaling.")
    effect = (
        dataset.intervention_future[paired] - dataset.baseline_future[paired]
    ).abs()
    reduce_dims = tuple(range(effect.ndim - 1))
    return effect.mean(dim=reduce_dims).clamp_min(float(floor))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


def _jsonable_config(config: Config) -> dict:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def make_loaders(
    data: PreparedCounterfactualData, config: Config
) -> dict[str, DataLoader]:
    sampler = BalancedPairedBatchSampler(
        data.train.paired,
        config.batch_size,
        config.seed,
    )
    return {
        "train": DataLoader(
            data.train,
            batch_sampler=sampler,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            data.val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            data.test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }


def build_model(config: Config, device: torch.device) -> InterventionalSRIV5:
    return InterventionalSRIV5(
        state_dim=config.state_dim,
        context_dim=config.context_dim,
        intervention_dim=config.intervention_dim,
        num_edge_types=config.num_edge_types,
        num_node_states=config.num_node_states,
        encoder_hidden=config.encoder_hidden,
        attention_heads=config.attention_heads,
        decoder_hidden=config.decoder_hidden,
        message_hidden=config.message_hidden,
        dropout=config.dropout,
        strength_floor=config.strength_floor,
        temperature=config.gumbel_temperature,
        hard_edge_threshold=config.hard_edge_threshold,
        initial_edge_probability=config.initial_edge_probability,
    ).to(device)


def training_schedule(config: Config, epoch: int) -> tuple[float, float]:
    """Return Gumbel temperature and KL warm-up multiplier."""

    progress = min(max((epoch - 1) / max(config.temperature_anneal_epochs - 1, 1), 0.0), 1.0)
    temperature = config.gumbel_temperature * (
        config.final_gumbel_temperature / config.gumbel_temperature
    ) ** progress
    kl_scale = min(epoch / max(config.kl_warmup_epochs, 1), 1.0)
    return float(temperature), float(kl_scale)


def _loss_weights(config: Config) -> SRIV5LossWeights:
    return SRIV5LossWeights(
        lambda_history=config.lambda_history,
        lambda_intervention=config.lambda_intervention,
        lambda_effect=config.lambda_effect,
        lambda_non_target_effect=config.lambda_non_target_effect,
        lambda_graph_necessity=config.lambda_graph_necessity,
        lambda_graph_contrast=config.lambda_graph_contrast,
        beta_edge_kl=config.beta_edge_kl,
        beta_node_kl=config.beta_node_kl,
        graph_margin=config.graph_margin,
        non_target_min_effect=config.non_target_min_effect,
    )


def _horizon_weights(config: Config, device: torch.device) -> torch.Tensor:
    weights = torch.pow(
        torch.tensor(config.horizon_decay, device=device),
        torch.arange(config.rollout_steps, device=device),
    )
    return weights / weights.mean()


def run_epoch(
    model: InterventionalSRIV5,
    loader: DataLoader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    config: Config,
    device: torch.device,
    effect_scale: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    *,
    decoder_graph_mode: str,
    kl_scale: float,
    include_graph_ablations: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    part_names = (
        "history_nll",
        "baseline_nll",
        "intervention_nll",
        "effect_mse",
        "effect_standardized_mse",
        "non_target_effect_relative",
        "graph_necessity",
        "graph_contrast",
        "edge_kl",
        "node_kl",
        "kl_scale",
        "propagation_nodes",
        "paired_fraction",
    )
    scalar_names = (
        "loss",
        *part_names,
        "history_squared",
        "baseline_squared",
        "intervention_squared",
        "effect_squared",
        "edge_probability",
        "hard_edge_density",
        "strength",
        "effective_edge",
        "edge_type_entropy",
        "node_state_entropy",
    )
    # Accumulate metrics on the accelerator and transfer them once per epoch.
    # Calling float(tensor) or .cpu() for every metric synchronizes MPS/CUDA.
    epoch_totals = torch.zeros(
        len(scalar_names) + config.num_edge_types + config.num_node_states,
        device=device,
    )
    samples = 0
    history_values = 0
    baseline_values = 0
    intervention_values = 0
    effect_values = 0
    horizon_weights = _horizon_weights(config, device)
    effect_scale = effect_scale.to(device)
    gradient_context = torch.enable_grad() if training else torch.inference_mode()
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
                include_graph_ablations=include_graph_ablations,
                rollout_diagnostics=False,
            )
            loss, parts = interventional_sri_v5_loss(
                output,
                history,
                baseline_target,
                intervention_target,
                paired,
                state_sigma=config.state_sigma,
                effect_scale=effect_scale,
                weights=_loss_weights(config),
                kl_scale=kl_scale,
                horizon_weights=horizon_weights,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

            batch_size = int(history.shape[0])
            paired_count = int(batch["paired"].sum())
            samples += batch_size

            history_residual = output["history_prediction"] - history[:, :, 1:]
            baseline_residual = output["baseline_prediction"] - baseline_target
            intervention_residual = (
                output["intervention_prediction"][paired]
                - intervention_target[paired]
            )
            true_effect = (intervention_target - baseline_target)[paired]
            effect_residual = output["effect_prediction"][paired] - true_effect
            history_values += history_residual.numel()
            baseline_values += baseline_residual.numel()
            paired_values = paired_count * intervention_target[0].numel()
            intervention_values += paired_values
            effect_values += paired_values

            existence = output["existence_probability"][..., 1].detach()
            strength = output["strength"].detach()
            edge_types = output["edge_type_probability"].detach()
            node_states = output["node_state_probability"].detach()
            edge_entropy = -(
                edge_types.clamp_min(1e-9)
                * edge_types.clamp_min(1e-9).log()
            ).sum(-1).mean()
            node_entropy = -(
                node_states.clamp_min(1e-9)
                * node_states.clamp_min(1e-9).log()
            ).sum(-1).mean()
            scalar_values = (
                loss.detach() * batch_size,
                *(parts[name].detach() * batch_size for name in part_names),
                history_residual.detach().square().sum(),
                baseline_residual.detach().square().sum(),
                intervention_residual.detach().square().sum(),
                effect_residual.detach().square().sum(),
                existence.mean() * batch_size,
                (existence >= model.hard_edge_threshold).float().mean()
                * batch_size,
                strength.mean() * batch_size,
                (existence * strength).mean() * batch_size,
                edge_entropy * batch_size,
                node_entropy * batch_size,
            )
            batch_totals = torch.cat(
                [
                    torch.stack(scalar_values),
                    edge_types.mean(dim=(0, 1)) * batch_size,
                    node_states.mean(dim=(0, 1)) * batch_size,
                ]
            )
            epoch_totals.add_(batch_totals)

    total_values = epoch_totals.cpu().tolist()
    scalar_totals = dict(zip(scalar_names, total_values[: len(scalar_names)]))
    offset = len(scalar_names)
    edge_type_totals = total_values[offset : offset + config.num_edge_types]
    offset += config.num_edge_types
    node_state_totals = total_values[offset : offset + config.num_node_states]
    samples = max(samples, 1)
    history_values = max(history_values, 1)
    baseline_values = max(baseline_values, 1)
    intervention_values = max(intervention_values, 1)
    effect_values = max(effect_values, 1)
    averaged_names = (
        "loss",
        *part_names,
        "edge_probability",
        "hard_edge_density",
        "strength",
        "effective_edge",
        "edge_type_entropy",
        "node_state_entropy",
    )
    metrics = {
        name: scalar_totals[name] / samples
        for name in averaged_names
    }
    metrics.update(
        {
            "history_rmse": float(
                np.sqrt(scalar_totals["history_squared"] / history_values)
            ),
            "baseline_rmse": float(
                np.sqrt(scalar_totals["baseline_squared"] / baseline_values)
            ),
            "intervention_rmse": float(
                np.sqrt(
                    scalar_totals["intervention_squared"] / intervention_values
                )
            ),
            "effect_rmse": float(
                np.sqrt(scalar_totals["effect_squared"] / effect_values)
            ),
        }
    )
    metrics.update(
        {
            f"edge_type_{index}_usage": float(value / samples)
            for index, value in enumerate(edge_type_totals)
        }
    )
    metrics.update(
        {
            f"node_state_{index}_usage": float(value / samples)
            for index, value in enumerate(node_state_totals)
        }
    )
    return metrics


def checkpoint_score(
    hard_metrics: dict[str, float],
    shuffled_metrics: dict[str, float],
    graph_gap_weight: float,
) -> tuple[float, float]:
    advantage = (
        shuffled_metrics["non_target_effect_relative"]
        - hard_metrics["non_target_effect_relative"]
    )
    return float(hard_metrics["loss"] - graph_gap_weight * advantage), float(advantage)


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
    effect_scale = estimate_effect_scale(data.train, config.effect_scale_floor)
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay, gamma=config.lr_gamma
    )

    best_score = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        temperature, kl_scale = training_schedule(config, epoch)
        model.set_temperature(temperature)
        train_metrics = run_epoch(
            model,
            loaders["train"],
            rel_rec,
            rel_send,
            config,
            device,
            effect_scale,
            optimizer,
            decoder_graph_mode="scheduled",
            kl_scale=kl_scale,
            include_graph_ablations=(
                config.lambda_graph_necessity > 0
                or config.lambda_graph_contrast > 0
            ),
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            rel_rec,
            rel_send,
            config,
            device,
            effect_scale,
            None,
            decoder_graph_mode="hard",
            kl_scale=1.0,
            include_graph_ablations=True,
        )
        shuffled_metrics = run_epoch(
            model,
            loaders["val"],
            rel_rec,
            rel_send,
            config,
            device,
            effect_scale,
            None,
            decoder_graph_mode="shuffled",
            kl_scale=1.0,
            include_graph_ablations=False,
        )
        scheduler.step()
        selection_score, graph_advantage = checkpoint_score(
            val_metrics, shuffled_metrics, config.checkpoint_graph_gap_weight
        )
        row = {
            "epoch": epoch,
            "temperature": temperature,
            "kl_scale": kl_scale,
            "checkpoint_score": selection_score,
            "val_graph_advantage": graph_advantage,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update(
            {f"val_shuffled_{key}": value for key, value in shuffled_metrics.items()}
        )
        history.append(row)
        if epoch >= config.minimum_checkpoint_epoch and selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state": model.state_dict(),
                    "config": _jsonable_config(config),
                    "normalization": data.normalization.as_dict(),
                    "effect_scale": effect_scale.tolist(),
                    "split_run_ids": data.split_run_ids,
                    "epoch": epoch,
                    "selection_score": selection_score,
                    "validation_hard": val_metrics,
                    "validation_shuffled": shuffled_metrics,
                    "validation_graph_advantage": graph_advantage,
                    "hard_edge_threshold": config.hard_edge_threshold,
                    "temperature": temperature,
                },
                config.output_dir / "best_model.pt",
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} "
                f"effect={val_metrics['effect_standardized_mse']:.4f} "
                f"graph_adv={graph_advantage:.4f} "
                f"p_edge={val_metrics['edge_probability']:.3f} "
                f"strength={val_metrics['strength']:.3f} "
                f"tau={temperature:.3f} kl={kl_scale:.2f}"
            )

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was eligible for selection.")
    _write_history(history, config.output_dir)
    checkpoint = torch.load(
        config.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.set_edge_decision_threshold(float(checkpoint["hard_edge_threshold"]))
    model.set_temperature(config.final_gumbel_temperature)
    test_metrics = run_epoch(
        model,
        loaders["test"],
        rel_rec,
        rel_send,
        config,
        device,
        effect_scale,
        None,
        decoder_graph_mode="hard",
        kl_scale=1.0,
        include_graph_ablations=True,
    )
    calibration = calibrate_edge_threshold(
        model,
        loaders["val"],
        rel_rec,
        rel_send,
        config.log_dir,
        config.drone_names,
        device,
    )
    calibration["applied"] = False
    calibration["applied_threshold"] = config.hard_edge_threshold
    (config.output_dir / "edge_threshold_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    detailed = create_evaluation_artifacts(
        model,
        loaders["test"],
        rel_rec,
        rel_send,
        data,
        config,
        device,
        edge_threshold=config.hard_edge_threshold,
        primary_graph_mode="hard",
        graph_is_dynamic=True,
    )
    latent = write_latent_diagnostics(
        model, loaders["test"], rel_rec, rel_send, device, config.output_dir
    )
    report = {
        "model_type": MODEL_TYPE,
        "checkpoint_epoch": best_epoch,
        "checkpoint_selection_score": best_score,
        "test": test_metrics,
        "samples": {
            split: len(getattr(data, split)) for split in ("train", "val", "test")
        },
        "split_run_ids": data.split_run_ids,
        "skipped_artifacts": data.skipped,
        "effect_scale_train_normalized": effect_scale.tolist(),
        "edge_semantics": {
            "type_0": "no direct interaction",
            "active_types": "unsupervised interaction mechanisms; existence is their sum",
            "strength": "attention-conditioned bounded relation strength",
            "matrix": "row=receiver,column=sender",
        },
        "node_state_semantics": "unsupervised categorical local-dynamics modes",
        "current_intervention_visible_to_encoder": False,
        "paired_rollouts_share_initial_latents": True,
        "future_graph_prior_is_causal_and_dynamic": True,
        "training_batches": "exactly 50% paired forks and 50% nominal windows",
        "edge_threshold_calibration": calibration,
        "latent_diagnostics": latent,
        **detailed,
    }
    (config.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (config.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": _jsonable_config(config),
                "normalization": data.normalization.as_dict(),
                "effect_scale": effect_scale.tolist(),
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
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default=CONFIG.device
    )
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


__all__ = [
    "MODEL_TYPE",
    "BalancedPairedBatchSampler",
    "build_model",
    "checkpoint_score",
    "estimate_effect_scale",
    "make_loaders",
    "run_epoch",
    "train",
    "training_schedule",
]
