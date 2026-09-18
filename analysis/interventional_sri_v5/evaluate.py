"""Evaluate a saved full interventional SRI v5 checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from analysis.interventional_sri.data import prepare_counterfactual_data
from analysis.interventional_sri.reporting import (
    calibrate_edge_threshold,
    create_evaluation_artifacts,
)
from analysis.nri_original_swarm.model import relation_matrices

from .config import CONFIG
from .diagnostics import write_latent_diagnostics
from .train import (
    MODEL_TYPE,
    _jsonable_config,
    build_model,
    estimate_effect_scale,
    make_loaders,
    run_epoch,
    seed_everything,
    select_device,
)


def evaluate(config=CONFIG) -> dict:
    config.validate()
    seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = select_device(config.device)
    data = prepare_counterfactual_data(config)
    loaders = make_loaders(data, config)
    checkpoint_path = config.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"This evaluator requires a {MODEL_TYPE} checkpoint.")
    if checkpoint.get("config") != _jsonable_config(config):
        raise ValueError("Current configuration differs from the saved checkpoint.")
    if checkpoint.get("split_run_ids") != data.split_run_ids:
        raise ValueError("Current run-level splits differ from the checkpoint.")

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint.get("hard_edge_threshold", config.hard_edge_threshold))
    model.set_edge_decision_threshold(threshold)
    model.set_temperature(config.final_gumbel_temperature)
    effect_scale = torch.as_tensor(
        checkpoint.get(
            "effect_scale",
            estimate_effect_scale(data.train, config.effect_scale_floor).tolist(),
        ),
        dtype=torch.float32,
    )
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    metrics = run_epoch(
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
    calibration["applied_threshold"] = threshold
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
        edge_threshold=threshold,
        primary_graph_mode="hard",
        graph_is_dynamic=True,
    )
    latent = write_latent_diagnostics(
        model, loaders["test"], rel_rec, rel_send, device, config.output_dir
    )
    report = {
        "model_type": MODEL_TYPE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_score": float(checkpoint["selection_score"]),
        "test": metrics,
        "effect_scale_train_normalized": effect_scale.tolist(),
        "split_run_ids": data.split_run_ids,
        "edge_threshold_calibration": calibration,
        "latent_diagnostics": latent,
        **detailed,
    }
    (config.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=CONFIG.log_dir)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default=CONFIG.device
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    evaluate(
        replace(
            CONFIG,
            log_dir=arguments.log_dir,
            output_dir=arguments.output_dir,
            device=arguments.device,
        )
    )


__all__ = ["evaluate"]
