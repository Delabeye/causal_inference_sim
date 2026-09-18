"""Evaluate a saved interventional SRI checkpoint on its held-out split."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from analysis.nri_original_swarm.model import relation_matrices

from .config import CONFIG
from .data import prepare_counterfactual_data
from .reporting import calibrate_edge_threshold, create_evaluation_artifacts
from .train import (
    _jsonable_config,
    build_model,
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
    if checkpoint.get("model_type") != "interventional_sri_dynamic_graph_v3":
        raise ValueError(
            "This evaluator requires an interventional_sri_dynamic_graph_v3 "
            "checkpoint. Earlier hard-decoder checkpoints must be retrained."
        )
    if checkpoint.get("config") != _jsonable_config(config):
        raise ValueError("Current configuration differs from the saved checkpoint.")
    if checkpoint.get("split_run_ids") != data.split_run_ids:
        raise ValueError("Current run-level splits differ from the checkpoint.")
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_graph_schedule(
        discrete_blend=float(checkpoint["decoder_graph_blend"]),
        temperature=float(checkpoint["gumbel_temperature"]),
    )
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device=device)
    metrics = run_epoch(
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
    detailed = create_evaluation_artifacts(
        model,
        loaders["test"],
        rel_rec,
        rel_send,
        data,
        config,
        device,
        edge_threshold=float(edge_calibration["threshold"]),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test": metrics,
        "split_run_ids": data.split_run_ids,
        "edge_threshold_calibration": edge_calibration,
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
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=CONFIG.device)
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
