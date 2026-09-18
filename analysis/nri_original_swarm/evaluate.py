"""Evaluate a saved NRI checkpoint without retraining it."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import CONFIG, Config
from .data import prepare_data
from .model import MLPEncoder, RNNDecoder, relation_matrices
from .train import (
    export_test_edges,
    export_test_trajectories,
    make_loaders,
    run_epoch,
    seed_everything,
    select_device,
)


def _config_from_checkpoint(checkpoint: dict, checkpoint_path: Path) -> Config:
    """Restore the training data/model settings and preserve its exact split."""
    saved_manifest = checkpoint.get("manifest", {})
    saved_values = saved_manifest.get("config", {})
    valid_names = {item.name for item in fields(Config)}
    values = {key: value for key, value in saved_values.items() if key in valid_names}

    for key in ("log_dir", "output_dir"):
        if key in values:
            values[key] = Path(values[key])
    for key in ("drone_names", "feature_columns", "context_feature_columns", "prior"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])

    # Checkpoints created before kl_weight was introduced used loss = NLL + KL.
    values.setdefault("kl_weight", 1.0)

    # New visualization options may not exist in older checkpoints.
    values["output_dir"] = checkpoint_path.parent
    values["max_saved_trajectory_runs"] = CONFIG.max_saved_trajectory_runs
    values["trajectory_windows_per_run"] = CONFIG.trajectory_windows_per_run

    split_ids = saved_manifest.get("split_run_ids", {})
    if all(split_ids.get(name) for name in ("train", "val", "test")):
        values["train_run_ids"] = list(split_ids["train"])
        values["val_run_ids"] = list(split_ids["val"])
        values["test_run_ids"] = list(split_ids["test"])
    return Config(**values)


def evaluate(checkpoint_path: Path | None = None) -> dict:
    """Load the best checkpoint and export graph and trajectory test results."""
    checkpoint_path = Path(
        checkpoint_path or (Path(CONFIG.output_dir) / "best_model.pt")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("decoder_type") != "recurrent_nri_gru_context_v1":
        raise ValueError(
            "This checkpoint predates the context-conditioned recurrent model. "
            "Retrain analysis.nri_original_swarm before evaluation."
        )
    config = _config_from_checkpoint(checkpoint, checkpoint_path)
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = select_device(config.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Preparing test split from: {config.log_dir}")
    data = prepare_data(config)
    loaders = make_loaders(data, config)

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
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    rel_rec, rel_send = relation_matrices(len(config.drone_names), device)

    output_dir = checkpoint_path.parent
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
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test": test_metrics,
        "test_trajectory_physical_units": trajectory_metrics,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Trajectory comparison written under: {output_dir}")
    return report


if __name__ == "__main__":
    evaluate()
