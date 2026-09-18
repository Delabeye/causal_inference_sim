"""Latent-type and posterior-collapse diagnostics specific to v5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1e-9, 1.0)
    return -(values * np.log(values)).sum(axis=-1)


def write_latent_diagnostics(
    model,
    loader,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    *,
    maximum_saved_samples: int = 128,
) -> dict:
    """Write probabilities that the generic binary reporter intentionally omits."""

    model.eval()
    edge_types = []
    node_states = []
    strengths = []
    baseline_future_types = []
    intervention_future_types = []
    run_ids = []
    snapshot_times = []
    saved = 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["history_states"].to(device),
                batch["intervention_input"].to(device),
                rel_rec,
                rel_send,
                decoder_graph_mode="hard",
            )
            remaining = maximum_saved_samples - saved
            if remaining <= 0:
                break
            count = min(int(batch["history_states"].shape[0]), remaining)
            edge_types.append(output["edge_type_probability"][:count].cpu().numpy())
            node_states.append(output["node_state_probability"][:count].cpu().numpy())
            strengths.append(output["strength"][:count].cpu().numpy())
            baseline_future_types.append(
                output["baseline_edge_type_probability"][:count].cpu().numpy()
            )
            intervention_future_types.append(
                output["intervention_edge_type_probability"][:count].cpu().numpy()
            )
            run_ids.append(batch["run_id"][:count].cpu().numpy())
            snapshot_times.append(batch["snapshot_time"][:count].cpu().numpy())
            saved += count
    if not edge_types:
        report = {"available": False, "saved_samples": 0}
        (output_dir / "v5_latent_diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    edge = np.concatenate(edge_types)
    node = np.concatenate(node_states)
    strength = np.concatenate(strengths)
    baseline_future = np.concatenate(baseline_future_types)
    intervention_future = np.concatenate(intervention_future_types)
    p_edge = 1.0 - edge[..., 0]
    active_mass = edge[..., 1:].sum(axis=(0, 1))
    if active_mass.sum() > 0:
        active_usage = active_mass / active_mass.sum()
    else:
        active_usage = np.zeros_like(active_mass)
    edge_entropy = _entropy(edge)
    node_entropy = _entropy(node)
    report = {
        "available": True,
        "saved_samples": int(edge.shape[0]),
        "edge_type_mean_probability": edge.mean(axis=(0, 1)).tolist(),
        "active_type_usage_conditional": active_usage.tolist(),
        "node_state_mean_probability": node.mean(axis=(0, 1)).tolist(),
        "mean_edge_probability": float(p_edge.mean()),
        "std_edge_probability": float(p_edge.std()),
        "mean_strength": float(strength.mean()),
        "std_strength": float(strength.std()),
        "mean_edge_type_entropy": float(edge_entropy.mean()),
        "normalized_edge_type_entropy": float(
            edge_entropy.mean() / np.log(edge.shape[-1])
        ),
        "mean_node_state_entropy": float(node_entropy.mean()),
        "normalized_node_state_entropy": float(
            node_entropy.mean() / np.log(node.shape[-1])
        ),
        "warning": (
            "A near-zero p_edge standard deviation, one active type near 100%, "
            "or near-zero normalized entropy are posterior-collapse warnings; "
            "they are diagnostics, not automatic failure criteria."
        ),
    }
    np.savez_compressed(
        output_dir / "v5_latent_probabilities.npz",
        edge_type_probability=edge,
        node_state_probability=node,
        strength=strength,
        baseline_future_edge_type_probability=baseline_future,
        intervention_future_edge_type_probability=intervention_future,
        run_id=np.concatenate(run_ids),
        snapshot_time=np.concatenate(snapshot_times),
    )
    (output_dir / "v5_latent_diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


__all__ = ["write_latent_diagnostics"]
