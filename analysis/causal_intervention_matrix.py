#!/usr/bin/env python3
"""Estimate an interventional drone-to-drone causal effect matrix.

For a deterministic baseline/perturbed run pair, an intervention on sender
``j`` defines the total trajectory effect on receiver ``i`` at horizon ``h``:

    C[j -> i](h) = ||x_i^pert(t_event_end + h)
                       - x_i^base(t_event_end + h)|| / ||force_j||.

Matrices use ``A[receiver, sender]`` throughout.  Untested entries are NaN,
not zero.  The diagonal is reserved for the directly intervened drone and is
set to zero in exported inter-drone matrices; its response remains available
in the observation table as a manipulation-strength diagnostic.

This is an empirical total-effect matrix under the simulator intervention.
It is separate from an NRI adjacency learned through trajectory prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_COLUMNS = (
    "gt_x",
    "gt_y",
    "gt_z",
    "gt_vx",
    "gt_vy",
    "gt_vz",
)
METRICS = (
    "position_effect_m",
    "velocity_effect_mps",
    "position_effect_per_force",
    "velocity_effect_per_force",
    "position_effect_per_impulse",
    "velocity_effect_per_impulse",
)


@dataclass(frozen=True)
class InterventionPair:
    baseline_run_id: int
    perturbed_run_id: int
    split: str
    sender: str
    leader: str
    event_time: float
    duration: float
    force: tuple[float, float, float]

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.force))

    @property
    def impulse_norm(self) -> float:
        return self.force_norm * self.duration


@dataclass
class RunStates:
    times: np.ndarray
    names: list[str]
    states: np.ndarray  # [time, node, x/y/z/vx/vy/vz]
    crashed: np.ndarray  # [time, node]


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    return float(value)


def read_intervention_pairs(log_dir: Path) -> list[InterventionPair]:
    """Read paired interventions from plan.csv."""

    plan_path = log_dir / "plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing experimental plan: {plan_path}")
    plan = pd.read_csv(plan_path)
    required = {
        "run_id",
        "paired_baseline_run_id",
        "intervention_enabled",
        "intervention_target",
        "intervention_start_time",
        "intervention_duration",
    }
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"plan.csv lacks columns: {sorted(missing)}")

    pairs = []
    for _, row in plan.iterrows():
        if not bool(int(row["intervention_enabled"])):
            continue
        baseline = _optional_float(row["paired_baseline_run_id"])
        event_time = _optional_float(row["intervention_start_time"])
        duration = _optional_float(row["intervention_duration"])
        if baseline is None or event_time is None or duration is None:
            raise ValueError(
                f"Perturbed run {int(row['run_id'])} has incomplete pair metadata."
            )
        force = tuple(
            float(row.get(f"intervention_force_{axis}", 0.0))
            for axis in "xyz"
        )
        pair = InterventionPair(
            baseline_run_id=int(baseline),
            perturbed_run_id=int(row["run_id"]),
            split=str(row.get("split", "unspecified")),
            sender=str(row["intervention_target"]),
            leader=str(row.get("leader", "unknown")),
            event_time=float(event_time),
            duration=float(duration),
            force=force,
        )
        if pair.duration <= 0.0 or pair.force_norm <= 0.0:
            raise ValueError(
                f"Perturbed run {pair.perturbed_run_id} has a zero intervention dose."
            )
        pairs.append(pair)
    if not pairs:
        raise ValueError(f"No paired intervention was found in {plan_path}.")
    return sorted(pairs, key=lambda pair: pair.perturbed_run_id)


def discover_drone_names(log_dir: Path, run_id: int) -> list[str]:
    prefix = f"run_{int(run_id)}_"
    names = []
    for path in log_dir.glob(f"{prefix}drone_*.csv"):
        name = path.stem[len(prefix) :]
        if re.fullmatch(r"drone_\d+", name):
            names.append(name)
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def load_run_states(log_dir: Path, run_id: int, names: list[str]) -> RunStates:
    frames = []
    required = {"time", *STATE_COLUMNS}
    for name in names:
        path = log_dir / f"run_{int(run_id)}_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing UAV trajectory: {path}")
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in required | {"crash_flag"},
        )
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
        if "crash_flag" not in frame:
            frame["crash_flag"] = 0.0
        frame = frame.drop_duplicates("time", keep="first").set_index("time")
        frames.append(frame)

    common_times = sorted(set.intersection(*(set(frame.index) for frame in frames)))
    if len(common_times) < 2:
        raise ValueError(f"Run {run_id} has fewer than two aligned timestamps.")
    times = np.asarray(common_times, dtype=np.float64)
    states = np.stack(
        [frame.loc[times, list(STATE_COLUMNS)].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    crashed = np.stack(
        [frame.loc[times, "crash_flag"].to_numpy(float) > 0.5 for frame in frames],
        axis=1,
    )
    if not np.isfinite(states).all():
        raise ValueError(f"Run {run_id} contains non-finite trajectory states.")
    return RunStates(times=times, names=names, states=states, crashed=crashed)


def validate_deterministic_pair(
    baseline: RunStates,
    perturbed: RunStates,
    event_time: float,
    tolerance: float,
) -> dict[str, Any]:
    """Audit equality before intervention and return maximum discrepancies."""

    if baseline.names != perturbed.names:
        raise ValueError("Paired runs use different drone ordering.")
    if baseline.times.shape != perturbed.times.shape or not np.array_equal(
        baseline.times, perturbed.times
    ):
        raise ValueError("Paired runs do not have identical timestamps.")
    before = baseline.times < float(event_time)
    if not np.any(before):
        raise ValueError("No trajectory sample exists before the intervention.")
    difference = np.abs(perturbed.states[before] - baseline.states[before])
    position_max = float(difference[..., :3].max(initial=0.0))
    velocity_max = float(difference[..., 3:].max(initial=0.0))
    crash_equal = bool(
        np.array_equal(perturbed.crashed[before], baseline.crashed[before])
    )
    return {
        "position_max_abs_difference_m": position_max,
        "velocity_max_abs_difference_mps": velocity_max,
        "crash_flags_identical": crash_equal,
        "passed": bool(
            position_max <= tolerance
            and velocity_max <= tolerance
            and crash_equal
        ),
    }


def _interpolate_state(run: RunStates, timestamp: float) -> np.ndarray:
    if timestamp < run.times[0] or timestamp > run.times[-1]:
        raise ValueError("Requested causal horizon lies outside the run timeline.")
    flat = run.states.reshape(len(run.times), -1)
    values = np.asarray(
        [np.interp(timestamp, run.times, flat[:, column]) for column in range(flat.shape[1])]
    )
    return values.reshape(len(run.names), len(STATE_COLUMNS))


def _first_crash_times(run: RunStates) -> np.ndarray:
    result = np.full(len(run.names), np.inf, dtype=float)
    for node in range(len(run.names)):
        indices = np.flatnonzero(run.crashed[:, node])
        if indices.size:
            result[node] = float(run.times[int(indices[0])])
    return result


def pair_effect_observations(
    pair: InterventionPair,
    baseline: RunStates,
    perturbed: RunStates,
    horizons: Iterable[float],
    *,
    censor_after_crash: bool = True,
) -> list[dict[str, Any]]:
    """Return direct and propagated effects for one paired intervention."""

    if pair.sender not in baseline.names:
        raise ValueError(
            f"Intervention sender {pair.sender} is absent from run {pair.perturbed_run_id}."
        )
    base_crash = _first_crash_times(baseline)
    pert_crash = _first_crash_times(perturbed)
    sender_index = baseline.names.index(pair.sender)
    sender_crash_time = min(
        base_crash[sender_index], pert_crash[sender_index]
    )
    rows = []
    for horizon in horizons:
        sample_time = pair.event_time + pair.duration + float(horizon)
        try:
            base_state = _interpolate_state(baseline, sample_time)
            pert_state = _interpolate_state(perturbed, sample_time)
        except ValueError:
            base_state = pert_state = None
        for receiver_index, receiver in enumerate(baseline.names):
            receiver_crash_time = min(
                base_crash[receiver_index], pert_crash[receiver_index]
            )
            sender_censored = bool(
                censor_after_crash and sample_time >= sender_crash_time
            )
            receiver_censored = bool(
                censor_after_crash
                and sample_time >= receiver_crash_time
            )
            valid = (
                base_state is not None
                and not sender_censored
                and not receiver_censored
            )
            if valid:
                delta = pert_state[receiver_index] - base_state[receiver_index]
                position = float(np.linalg.norm(delta[:3]))
                velocity = float(np.linalg.norm(delta[3:]))
            else:
                position = velocity = math.nan
            rows.append(
                {
                    "baseline_run_id": pair.baseline_run_id,
                    "perturbed_run_id": pair.perturbed_run_id,
                    "split": pair.split,
                    "leader": pair.leader,
                    "sender": pair.sender,
                    "receiver": receiver,
                    "sender_role": "leader" if pair.sender == pair.leader else "follower",
                    "receiver_role": "leader" if receiver == pair.leader else "follower",
                    "is_self_effect": receiver == pair.sender,
                    "event_time_s": pair.event_time,
                    "event_duration_s": pair.duration,
                    "horizon_after_event_s": float(horizon),
                    "sample_time_s": sample_time,
                    "force_x": pair.force[0],
                    "force_y": pair.force[1],
                    "force_z": pair.force[2],
                    "force_norm": pair.force_norm,
                    "impulse_norm": pair.impulse_norm,
                    "valid": valid,
                    "censored_after_crash": sender_censored or receiver_censored,
                    "censored_sender_crash": sender_censored,
                    "censored_receiver_crash": receiver_censored,
                    "position_effect_m": position,
                    "velocity_effect_mps": velocity,
                    "position_effect_per_force": position / pair.force_norm,
                    "velocity_effect_per_force": velocity / pair.force_norm,
                    "position_effect_per_impulse": position / pair.impulse_norm,
                    "velocity_effect_per_impulse": velocity / pair.impulse_norm,
                }
            )
    return rows


def aggregate_causal_matrix(
    observations: pd.DataFrame,
    names: list[str],
    horizon: float,
    metric: str,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one inter-drone matrix, standard deviation and sample count."""

    if metric not in METRICS:
        raise ValueError(f"Unknown causal metric: {metric}")
    matrix = np.full((len(names), len(names)), np.nan, dtype=float)
    standard_deviation = np.full_like(matrix, np.nan)
    count = np.zeros_like(matrix, dtype=np.int64)
    selected = observations[
        np.isclose(observations["horizon_after_event_s"], float(horizon))
        & observations["valid"].astype(bool)
        & ~observations["is_self_effect"].astype(bool)
    ]
    for receiver_index, receiver in enumerate(names):
        for sender_index, sender in enumerate(names):
            if receiver == sender:
                matrix[receiver_index, sender_index] = 0.0
                standard_deviation[receiver_index, sender_index] = 0.0
                continue
            values = selected.loc[
                (selected["receiver"] == receiver) & (selected["sender"] == sender),
                metric,
            ].dropna().to_numpy(float)
            if not values.size:
                continue
            if aggregation == "mean":
                matrix[receiver_index, sender_index] = float(values.mean())
            elif aggregation == "median":
                matrix[receiver_index, sender_index] = float(np.median(values))
            else:
                raise ValueError("aggregation must be 'mean' or 'median'.")
            standard_deviation[receiver_index, sender_index] = float(values.std(ddof=0))
            count[receiver_index, sender_index] = int(values.size)
    return matrix, standard_deviation, count


def _safe_group_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()) or "unspecified"


def _metric_label(metric: str) -> str:
    return {
        "position_effect_m": "Position effect (m)",
        "velocity_effect_mps": "Velocity effect (m/s)",
        "position_effect_per_force": "Position effect / force (m/N)",
        "velocity_effect_per_force": "Velocity effect / force ((m/s)/N)",
        "position_effect_per_impulse": "Position effect / impulse (m/(N.s))",
        "velocity_effect_per_impulse": "Velocity effect / impulse ((m/s)/(N.s))",
    }[metric]


def plot_horizon_matrices(
    matrices: list[np.ndarray],
    horizons: list[float],
    names: list[str],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    columns = min(3, len(horizons))
    rows = int(math.ceil(len(horizons) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    positive = finite[finite > 0.0]
    limit = float(np.percentile(positive, 95.0)) if positive.size else 1.0
    limit = max(limit, 1e-12)
    image = None
    for index, (matrix, horizon) in enumerate(zip(matrices, horizons)):
        axis = axes.flat[index]
        image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=limit)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set(xlabel="Sender intervened", ylabel="Receiver measured")
        axis.set_title(f"h = {horizon:g} s after intervention")
        for receiver in range(len(names)):
            for sender in range(len(names)):
                value = matrix[receiver, sender]
                label = "NA" if not np.isfinite(value) else f"{value:.2g}"
                color = (
                    "white"
                    if np.isfinite(value) and value < 0.45 * limit
                    else "black"
                )
                axis.text(sender, receiver, label, ha="center", va="center", fontsize=7, color=color)
    for index in range(len(horizons), rows * columns):
        axes.flat[index].axis("off")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label=_metric_label(metric))
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _export_group(
    group_name: str,
    observations: pd.DataFrame,
    names: list[str],
    horizons: list[float],
    metric: str,
    aggregation: str,
    output_dir: Path,
) -> dict[str, Any]:
    destination = output_dir / _safe_group_name(group_name)
    destination.mkdir(parents=True, exist_ok=True)
    matrices = []
    deviations = []
    counts = []
    for horizon in horizons:
        matrix, deviation, count = aggregate_causal_matrix(
            observations, names, horizon, metric, aggregation
        )
        matrices.append(matrix)
        deviations.append(deviation)
        counts.append(count)
        suffix = str(horizon).replace(".", "p")
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            destination / f"causal_matrix_h{suffix}s.csv",
            index_label="receiver\\sender",
        )
        pd.DataFrame(count, index=names, columns=names).to_csv(
            destination / f"causal_matrix_h{suffix}s_count.csv",
            index_label="receiver\\sender",
        )
    matrix_array = np.stack(matrices)
    deviation_array = np.stack(deviations)
    count_array = np.stack(counts)
    np.savez_compressed(
        destination / "causal_matrices.npz",
        horizons_s=np.asarray(horizons, dtype=float),
        drone_names=np.asarray(names),
        effect=matrix_array,
        standard_deviation=deviation_array,
        count=count_array,
        metric=np.asarray(metric),
        aggregation=np.asarray(aggregation),
        matrix_convention=np.asarray("row=receiver,column=intervened_sender"),
    )
    plot_path = destination / "causal_matrices_by_horizon.png"
    plot_horizon_matrices(
        matrices,
        horizons,
        names,
        metric,
        f"Interventional causal matrix - {group_name}",
        plot_path,
    )
    return {
        "observations": int(len(observations)),
        "valid_interdrone_observations": int(
            (observations["valid"].astype(bool) & ~observations["is_self_effect"].astype(bool)).sum()
        ),
        "directory": str(destination.resolve()),
        "plot": str(plot_path.resolve()),
    }


def _pair_output_name(pair: InterventionPair) -> str:
    return (
        f"pair_base_{pair.baseline_run_id:04d}_"
        f"pert_{pair.perturbed_run_id:04d}_sender_{pair.sender}"
    )


def _export_individual_pairs(
    pairs: list[InterventionPair],
    observations: pd.DataFrame,
    names: list[str],
    horizons: list[float],
    metric: str,
    aggregation: str,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Export one partially observed causal matrix for every run pair.

    One physical intervention identifies only the column associated with its
    sender. All other sender columns remain NaN. The sender's direct response
    is exported separately because inter-drone matrices have a zero diagonal.
    """

    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    index_rows = []
    direct_columns = [
        "horizon_after_event_s",
        "sample_time_s",
        "valid",
        "censored_after_crash",
        *METRICS,
    ]
    for pair in pairs:
        selected = observations[
            (observations["baseline_run_id"] == pair.baseline_run_id)
            & (observations["perturbed_run_id"] == pair.perturbed_run_id)
        ].copy()
        if selected.empty:
            continue
        pair_name = _pair_output_name(pair)
        artifact = _export_group(
            pair_name,
            selected,
            names,
            horizons,
            metric,
            aggregation,
            pairs_dir,
        )
        destination = Path(artifact["directory"])
        direct = selected[
            selected["is_self_effect"].astype(bool)
            & (selected["receiver"] == pair.sender)
        ][direct_columns]
        direct_path = destination / "direct_intervention_response.csv"
        direct.to_csv(direct_path, index=False)
        metadata = {
            "baseline_run_id": pair.baseline_run_id,
            "perturbed_run_id": pair.perturbed_run_id,
            "split": pair.split,
            "leader": pair.leader,
            "intervened_sender": pair.sender,
            "event_time_s": pair.event_time,
            "event_duration_s": pair.duration,
            "force_xyz": list(pair.force),
            "force_norm": pair.force_norm,
            "observed_matrix_column": pair.sender,
            "untested_matrix_columns": [name for name in names if name != pair.sender],
            "matrix_convention": "row=receiver,column=intervened_sender",
            "note": (
                "This pair identifies one sender column only; a complete causal "
                "matrix requires matched interventions on every sender."
            ),
        }
        metadata_path = destination / "pair_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifact.update(
            baseline_run_id=pair.baseline_run_id,
            perturbed_run_id=pair.perturbed_run_id,
            leader=pair.leader,
            sender=pair.sender,
            direct_response=str(direct_path.resolve()),
            metadata=str(metadata_path.resolve()),
        )
        artifacts[pair_name] = artifact
        index_rows.append(
            {
                "pair_name": pair_name,
                "baseline_run_id": pair.baseline_run_id,
                "perturbed_run_id": pair.perturbed_run_id,
                "split": pair.split,
                "leader": pair.leader,
                "sender": pair.sender,
                "force_norm": pair.force_norm,
                "event_time_s": pair.event_time,
                "event_duration_s": pair.duration,
                "directory": artifact["directory"],
                "plot": artifact["plot"],
            }
        )
    pd.DataFrame(index_rows).to_csv(pairs_dir / "pair_index.csv", index=False)
    return artifacts


def build_causal_matrices(
    log_dir: Path,
    output_dir: Path,
    horizons: Iterable[float],
    *,
    metric: str = "position_effect_per_force",
    aggregation: str = "median",
    pre_tolerance: float = 1e-9,
    strict_preintervention: bool = True,
    censor_after_crash: bool = True,
) -> dict[str, Any]:
    horizons = sorted(set(float(value) for value in horizons))
    if not horizons or min(horizons) < 0.0:
        raise ValueError("Causal horizons must be non-negative.")
    if pre_tolerance < 0.0:
        raise ValueError("pre_tolerance must be non-negative.")
    pairs = read_intervention_pairs(log_dir)
    names = discover_drone_names(log_dir, pairs[0].baseline_run_id)
    if not names:
        raise ValueError("No drone trajectory CSV was found for the first baseline.")

    rows: list[dict[str, Any]] = []
    audits = []
    for pair in pairs:
        pair_names = discover_drone_names(log_dir, pair.baseline_run_id)
        if pair_names != names:
            raise ValueError(
                f"Run {pair.baseline_run_id} uses drone ordering {pair_names}, expected {names}."
            )
        baseline = load_run_states(log_dir, pair.baseline_run_id, names)
        perturbed = load_run_states(log_dir, pair.perturbed_run_id, names)
        audit = validate_deterministic_pair(
            baseline, perturbed, pair.event_time, pre_tolerance
        )
        audit.update(
            baseline_run_id=pair.baseline_run_id,
            perturbed_run_id=pair.perturbed_run_id,
        )
        audits.append(audit)
        if not audit["passed"]:
            if strict_preintervention:
                raise ValueError(
                    f"Runs {pair.baseline_run_id}/{pair.perturbed_run_id} diverge "
                    f"before intervention: {audit}"
                )
            continue
        rows.extend(
            pair_effect_observations(
                pair,
                baseline,
                perturbed,
                horizons,
                censor_after_crash=censor_after_crash,
            )
        )

    if not rows:
        raise ValueError("No valid paired intervention effect could be computed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = pd.DataFrame(rows)
    observations.to_csv(output_dir / "causal_effect_observations.csv", index=False)
    pd.DataFrame(audits).to_csv(output_dir / "pair_determinism_audit.csv", index=False)

    groups: dict[str, pd.DataFrame] = {"all": observations}
    for split, selected in observations.groupby("split", sort=True):
        groups[f"split_{split}"] = selected
    for leader, selected in observations.groupby("leader", sort=True):
        groups[f"leader_{leader}"] = selected
    artifacts = {
        name: _export_group(
            name,
            selected,
            names,
            horizons,
            metric,
            aggregation,
            output_dir,
        )
        for name, selected in groups.items()
    }
    pair_artifacts = _export_individual_pairs(
        pairs,
        observations,
        names,
        horizons,
        metric,
        aggregation,
        output_dir,
    )
    report = {
        "definition": (
            "Each C[receiver,sender](h) aggregates the selected paired-run effect "
            "metric for receiver after intervening on sender."
        ),
        "position_effect_definition": (
            "||position_receiver_perturbed(t_event_end+h) - "
            "position_receiver_baseline(t_event_end+h)||"
        ),
        "interpretation": (
            "Empirical total effect of a localized simulator intervention; "
            "separate from an NRI predictive adjacency."
        ),
        "matrix_convention": "row=receiver,column=intervened_sender",
        "horizon_origin": "end of first intervention",
        "horizons_s": horizons,
        "metric": metric,
        "aggregation": aggregation,
        "drone_names": names,
        "paired_interventions": len(pairs),
        "valid_pairs": int(sum(bool(audit["passed"]) for audit in audits)),
        "quality_control": {
            "valid_observations_by_horizon": {
                str(float(horizon)): int(count)
                for horizon, count in observations.loc[
                    observations["valid"].astype(bool)
                ]
                .groupby("horizon_after_event_s")
                .size()
                .items()
            },
            "sender_crash_censored_rows": int(
                observations["censored_sender_crash"].sum()
            ),
            "receiver_crash_censored_rows": int(
                observations["censored_receiver_crash"].sum()
            ),
        },
        "censor_after_crash": censor_after_crash,
        "untested_entries": "NaN",
        "self_effect": "exported in observations but diagonal is zero in matrices",
        "individual_pair_matrices": {
            "count": len(pair_artifacts),
            "index": str((output_dir / "pairs" / "pair_index.csv").resolve()),
            "note": (
                "Each pair matrix contains only the experimentally observed sender "
                "column; all other sender columns are untested."
            ),
            "pairs": pair_artifacts,
        },
        "groups": artifacts,
        "artifacts": {
            "observations": str((output_dir / "causal_effect_observations.csv").resolve()),
            "determinism_audit": str((output_dir / "pair_determinism_audit.csv").resolve()),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build intervention-validated drone causal effect matrices."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("datasets/leader_switch_triangle_100runs_60s"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_out/interventional_causal_matrix"),
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    parser.add_argument("--metric", choices=METRICS, default="position_effect_per_force")
    parser.add_argument("--aggregation", choices=("mean", "median"), default="median")
    parser.add_argument("--pre-tolerance", type=float, default=1e-9)
    parser.add_argument("--allow-preintervention-mismatch", action="store_true")
    parser.add_argument("--include-post-crash", action="store_true")
    args = parser.parse_args()

    report = build_causal_matrices(
        args.logs_dir,
        args.output_dir,
        args.horizons,
        metric=args.metric,
        aggregation=args.aggregation,
        pre_tolerance=args.pre_tolerance,
        strict_preintervention=not args.allow_preintervention_mismatch,
        censor_after_crash=not args.include_post_crash,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
