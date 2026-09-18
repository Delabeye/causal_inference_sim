"""Run-level data loading for the conditional Granger baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.nri_original_swarm.data import choose_run_splits, discover_state_files

from .config import Config


CONTROL_SOURCE_COLUMNS = {
    "navigation_target_rel_x": "target_x",
    "navigation_target_rel_y": "target_y",
    "navigation_target_rel_z": "target_z",
    "desired_offset_x": "desired_offset_x",
    "desired_offset_y": "desired_offset_y",
    "desired_offset_z": "desired_offset_z",
    "wind_x": "wind_x",
    "wind_y": "wind_y",
    "wind_z": "wind_z",
    "nearest_obstacle_dist": "nearest_obstacle_dist",
    "nearest_obstacle_dir_x": "nearest_obstacle_dir_x",
    "nearest_obstacle_dir_y": "nearest_obstacle_dir_y",
    "nearest_obstacle_dir_z": "nearest_obstacle_dir_z",
    "external_force_x": "external_force_x",
    "external_force_y": "external_force_y",
    "external_force_z": "external_force_z",
}


@dataclass
class RunData:
    run_id: int
    times: np.ndarray
    states: np.ndarray  # [time, node, state]
    controls: np.ndarray  # [time, node, control]
    structural_relations: np.ndarray  # [receiver, sender]
    formation: str
    intervention_target: str
    paired_baseline_run_id: int | None


@dataclass
class PreparedRuns:
    train: list[RunData]
    val: list[RunData]
    test: list[RunData]
    split_run_ids: dict[str, list[int]]
    skipped_runs: dict[int, str]


def _read_plan(log_dir: Path) -> dict[int, dict[str, str]]:
    path = Path(log_dir) / "plan.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["run_id"]): row
            for row in csv.DictReader(handle)
            if row.get("run_id", "").strip()
        }


def _load_run(
    run_id: int,
    paths: dict[str, Path],
    plan_row: dict[str, str],
    config: Config,
) -> RunData:
    unknown_controls = set(config.control_columns) - set(CONTROL_SOURCE_COLUMNS)
    if unknown_controls:
        raise ValueError(f"Unknown Granger controls: {sorted(unknown_controls)}")
    control_sources = {
        CONTROL_SOURCE_COLUMNS[column] for column in config.control_columns
    }
    wanted = {
        "time",
        *config.state_columns,
        *control_sources,
        "formation_type",
        "swarm_id",
        "is_leader",
    }
    frames = {}
    controls_by_drone = {}
    formations = []
    for name in config.drone_names:
        frame = pd.read_csv(paths[name], usecols=lambda column: column in wanted)
        missing = {"time", *config.state_columns} - set(frame.columns)
        if missing:
            raise ValueError(f"{paths[name].name} lacks {sorted(missing)}")
        for column in control_sources:
            if column not in frame:
                frame[column] = 0.0
        if "formation_type" in frame and not frame["formation_type"].dropna().empty:
            formations.append(str(frame["formation_type"].dropna().iloc[0]))
        frame["time_key"] = frame["time"].round(6)
        frame = frame.drop_duplicates("time_key", keep="first").set_index("time_key")
        frames[name] = frame

    common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    times = np.asarray(common[:: config.downsample], dtype=np.float64)
    if len(times) <= config.granger_lags + 2:
        raise ValueError("Run is too short after downsampling.")
    states = np.stack(
        [
            frames[name].loc[times, list(config.state_columns)].to_numpy(np.float64)
            for name in config.drone_names
        ],
        axis=1,
    )
    for node_index, name in enumerate(config.drone_names):
        aligned = frames[name].loc[times]
        swarm_id = (
            str(aligned["swarm_id"].iloc[0])
            if "swarm_id" in aligned and pd.notna(aligned["swarm_id"].iloc[0])
            else ""
        )
        is_leader = (
            bool(int(aligned["is_leader"].iloc[0]))
            if "is_leader" in aligned
            else False
        )
        is_follower = bool(swarm_id) and not is_leader
        values = {}
        for axis_index, axis in enumerate("xyz"):
            control_name = f"navigation_target_rel_{axis}"
            if control_name not in config.control_columns:
                continue
            values[control_name] = (
                np.zeros(len(times), dtype=np.float64)
                if is_follower
                else aligned[f"target_{axis}"].to_numpy(np.float64)
                - states[:, node_index, axis_index]
            )
        for control_name in config.control_columns:
            if control_name in values:
                continue
            source_name = CONTROL_SOURCE_COLUMNS[control_name]
            column = aligned[source_name].to_numpy(np.float64)
            if control_name == "nearest_obstacle_dist":
                column = np.clip(column, 0.0, config.obstacle_distance_clip)
            values[control_name] = column
        controls_by_drone[name] = np.stack(
            [values[column] for column in config.control_columns], axis=-1
        )
    controls = np.stack(
        [controls_by_drone[name] for name in config.drone_names],
        axis=1,
    )
    structural_relations = np.zeros(
        (len(config.drone_names), len(config.drone_names)), dtype=np.int64
    )
    metadata = {}
    for name in config.drone_names:
        aligned = frames[name].loc[times]
        metadata[name] = {
            "swarm_id": (
                str(aligned["swarm_id"].iloc[0])
                if "swarm_id" in aligned and pd.notna(aligned["swarm_id"].iloc[0])
                else ""
            ),
            "is_leader": (
                bool(int(aligned["is_leader"].iloc[0]))
                if "is_leader" in aligned
                else False
            ),
        }
    for receiver_index, receiver in enumerate(config.drone_names):
        receiver_meta = metadata[receiver]
        if receiver_meta["is_leader"] or not receiver_meta["swarm_id"]:
            continue
        for sender_index, sender in enumerate(config.drone_names):
            sender_meta = metadata[sender]
            if (
                sender_meta["is_leader"]
                and sender_meta["swarm_id"] == receiver_meta["swarm_id"]
            ):
                structural_relations[receiver_index, sender_index] = 1
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        raise ValueError("Non-finite state or control values.")

    paired = str(plan_row.get("paired_baseline_run_id", "")).strip()
    return RunData(
        run_id=run_id,
        times=times,
        states=states,
        controls=controls,
        structural_relations=structural_relations,
        formation=str(plan_row.get("formation", formations[0] if formations else "unknown")),
        intervention_target=str(plan_row.get("intervention_target", "none") or "none"),
        paired_baseline_run_id=int(float(paired)) if paired else None,
    )


def prepare_runs(config: Config) -> PreparedRuns:
    config.validate()
    discovered = discover_state_files(config.log_dir)
    required = set(config.drone_names)
    complete = {
        run_id: paths
        for run_id, paths in discovered.items()
        if required.issubset(paths)
    }
    plan = _read_plan(config.log_dir)
    if config.require_all_planned_runs and plan:
        missing = sorted(set(plan) - set(complete))
        if missing:
            preview = missing[:12]
            suffix = "..." if len(missing) > len(preview) else ""
            raise RuntimeError(
                f"Dataset is incomplete: {len(missing)} of {len(plan)} planned runs "
                f"still lack complete CSV/NPZ artifacts ({preview}{suffix})."
            )
    if not complete:
        raise FileNotFoundError(f"No complete run found in {config.log_dir}.")
    splits = choose_run_splits(config, set(complete))
    loaded = {}
    skipped = {}
    for run_id in sorted(set().union(*(set(ids) for ids in splits.values()))):
        try:
            loaded[run_id] = _load_run(
                run_id, complete[run_id], plan.get(run_id, {}), config
            )
        except Exception as exc:
            skipped[run_id] = str(exc)
    split_runs = {
        split: [loaded[run_id] for run_id in ids if run_id in loaded]
        for split, ids in splits.items()
    }
    if any(not values for values in split_runs.values()):
        raise ValueError(
            f"An empty split remains after loading: "
            f"{ {key: len(value) for key, value in split_runs.items()} }; skipped={skipped}"
        )
    return PreparedRuns(
        train=split_runs["train"],
        val=split_runs["val"],
        test=split_runs["test"],
        split_run_ids={key: [run.run_id for run in value] for key, value in split_runs.items()},
        skipped_runs=skipped,
    )


__all__ = ["PreparedRuns", "RunData", "prepare_runs"]
