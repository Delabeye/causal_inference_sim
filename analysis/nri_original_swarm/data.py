"""Run-level splitting and window construction for the swarm NRI baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config


STATE_FILE_RE = re.compile(r"run_(\d+)_(drone_\d+)\.csv$")


@dataclass
class Normalization:
    minimum: np.ndarray
    maximum: np.ndarray

    def apply(self, states: np.ndarray) -> np.ndarray:
        span = np.maximum(self.maximum - self.minimum, 1e-8)
        return ((states - self.minimum) * 2.0 / span - 1.0).astype(np.float32)

    def inverse(self, states: np.ndarray) -> np.ndarray:
        span = np.maximum(self.maximum - self.minimum, 1e-8)
        return ((states + 1.0) * 0.5 * span + self.minimum).astype(np.float32)

    def as_dict(self) -> dict:
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist()}


class SwarmWindowDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        contexts: np.ndarray,
        relations: np.ndarray,
        has_truth: np.ndarray,
        run_ids: np.ndarray,
        start_times: np.ndarray,
        end_times: np.ndarray,
        times: np.ndarray,
    ):
        self.states = torch.from_numpy(states.astype(np.float32, copy=False))
        self.contexts = torch.from_numpy(contexts.astype(np.float32, copy=False))
        self.relations = torch.from_numpy(relations.astype(np.int64, copy=False))
        self.has_truth = torch.from_numpy(has_truth.astype(np.bool_, copy=False))
        self.run_ids = torch.from_numpy(run_ids.astype(np.int64, copy=False))
        self.start_times = torch.from_numpy(start_times.astype(np.float64, copy=False))
        self.end_times = torch.from_numpy(end_times.astype(np.float64, copy=False))
        self.times = torch.from_numpy(times.astype(np.float64, copy=False))

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "contexts": self.contexts[index],
            "relations": self.relations[index],
            "has_truth": self.has_truth[index],
            "run_id": self.run_ids[index],
            "start_time": self.start_times[index],
            "end_time": self.end_times[index],
            "times": self.times[index],
        }


@dataclass
class PreparedData:
    train: SwarmWindowDataset
    val: SwarmWindowDataset
    test: SwarmWindowDataset
    normalization: Normalization
    context_normalization: Normalization
    split_run_ids: dict[str, list[int]]
    skipped_runs: dict[int, str]


def edge_pairs(drone_names: tuple[str, ...]):
    return [
        (sender, receiver)
        for receiver in drone_names
        for sender in drone_names
        if sender != receiver
    ]


def discover_state_files(log_dir: Path) -> dict[int, dict[str, Path]]:
    runs: dict[int, dict[str, Path]] = {}
    for path in Path(log_dir).glob("run_*_drone_*.csv"):
        match = STATE_FILE_RE.match(path.name)
        if match:
            runs.setdefault(int(match.group(1)), {})[match.group(2)] = path
    return runs


def _read_plan_splits(log_dir: Path, available: set[int]):
    plan_path = Path(log_dir) / "plan.csv"
    if not plan_path.exists():
        return None
    frame = pd.read_csv(plan_path)
    if not {"run_id", "split"}.issubset(frame.columns):
        return None
    result = {"train": [], "val": [], "test": []}
    for row in frame[["run_id", "split"]].itertuples(index=False):
        run_id = int(row.run_id)
        split = str(row.split).strip().lower()
        if run_id in available and split in result:
            result[split].append(run_id)
    if all(result.values()):
        return {key: sorted(set(value)) for key, value in result.items()}
    return None


def _run_group_key(log_dir: Path, run_id: int):
    summary_path = Path(log_dir) / f"run_{run_id}_summary.json"
    if not summary_path.exists():
        return ("run", run_id)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seeds = summary.get("seeds", {})
        formation_seeds = tuple(
            sorted(
                (str(key), value.get("effective_seed"))
                for key, value in seeds.get("formation_seeds", {}).items()
            )
        )
        return (
            "seed",
            seeds.get("effective_seed", seeds.get("simulation_seed")),
            seeds.get("random_waypoints_effective_seed"),
            formation_seeds,
        )
    except (OSError, ValueError, TypeError):
        return ("run", run_id)


def choose_run_splits(config: Config, available: set[int]):
    explicit = {
        "train": sorted(set(config.train_run_ids)),
        "val": sorted(set(config.val_run_ids)),
        "test": sorted(set(config.test_run_ids)),
    }
    if any(explicit.values()):
        if not all(explicit.values()):
            raise ValueError("Set all three explicit run lists, or leave all three empty.")
        selected = set().union(*map(set, explicit.values()))
        missing = sorted(selected - available)
        if missing:
            raise ValueError(f"Explicit run ids are missing complete UAV logs: {missing}")
        if sum(map(len, explicit.values())) != len(selected):
            raise ValueError("The explicit train/val/test run lists overlap.")
        return explicit

    planned = _read_plan_splits(config.log_dir, available)
    if planned is not None:
        return planned

    groups: dict[tuple, list[int]] = {}
    for run_id in sorted(available):
        groups.setdefault(_run_group_key(config.log_dir, run_id), []).append(run_id)
    keys = list(groups)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(keys)
    if len(keys) < 3:
        raise ValueError("At least three independent run/seed groups are required.")
    train_end = max(1, int(round(len(keys) * config.train_fraction)))
    val_count = max(1, int(round(len(keys) * config.val_fraction)))
    train_end = min(train_end, len(keys) - 2)
    val_end = min(train_end + val_count, len(keys) - 1)
    key_splits = {
        "train": keys[:train_end],
        "val": keys[train_end:val_end],
        "test": keys[val_end:],
    }
    return {
        split: sorted(run_id for key in selected for run_id in groups[key])
        for split, selected in key_splits.items()
    }


def _context_source_columns(config: Config) -> set[str]:
    source_by_feature = {
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
    unknown = set(config.context_feature_columns) - set(source_by_feature)
    if unknown:
        raise ValueError(f"Unknown context features: {sorted(unknown)}")
    return {source_by_feature[name] for name in config.context_feature_columns}


def read_run_states(paths: dict[str, Path], config: Config):
    context_sources = _context_source_columns(config)
    required = {
        "time",
        *config.feature_columns,
        *context_sources,
        "swarm_id",
        "is_leader",
    }
    frames = {}
    metadata = {}
    for drone_name in config.drone_names:
        frame = pd.read_csv(paths[drone_name], usecols=lambda name: name in required)
        missing = {"time", *config.feature_columns, *context_sources} - set(frame.columns)
        if missing:
            raise ValueError(f"{paths[drone_name].name} lacks columns {sorted(missing)}")
        frame["time_key"] = frame["time"].round(6)
        frame = frame.drop_duplicates("time_key", keep="first").set_index("time_key")
        frames[drone_name] = frame
        metadata[drone_name] = {
            "swarm_id": str(frame["swarm_id"].iloc[0])
            if "swarm_id" in frame and pd.notna(frame["swarm_id"].iloc[0])
            else "",
            "is_leader": bool(int(frame["is_leader"].iloc[0]))
            if "is_leader" in frame
            else False,
        }

    common_times = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    common_times = np.asarray(common_times[:: config.downsample], dtype=np.float64)
    if len(common_times) < config.seq_len:
        raise ValueError(
            f"only {len(common_times)} aligned/downsampled timestamps, need {config.seq_len}"
        )
    states = np.stack(
        [
            frames[name].loc[common_times, list(config.feature_columns)].to_numpy(dtype=np.float32)
            for name in config.drone_names
        ],
        axis=1,
    )  # [time, nodes, features]
    if not np.isfinite(states).all():
        raise ValueError("non-finite values in selected state features")

    if not config.context_feature_columns:
        contexts = np.empty(
            (len(common_times), len(config.drone_names), 0), dtype=np.float32
        )
        return common_times, states, contexts, metadata

    contexts_by_drone = []
    for node_index, drone_name in enumerate(config.drone_names):
        aligned = frames[drone_name].loc[common_times]
        drone_meta = metadata[drone_name]
        is_follower = bool(drone_meta["swarm_id"]) and not drone_meta["is_leader"]
        values = {}
        for axis_index, axis in enumerate("xyz"):
            context_name = f"navigation_target_rel_{axis}"
            source_name = f"target_{axis}"
            if not is_follower and source_name in aligned:
                values[context_name] = (
                    aligned[source_name].to_numpy(dtype=np.float32)
                    - states[:, node_index, axis_index]
                )
            else:
                values[context_name] = np.zeros(len(common_times), dtype=np.float32)
        for name in config.context_feature_columns:
            if name in values:
                continue
            column = aligned[name].to_numpy(dtype=np.float32)
            if name == "nearest_obstacle_dist":
                column = np.clip(column, 0.0, config.obstacle_distance_clip)
            values[name] = column
        contexts_by_drone.append(
            np.stack([values[name] for name in config.context_feature_columns], axis=-1)
        )
    contexts = np.stack(contexts_by_drone, axis=1)
    if not np.isfinite(contexts).all():
        raise ValueError("non-finite values in selected exogenous context")
    return common_times, states, contexts, metadata


def structural_relations(metadata: dict, drone_names: tuple[str, ...]):
    matrix = np.zeros((len(drone_names), len(drone_names)), dtype=np.int64)
    for receiver_index, receiver in enumerate(drone_names):
        receiver_meta = metadata[receiver]
        if receiver_meta["is_leader"] or not receiver_meta["swarm_id"]:
            continue
        for sender_index, sender in enumerate(drone_names):
            sender_meta = metadata[sender]
            if (
                sender_meta["is_leader"]
                and sender_meta["swarm_id"]
                and sender_meta["swarm_id"] == receiver_meta["swarm_id"]
            ):
                matrix[receiver_index, sender_index] = 1
    return matrix


def flatten_off_diagonal(matrix: np.ndarray):
    return np.asarray(
        [
            matrix[receiver, sender]
            for receiver in range(matrix.shape[0])
            for sender in range(matrix.shape[1])
            if receiver != sender
        ],
        dtype=np.int64,
    )


def read_interactions(log_dir: Path, run_id: int):
    path = Path(log_dir) / f"run_{run_id}_interactions.csv"
    if not path.exists():
        return None
    needed = {"time", "sender", "receiver", "interaction_active"}
    frame = pd.read_csv(path, usecols=lambda name: name in needed)
    if not needed.issubset(frame.columns):
        raise ValueError(f"{path.name} lacks {sorted(needed - set(frame.columns))}")
    return frame


def active_relations_for_window(
    interactions: pd.DataFrame,
    drone_names: tuple[str, ...],
    start_time: float,
    end_time: float,
    threshold: float,
):
    selected = interactions[
        (interactions["time"] >= start_time - 1e-8)
        & (interactions["time"] <= end_time + 1e-8)
    ]
    if selected.empty:
        return None
    grouped = selected.groupby(["receiver", "sender"])["interaction_active"].mean()
    matrix = np.zeros((len(drone_names), len(drone_names)), dtype=np.int64)
    for receiver_index, receiver in enumerate(drone_names):
        for sender_index, sender in enumerate(drone_names):
            if receiver == sender:
                continue
            fraction = float(grouped.get((receiver, sender), 0.0))
            matrix[receiver_index, sender_index] = int(fraction >= threshold)
    return matrix


def _window_starts(length: int, config: Config):
    starts = list(range(0, length - config.seq_len + 1, config.window_stride))
    maximum = config.max_windows_per_run
    if maximum is not None and len(starts) > maximum:
        indices = np.linspace(0, len(starts) - 1, maximum, dtype=int)
        starts = [starts[index] for index in sorted(set(indices.tolist()))]
    return starts


def _empty_dataset(config: Config):
    edges = len(config.drone_names) * (len(config.drone_names) - 1)
    return SwarmWindowDataset(
        np.empty((0, len(config.drone_names), config.seq_len, len(config.feature_columns)), np.float32),
        np.empty(
            (
                0,
                len(config.drone_names),
                config.seq_len,
                len(config.context_feature_columns),
            ),
            np.float32,
        ),
        np.empty((0, edges), np.int64),
        np.empty((0,), bool),
        np.empty((0,), np.int64),
        np.empty((0,), np.float64),
        np.empty((0,), np.float64),
        np.empty((0, config.seq_len), np.float64),
    )


def prepare_data(config: Config) -> PreparedData:
    config.validate()
    discovered = discover_state_files(config.log_dir)
    required_drones = set(config.drone_names)
    complete = {
        run_id: paths
        for run_id, paths in discovered.items()
        if required_drones.issubset(paths)
    }
    if not complete:
        raise FileNotFoundError(
            f"No run in {config.log_dir} contains all drones {config.drone_names}."
        )
    split_ids = choose_run_splits(config, set(complete))

    loaded = {}
    skipped: dict[int, str] = {}
    for run_id in sorted(set().union(*map(set, split_ids.values()))):
        try:
            loaded[run_id] = read_run_states(complete[run_id], config)
        except Exception as exc:
            skipped[run_id] = str(exc)

    usable_train = [run_id for run_id in split_ids["train"] if run_id in loaded]
    if not usable_train:
        raise ValueError("No usable training run remains after loading/alignment.")
    train_values = np.concatenate([loaded[run_id][1] for run_id in usable_train], axis=0)
    normalization = Normalization(
        minimum=train_values.min(axis=(0, 1), keepdims=True),
        maximum=train_values.max(axis=(0, 1), keepdims=True),
    )
    train_contexts = np.concatenate(
        [loaded[run_id][2] for run_id in usable_train], axis=0
    )
    if train_contexts.shape[-1]:
        context_normalization = Normalization(
            minimum=train_contexts.min(axis=(0, 1), keepdims=True),
            maximum=train_contexts.max(axis=(0, 1), keepdims=True),
        )
    else:
        empty = np.empty((1, 1, 0), dtype=np.float32)
        context_normalization = Normalization(minimum=empty, maximum=empty)

    datasets = {}
    edges_count = len(config.drone_names) * (len(config.drone_names) - 1)
    for split, run_ids in split_ids.items():
        states_out = []
        contexts_out = []
        relations_out = []
        truth_out = []
        run_out = []
        start_out = []
        end_out = []
        times_out = []
        for run_id in run_ids:
            if run_id not in loaded:
                continue
            times, states, contexts, metadata = loaded[run_id]
            normalized = normalization.apply(states)
            normalized_contexts = context_normalization.apply(contexts)
            navigation_indices = [
                config.context_feature_columns.index(name)
                for name in (
                    "navigation_target_rel_x",
                    "navigation_target_rel_y",
                    "navigation_target_rel_z",
                )
                if name in config.context_feature_columns
            ]
            for node_index, drone_name in enumerate(config.drone_names):
                drone_meta = metadata[drone_name]
                if bool(drone_meta["swarm_id"]) and not drone_meta["is_leader"]:
                    normalized_contexts[:, node_index, navigation_indices] = 0.0
            static_matrix = structural_relations(metadata, config.drone_names)
            interactions = (
                read_interactions(config.log_dir, run_id)
                if config.ground_truth_mode == "active"
                else None
            )
            for start in _window_starts(len(times), config):
                stop = start + config.seq_len
                if config.ground_truth_mode == "structural":
                    relation_matrix = static_matrix
                elif interactions is not None:
                    relation_matrix = active_relations_for_window(
                        interactions,
                        config.drone_names,
                        float(times[start]),
                        float(times[stop - 1]),
                        config.active_fraction_threshold,
                    )
                else:
                    relation_matrix = None

                has_truth = relation_matrix is not None
                if config.require_ground_truth and not has_truth:
                    continue
                relation_vector = (
                    flatten_off_diagonal(relation_matrix)
                    if has_truth
                    else np.full(edges_count, -1, dtype=np.int64)
                )
                states_out.append(normalized[start:stop].transpose(1, 0, 2))
                contexts_out.append(
                    normalized_contexts[start:stop].transpose(1, 0, 2)
                )
                relations_out.append(relation_vector)
                truth_out.append(has_truth)
                run_out.append(run_id)
                start_out.append(times[start])
                end_out.append(times[stop - 1])
                times_out.append(times[start:stop])

        if not states_out:
            datasets[split] = _empty_dataset(config)
        else:
            datasets[split] = SwarmWindowDataset(
                np.stack(states_out),
                np.stack(contexts_out),
                np.stack(relations_out),
                np.asarray(truth_out),
                np.asarray(run_out),
                np.asarray(start_out),
                np.asarray(end_out),
                np.stack(times_out),
            )

    if any(len(datasets[split]) == 0 for split in ("train", "val", "test")):
        sizes = {split: len(dataset) for split, dataset in datasets.items()}
        raise ValueError(f"An empty split was produced: {sizes}")
    return PreparedData(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        normalization=normalization,
        context_normalization=context_normalization,
        split_run_ids={
            split: [run_id for run_id in ids if run_id in loaded]
            for split, ids in split_ids.items()
        },
        skipped_runs=skipped,
    )
