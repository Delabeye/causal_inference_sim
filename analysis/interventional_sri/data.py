"""Build nominal/fork windows from simulator learning-trace artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MAIN_TRACE_PATTERN = re.compile(r"run_(\d+)_learning_trace\.npz$")


@dataclass(frozen=True)
class StateNormalization:
    """Feature-wise min/max transformation fitted on training runs only."""

    minimum: np.ndarray
    maximum: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        scale = np.maximum(self.maximum - self.minimum, 1e-6)
        return (2.0 * (values - self.minimum) / scale - 1.0).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Return normalized states to simulator units."""

        scale = np.maximum(self.maximum - self.minimum, 1e-6)
        return (((values + 1.0) * 0.5) * scale + self.minimum).astype(
            np.float32
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "minimum": self.minimum.astype(float).tolist(),
            "maximum": self.maximum.astype(float).tolist(),
        }


@dataclass
class RawWindow:
    run_id: int
    group_id: int
    snapshot_time: float
    history_states: np.ndarray
    baseline_future: np.ndarray
    intervention_future: np.ndarray
    intervention_input: np.ndarray
    paired: bool


class CounterfactualWindowDataset(Dataset):
    """Fixed-shape windows consumed by :class:`InterventionalSRIModel`."""

    def __init__(self, samples: list[RawWindow], normalization: StateNormalization):
        self.run_ids = torch.as_tensor([sample.run_id for sample in samples], dtype=torch.long)
        self.snapshot_times = torch.as_tensor(
            [sample.snapshot_time for sample in samples], dtype=torch.float64
        )
        self.paired = torch.as_tensor(
            [sample.paired for sample in samples], dtype=torch.bool
        )
        self.history_states = torch.from_numpy(
            np.stack(
                [normalization.apply(sample.history_states) for sample in samples]
            )
        )
        self.baseline_future = torch.from_numpy(
            np.stack(
                [normalization.apply(sample.baseline_future) for sample in samples]
            )
        )
        self.intervention_future = torch.from_numpy(
            np.stack(
                [normalization.apply(sample.intervention_future) for sample in samples]
            )
        )
        self.intervention_input = torch.from_numpy(
            np.stack([sample.intervention_input for sample in samples]).astype(
                np.float32, copy=False
            )
        )

    def __len__(self) -> int:
        return int(self.run_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history_states": self.history_states[index],
            "baseline_future": self.baseline_future[index],
            "intervention_future": self.intervention_future[index],
            "intervention_input": self.intervention_input[index],
            "paired": self.paired[index],
            "run_id": self.run_ids[index],
            "snapshot_time": self.snapshot_times[index],
        }


@dataclass
class PreparedCounterfactualData:
    train: CounterfactualWindowDataset
    val: CounterfactualWindowDataset
    test: CounterfactualWindowDataset
    normalization: StateNormalization
    split_run_ids: dict[str, list[int]]
    skipped: list[str]


def _trace_path(value: str | Path, log_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = log_dir / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def _load_trace(path: Path, drone_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"times", "drone_names", "state"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path.name} lacks arrays {sorted(missing)}")
        stored_names = [str(value) for value in archive["drone_names"].tolist()]
        absent = set(drone_names) - set(stored_names)
        if absent:
            raise ValueError(f"{path.name} lacks drones {sorted(absent)}")
        order = [stored_names.index(name) for name in drone_names]
        result = {
            "times": np.asarray(archive["times"], dtype=np.float64),
            "state": np.asarray(archive["state"], dtype=np.float32)[:, order],
        }
        for name in (
            "intervention_force_command",
            "intervention_force_world",
            "intervention_active",
        ):
            if name in archive.files:
                values = np.asarray(archive[name])
                result[name] = values[:, order]
    if len(result["times"]) != len(result["state"]):
        raise ValueError(f"{path.name} contains inconsistent time/state lengths.")
    return result


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_left = np.abs(reference[left] - query) <= np.abs(reference[right] - query)
    indices = np.where(use_left, left, right)
    return indices, np.abs(reference[indices] - query)


def _history_indices(
    times: np.ndarray,
    snapshot_time: float,
    history_steps: int,
    downsample: int,
    tolerance: float,
) -> np.ndarray:
    eligible = np.flatnonzero(times <= snapshot_time + tolerance)
    if not len(eligible):
        return np.empty(0, dtype=int)
    selected = eligible[::-downsample][:history_steps][::-1]
    return selected if len(selected) == history_steps else np.empty(0, dtype=int)


def _intervention_descriptor(
    branch: dict[str, np.ndarray],
    selected: np.ndarray,
    force_scale: float,
) -> np.ndarray:
    force = branch.get("intervention_force_command")
    if force is None:
        force = branch.get("intervention_force_world")
    if force is None:
        force = np.zeros((*branch["state"].shape[:2], 3), dtype=np.float32)
    active = branch.get("intervention_active")
    if active is None:
        active = np.zeros(branch["state"].shape[:2], dtype=np.float32)
    normalized_force = np.asarray(force[selected], np.float32) / force_scale
    active = np.asarray(active[selected], np.float32)[..., None]
    return np.concatenate([normalized_force, active], axis=-1).transpose(1, 0, 2)


def _paired_samples(
    log_dir: Path,
    drone_names: tuple[str, ...],
    history_steps: int,
    rollout_steps: int,
    downsample: int,
    force_scale: float,
    time_tolerance: float,
    group_by_run: dict[int, int],
) -> tuple[list[RawWindow], set[int], list[str]]:
    samples: list[RawWindow] = []
    fork_runs: set[int] = set()
    skipped: list[str] = []
    for manifest_path in sorted(log_dir.glob("run_*_counterfactual_forks.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = int(manifest["run"])
            fork_runs.add(run_id)
            for record in manifest.get("forks", []):
                parent_path = _trace_path(record["baseline_trace"], log_dir)
                branch_path = _trace_path(record["intervention_trace"], log_dir)
                parent = _load_trace(parent_path, drone_names)
                branch = _load_trace(branch_path, drone_names)
                snapshot_time = float(record["actual_snapshot_time"])
                history_index = _history_indices(
                    parent["times"],
                    snapshot_time,
                    history_steps,
                    downsample,
                    time_tolerance,
                )
                selected = np.arange(0, len(branch["times"]), downsample)[
                    :rollout_steps
                ]
                if len(history_index) != history_steps or len(selected) != rollout_steps:
                    raise ValueError(
                        "insufficient history or fork rollout length "
                        f"(history={len(history_index)}/{history_steps}, "
                        f"rollout={len(selected)}/{rollout_steps}, "
                        f"branch_rows={len(branch['times'])}, "
                        f"downsample={downsample})"
                    )
                future_times = branch["times"][selected]
                parent_index, errors = _nearest_indices(parent["times"], future_times)
                if errors.max(initial=0.0) > time_tolerance:
                    raise ValueError(
                        f"parent/fork alignment error {errors.max():.6f}s"
                    )
                samples.append(
                    RawWindow(
                        run_id=run_id,
                        group_id=group_by_run.get(run_id, run_id),
                        snapshot_time=snapshot_time,
                        history_states=parent["state"][history_index].transpose(1, 0, 2),
                        baseline_future=parent["state"][parent_index].transpose(1, 0, 2),
                        intervention_future=branch["state"][selected].transpose(1, 0, 2),
                        intervention_input=_intervention_descriptor(
                            branch, selected, force_scale
                        ),
                        paired=True,
                    )
                )
        except Exception as exc:
            skipped.append(f"{manifest_path.name}: {exc}")
    return samples, fork_runs, skipped


def _baseline_samples(
    log_dir: Path,
    drone_names: tuple[str, ...],
    history_steps: int,
    rollout_steps: int,
    downsample: int,
    window_stride: int,
    maximum_per_run: int | None,
    excluded_runs: set[int],
    group_by_run: dict[int, int],
) -> tuple[list[RawWindow], list[str]]:
    samples: list[RawWindow] = []
    skipped: list[str] = []
    for path in sorted(log_dir.glob("run_*_learning_trace.npz")):
        match = MAIN_TRACE_PATTERN.search(path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        if run_id in excluded_runs:
            continue
        try:
            trace = _load_trace(path, drone_names)
            first_anchor = (history_steps - 1) * downsample
            final_anchor = len(trace["times"]) - 1 - rollout_steps * downsample
            anchors = list(
                range(first_anchor, final_anchor + 1, window_stride * downsample)
            )
            if maximum_per_run is not None and len(anchors) > maximum_per_run:
                positions = np.linspace(0, len(anchors) - 1, maximum_per_run, dtype=int)
                anchors = [anchors[index] for index in sorted(set(positions.tolist()))]
            for anchor in anchors:
                history_index = anchor - downsample * np.arange(history_steps - 1, -1, -1)
                future_index = anchor + downsample * np.arange(1, rollout_steps + 1)
                baseline = trace["state"][future_index].transpose(1, 0, 2)
                intervention = np.zeros(
                    (len(drone_names), rollout_steps, 4), dtype=np.float32
                )
                samples.append(
                    RawWindow(
                        run_id=run_id,
                        group_id=group_by_run.get(run_id, run_id),
                        snapshot_time=float(trace["times"][anchor]),
                        history_states=trace["state"][history_index].transpose(1, 0, 2),
                        baseline_future=baseline,
                        intervention_future=baseline.copy(),
                        intervention_input=intervention,
                        paired=False,
                    )
                )
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
    return samples, skipped


def _pair_groups(log_dir: Path) -> dict[int, int]:
    path = log_dir / "plan.csv"
    if not path.exists():
        return {}
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_text = str(row.get("run_id", "")).strip()
            if not run_text:
                continue
            run_id = int(float(run_text))
            baseline = str(row.get("paired_baseline_run_id", "")).strip().lower()
            result[run_id] = (
                run_id
                if baseline in {"", "none", "nan"}
                else int(float(baseline))
            )
    return result


def _split_groups(
    samples: list[RawWindow], seed: int, train_fraction: float, val_fraction: float
) -> dict[str, set[int]]:
    groups = np.asarray(sorted({sample.group_id for sample in samples}), dtype=int)
    if len(groups) < 3:
        raise ValueError("At least three independent run groups are required.")
    np.random.default_rng(seed).shuffle(groups)
    train_count = max(1, int(round(len(groups) * train_fraction)))
    val_count = max(1, int(round(len(groups) * val_fraction)))
    if train_count + val_count >= len(groups):
        train_count = max(1, len(groups) - 2)
        val_count = 1
    return {
        "train": set(groups[:train_count].tolist()),
        "val": set(groups[train_count : train_count + val_count].tolist()),
        "test": set(groups[train_count + val_count :].tolist()),
    }


def _fit_normalization(samples: list[RawWindow]) -> StateNormalization:
    values = np.concatenate(
        [
            np.concatenate(
                [
                    sample.history_states.reshape(-1, sample.history_states.shape[-1]),
                    sample.baseline_future.reshape(-1, sample.baseline_future.shape[-1]),
                    sample.intervention_future.reshape(
                        -1, sample.intervention_future.shape[-1]
                    ),
                ],
                axis=0,
            )
            for sample in samples
        ],
        axis=0,
    )
    return StateNormalization(values.min(axis=0), values.max(axis=0))


def prepare_counterfactual_data(config) -> PreparedCounterfactualData:
    """Load physical forks and nominal windows without run-level leakage."""

    config.validate()
    log_dir = Path(config.log_dir)
    group_by_run = _pair_groups(log_dir)
    paired, fork_runs, skipped = _paired_samples(
        log_dir,
        config.drone_names,
        config.history_steps,
        config.rollout_steps,
        config.downsample,
        config.intervention_force_scale,
        config.time_tolerance_s,
        group_by_run,
    )
    nominal, baseline_skipped = _baseline_samples(
        log_dir,
        config.drone_names,
        config.history_steps,
        config.rollout_steps,
        config.downsample,
        config.baseline_window_stride,
        config.max_baseline_windows_per_run,
        fork_runs,
        group_by_run,
    )
    skipped.extend(baseline_skipped)
    samples = paired + nominal
    if not paired:
        manifest_count = sum(
            1 for _ in log_dir.glob("run_*_counterfactual_forks.json")
        )
        details = "; ".join(skipped[:3])
        if len(skipped) > 3:
            details += f"; ... ({len(skipped) - 3} more)"
        raise FileNotFoundError(
            f"No usable counterfactual fork was found in {log_dir} "
            f"({manifest_count} manifest(s) inspected)."
            + (f" Rejections: {details}" if details else "")
        )
    if not nominal:
        raise FileNotFoundError(
            f"No usable non-perturbed parent trace was found in {log_dir}."
        )
    split_groups = _split_groups(
        samples, config.seed, config.train_fraction, config.val_fraction
    )
    raw_by_split = {
        split: [sample for sample in samples if sample.group_id in groups]
        for split, groups in split_groups.items()
    }
    if any(not values for values in raw_by_split.values()):
        raise ValueError(
            "The run-level split produced an empty dataset: "
            f"{ {key: len(value) for key, value in raw_by_split.items()} }"
        )
    normalization = _fit_normalization(raw_by_split["train"])
    datasets = {
        split: CounterfactualWindowDataset(values, normalization)
        for split, values in raw_by_split.items()
    }
    return PreparedCounterfactualData(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        normalization=normalization,
        split_run_ids={
            split: sorted({sample.run_id for sample in values})
            for split, values in raw_by_split.items()
        },
        skipped=skipped,
    )


__all__ = [
    "CounterfactualWindowDataset",
    "PreparedCounterfactualData",
    "RawWindow",
    "StateNormalization",
    "prepare_counterfactual_data",
]
