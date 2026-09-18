#!/usr/bin/env python3
"""Create a paired normal/perturbed seeded plan for UAV formation logs.

The output CSV is compatible with experiments/run_seeded_batch.py --plan-csv.

Default design:
- 4 formations
- paired baseline and perturbed seeds
- same simulation_seed, waypoint_seed and formation_seed for each pair
- configurable swarm leader, propagated to every plan row
- weak horizontal pushes, each restricted to exactly one of the x/y axes
- stratified, seeded intervention times after an 8 s warm-up
- 60 s simulations with an 8 s counterfactual horizon
- optional single_plus_isolated layout with drone_4 as a negative-control UAV
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_FORMATIONS = ["triangle", "trail", "line", "v"]
DEFAULT_TARGETS = ["drone_0", "drone_1", "drone_2", "drone_3"]
DEFAULT_PERTURBATION_AXES = ["x", "y"]
FIELDNAMES = [
    "run_id",
    "seed_block",
    "paired_baseline_run_id",
    "split",
    "simulation_seed",
    "waypoint_seed",
    "formation_seed",
    "deterministic_execution",
    "formation",
    "leader",
    "swarm_layout",
    "two_swarm_mode",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "random_waypoints_enabled",
    "max_sim_time_s",
    "warmup_s",
    "rollout_horizon_s",
    "snapshot_seed",
    "snapshot_times_s",
    "counterfactual_forks_enabled",
    "intervention_enabled",
    "intervention_target",
    "intervention_start_time",
    "intervention_interval_s",
    "intervention_repeat_until_s",
    "intervention_duration",
    "intervention_type",
    "intervention_axis",
    "intervention_force_x",
    "intervention_force_y",
    "intervention_force_z",
    "isolated_motion_type",
    "isolated_motion_center_x",
    "isolated_motion_center_y",
    "isolated_motion_center_z",
    "isolated_motion_amplitude_x",
    "isolated_motion_amplitude_y",
    "isolated_motion_amplitude_z",
    "isolated_motion_period_x",
    "isolated_motion_period_y",
    "isolated_motion_period_z",
    "isolated_motion_phase_x",
    "isolated_motion_phase_y",
    "isolated_motion_phase_z",
    "isolated_motion_warmup_s",
]


def parse_force(value: str) -> list[float]:
    """Parse a generic x,y,z vector used by the isolated motion profile."""

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Vector must contain exactly three comma-separated values.")
    return [float(part) for part in parts]


def next_available_run_id(logs_dir: Path) -> int:
    """Return one plus the largest run id currently present in logs_dir."""

    next_id = 0
    if logs_dir.exists():
        for path in logs_dir.iterdir():
            match = re.search(r"run_(\d+)", path.name)
            if match:
                next_id = max(next_id, int(match.group(1)) + 1)
    return next_id


def horizontal_force(axis: str, magnitude: float) -> list[float]:
    """Return a single-axis horizontal force; vertical interventions are forbidden."""

    axis = str(axis).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("Perturbation axis must be either 'x' or 'y'.")
    if magnitude <= 0.0:
        raise ValueError("Perturbation magnitude must be strictly positive.")
    return [magnitude, 0.0, 0.0] if axis == "x" else [0.0, magnitude, 0.0]


def stratified_snapshot_times(
    *,
    seed: int,
    first_window_start: float,
    interval: float,
    window_width: float,
    warmup: float,
    max_sim_time: float,
    rollout_horizon: float,
) -> list[float]:
    """Draw one reproducible time from each non-overlapping time stratum."""

    latest_time = float(max_sim_time) - float(rollout_horizon)
    if first_window_start < warmup:
        raise ValueError("The first snapshot window must start after warm-up.")
    if latest_time <= warmup:
        raise ValueError("max_sim_time must leave room for warm-up and rollout horizon.")
    if interval <= 0.0:
        raise ValueError("Snapshot interval must be strictly positive.")
    if window_width < 0.0 or window_width >= interval:
        raise ValueError("Snapshot window width must satisfy 0 <= width < interval.")

    rng = random.Random(int(seed))
    times = []
    window_start = float(first_window_start)
    while window_start <= latest_time + 1e-12:
        window_end = min(window_start + float(window_width), latest_time)
        times.append(round(rng.uniform(window_start, window_end), 6))
        window_start += float(interval)
    if not times:
        raise ValueError("No valid snapshot window fits inside the simulation.")
    return times


def parse_spacing(value: str) -> list[float]:
    """Parse a formation spacing vector."""

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) not in {1, 2, 3}:
        raise argparse.ArgumentTypeError("Spacing must contain one, two or three comma-separated values.")
    nums = [float(part) for part in parts]
    if len(nums) == 1:
        return [nums[0], nums[0], 0.0]
    if len(nums) == 2:
        return [nums[0], nums[1], 0.0]
    return nums


def split_for_seed(seed_block: int, n_seed_blocks: int) -> str:
    """Assign train/val/test by seed block, preserving paired rows in the same split."""

    train_cut = max(1, int(round(0.70 * n_seed_blocks)))
    val_cut = max(train_cut + 1, int(round(0.85 * n_seed_blocks)))
    if seed_block < train_cut:
        return "train"
    if seed_block < val_cut:
        return "val"
    return "test"


def perturbation_seed_blocks(n_seed_blocks: int, n_perturbations: int) -> list[int]:
    """Choose seed blocks spread across the whole split range."""

    if n_perturbations <= 0:
        return []
    if n_perturbations >= n_seed_blocks:
        return list(range(n_seed_blocks))
    step = n_seed_blocks / n_perturbations
    selected = []
    for idx in range(n_perturbations):
        selected.append(min(n_seed_blocks - 1, int(round(idx * step))))
    return sorted(dict.fromkeys(selected))


def make_row(
    run_id: int,
    seed_block: int,
    paired_baseline_run_id: int | str,
    split: str,
    formation: str,
    simulation_seed: int,
    waypoint_seed: int,
    formation_seed: int,
    args: argparse.Namespace,
    intervention_target: str = "none",
    intervention_axis: str = "none",
) -> dict[str, Any]:
    """Create one CSV row understood by run_seeded_batch.py."""

    perturbed = intervention_target not in {"", "none"}
    force = (
        horizontal_force(intervention_axis, args.perturbation_magnitude)
        if perturbed
        else [0.0, 0.0, 0.0]
    )
    snapshot_seed = int(simulation_seed) + 300_000
    snapshot_times = stratified_snapshot_times(
        seed=snapshot_seed,
        first_window_start=args.perturbation_start_time,
        interval=args.perturbation_interval,
        window_width=args.snapshot_window_width,
        warmup=args.warmup,
        max_sim_time=args.max_sim_time,
        rollout_horizon=args.rollout_horizon,
    )
    return {
        "run_id": run_id,
        "seed_block": seed_block,
        "paired_baseline_run_id": paired_baseline_run_id,
        "split": split,
        "simulation_seed": simulation_seed,
        "waypoint_seed": waypoint_seed,
        "formation_seed": formation_seed,
        "deterministic_execution": 1,
        "formation": formation,
        "leader": args.leader,
        "swarm_layout": args.swarm_layout,
        "two_swarm_mode": "",
        "spacing_x": args.spacing[0],
        "spacing_y": args.spacing[1],
        "spacing_z": args.spacing[2],
        "random_waypoints_enabled": 1,
        "max_sim_time_s": args.max_sim_time,
        "warmup_s": args.warmup,
        "rollout_horizon_s": args.rollout_horizon,
        "snapshot_seed": snapshot_seed,
        "snapshot_times_s": json.dumps(snapshot_times),
        "counterfactual_forks_enabled": int(perturbed),
        "intervention_enabled": int(perturbed),
        "intervention_target": intervention_target if perturbed else "none",
        "intervention_start_time": snapshot_times[0] if perturbed else "",
        "intervention_interval_s": args.perturbation_interval if perturbed else "",
        "intervention_repeat_until_s": (
            args.max_sim_time - args.rollout_horizon if perturbed else ""
        ),
        "intervention_duration": args.perturbation_duration if perturbed else "",
        "intervention_type": args.perturbation_type if perturbed else "none",
        "intervention_axis": intervention_axis if perturbed else "none",
        "intervention_force_x": force[0],
        "intervention_force_y": force[1],
        "intervention_force_z": 0.0,
        "isolated_motion_type": args.isolated_motion_type if args.swarm_layout == "single_plus_isolated" else "none",
        "isolated_motion_center_x": args.isolated_motion_center[0],
        "isolated_motion_center_y": args.isolated_motion_center[1],
        "isolated_motion_center_z": args.isolated_motion_center[2],
        "isolated_motion_amplitude_x": args.isolated_motion_amplitude[0],
        "isolated_motion_amplitude_y": args.isolated_motion_amplitude[1],
        "isolated_motion_amplitude_z": args.isolated_motion_amplitude[2],
        "isolated_motion_period_x": args.isolated_motion_period[0],
        "isolated_motion_period_y": args.isolated_motion_period[1],
        "isolated_motion_period_z": args.isolated_motion_period[2],
        "isolated_motion_phase_x": args.isolated_motion_phase[0],
        "isolated_motion_phase_y": args.isolated_motion_phase[1],
        "isolated_motion_phase_z": args.isolated_motion_phase[2],
        "isolated_motion_warmup_s": args.isolated_motion_warmup,
    }


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build rows with paired baseline and perturbed runs."""

    n_seed_blocks = args.runs_per_formation - args.perturbations_per_formation
    if n_seed_blocks <= 0:
        raise ValueError("--perturbations-per-formation must be smaller than --runs-per-formation.")
    perturbation_targets = (
        list(args.perturbation_targets)
        if args.perturbation_targets
        else [name for name in DEFAULT_TARGETS if name != args.leader]
    )
    invalid_targets = sorted(set(perturbation_targets) - set(DEFAULT_TARGETS))
    if invalid_targets:
        raise ValueError(
            "Perturbations must target members of swarm A only "
            f"({', '.join(DEFAULT_TARGETS)}); invalid targets: {invalid_targets}"
        )
    if args.leader not in DEFAULT_TARGETS:
        raise ValueError(f"Unknown leader '{args.leader}'.")
    if args.leader in perturbation_targets:
        raise ValueError("Perturbation targets must be followers; remove the configured leader.")
    if not perturbation_targets:
        raise ValueError("At least one follower must be available for perturbation.")
    invalid_axes = sorted(set(args.perturbation_axes) - set(DEFAULT_PERTURBATION_AXES))
    if invalid_axes or not args.perturbation_axes:
        raise ValueError("--perturbation-axes must contain one or both of: x y.")
    horizontal_force(args.perturbation_axes[0], args.perturbation_magnitude)
    if args.perturbation_interval <= 0.0:
        raise ValueError("--perturbation-interval must be strictly positive.")
    if args.perturbation_duration <= 0.0:
        raise ValueError("--perturbation-duration must be strictly positive.")
    if args.perturbation_duration >= args.perturbation_interval:
        raise ValueError("--perturbation-duration must be shorter than --perturbation-interval.")
    if args.perturbation_duration >= args.perturbation_interval - args.snapshot_window_width:
        raise ValueError(
            "--perturbation-duration must be shorter than the minimum spacing between snapshots."
        )
    perturb_blocks = set(perturbation_seed_blocks(n_seed_blocks, args.perturbations_per_formation))
    rows: list[dict[str, Any]] = []
    run_id = args.start_run_id

    for formation_index, formation in enumerate(args.formations):
        baseline_run_by_seed: dict[int, int] = {}
        for seed_block in range(n_seed_blocks):
            split = split_for_seed(seed_block, n_seed_blocks)
            seed_base = args.base_seed + formation_index * 10_000 + seed_block
            global_seed_block = formation_index * n_seed_blocks + seed_block
            baseline_run_id = run_id
            baseline_run_by_seed[seed_block] = baseline_run_id
            rows.append(
                make_row(
                    run_id=baseline_run_id,
                    seed_block=global_seed_block,
                    paired_baseline_run_id="",
                    split=split,
                    formation=formation,
                    simulation_seed=seed_base,
                    waypoint_seed=seed_base + 100_000,
                    formation_seed=seed_base + 200_000,
                    args=args,
                )
            )
            run_id += 1

        for local_idx, seed_block in enumerate(sorted(perturb_blocks)):
            split = split_for_seed(seed_block, n_seed_blocks)
            seed_base = args.base_seed + formation_index * 10_000 + seed_block
            global_seed_block = formation_index * n_seed_blocks + seed_block
            global_perturbation_idx = formation_index * len(perturb_blocks) + local_idx
            target = perturbation_targets[
                global_perturbation_idx % len(perturbation_targets)
            ]
            axis = args.perturbation_axes[
                global_perturbation_idx % len(args.perturbation_axes)
            ]
            rows.append(
                make_row(
                    run_id=run_id,
                    seed_block=global_seed_block,
                    paired_baseline_run_id=baseline_run_by_seed[seed_block],
                    split=split,
                    formation=formation,
                    simulation_seed=seed_base,
                    waypoint_seed=seed_base + 100_000,
                    formation_seed=seed_base + 200_000,
                    args=args,
                    intervention_target=target,
                    intervention_axis=axis,
                )
            )
            run_id += 1

    return rows


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the generated plan CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def add_plan_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add paired-plan options to a standalone parser or CLI subparser."""

    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--out-plan",
        type=Path,
        default=Path("logs/all_formations_60s_intervention_plan.csv"),
    )
    parser.add_argument("--start-run-id", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=824200)
    parser.add_argument("--formations", nargs="+", default=DEFAULT_FORMATIONS)
    parser.add_argument("--leader", choices=DEFAULT_TARGETS, default="drone_0")
    parser.add_argument("--swarm-layout", choices=["single", "single_plus_isolated"], default="single")
    parser.add_argument("--runs-per-formation", type=int, default=30)
    parser.add_argument("--perturbations-per-formation", type=int, default=10)
    parser.add_argument(
        "--perturbation-targets",
        nargs="+",
        default=None,
        help="Follower drones to perturb. Defaults to every UAV except --leader.",
    )
    parser.add_argument(
        "--perturbation-axes",
        nargs="+",
        choices=DEFAULT_PERTURBATION_AXES,
        default=DEFAULT_PERTURBATION_AXES,
        help="Horizontal axes assigned cyclically to perturbed runs.",
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=0.02,
        help="Weak horizontal force magnitude in newtons (default: 0.02 N).",
    )
    parser.add_argument("--perturbation-start-time", type=float, default=10.0)
    parser.add_argument(
        "--perturbation-interval",
        type=float,
        default=10.0,
        help="Seconds between repeated pushes in each perturbed run.",
    )
    parser.add_argument("--perturbation-duration", type=float, default=1.0)
    parser.add_argument("--perturbation-type", default="controlled_push")
    parser.add_argument("--max-sim-time", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--rollout-horizon", type=float, default=8.0)
    parser.add_argument(
        "--snapshot-window-width",
        type=float,
        default=2.0,
        help="Uniform jitter width inside each stratified snapshot window.",
    )
    parser.add_argument("--spacing", type=parse_spacing, default=[1.0, 1.0, 0.0])
    parser.add_argument("--isolated-motion-type", choices=["none", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--isolated-motion-center", type=parse_force, default=[0.0, 0.0, 55.0])
    parser.add_argument("--isolated-motion-amplitude", type=parse_force, default=[48.0, 48.0, 1.0])
    parser.add_argument("--isolated-motion-period", type=parse_force, default=[80.0, 80.0, 40.0])
    parser.add_argument("--isolated-motion-phase", type=parse_force, default=[0.0, -1.57079632679, 0.0])
    parser.add_argument("--isolated-motion-warmup", type=float, default=20.0)
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a paired perturbation plan CSV.")
    return add_plan_arguments(parser)


def main() -> None:
    args = build_argparser().parse_args()
    if args.start_run_id is None:
        args.start_run_id = next_available_run_id(args.logs_dir)
    rows = build_plan(args)
    write_plan(args.out_plan, rows)
    n_perturbed = sum(int(row["intervention_enabled"]) for row in rows)
    print(f"Wrote {len(rows)} rows to {args.out_plan}")
    print(f"Formations: {', '.join(args.formations)}")
    print(f"Leader: {args.leader}")
    print(
        "Perturbations: "
        f"axes={','.join(args.perturbation_axes)}, magnitude={args.perturbation_magnitude} N"
    )
    print(f"Runs per formation: {args.runs_per_formation}")
    print(f"Perturbed per formation: {args.perturbations_per_formation}")
    print(
        "Snapshot windows: "
        f"start={args.perturbation_start_time}s, interval={args.perturbation_interval}s, "
        f"jitter_width={args.snapshot_window_width}s, horizon={args.rollout_horizon}s"
    )
    print(f"Baseline rows: {len(rows) - n_perturbed}")
    print(f"Perturbed rows: {n_perturbed}")
    print(f"Run id range: {rows[0]['run_id']}..{rows[-1]['run_id']}")


if __name__ == "__main__":
    main()
