#!/usr/bin/env python3
"""Run reproducible seeded simulation batches.

Examples:
    python experiments/run_seeded_batch.py --dry-run --n-seeds 3
    python experiments/run_seeded_batch.py --n-seeds 5 --formations triangle trail line
    python experiments/run_seeded_batch.py --dry-run --n-seeds 3 --formations trail --swarm-layout two_pairs
    python experiments/run_seeded_batch.py --dry-run --n-seeds 3 --perturbation-targets none drone_0 drone_1
    python experiments/run_seeded_batch.py --plan-csv logs/seeded_batch_plan.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator.simulator_manager import SimulationManager
from utilities.config import load_config, save_config
from experiments.verify_paired_logs import compare_run_pair


DEFAULT_FORMATIONS = ["triangle", "trail", "line", "v"]
DEFAULT_UAV_NAMES = ["drone_0", "drone_1", "drone_2", "drone_3"]
LEGACY_TWO_SWARM_FORMATIONS = {"two_swarm", "two_swarm_trail", "two_swarms", "dual_swarm"}


def next_available_run_id(logs_dir: Path) -> int:
    next_id = 0
    if logs_dir.exists():
        for path in logs_dir.iterdir():
            match = re.search(r"run_(\d+)", path.name)
            if match:
                next_id = max(next_id, int(match.group(1)) + 1)
    return next_id


def parse_spacing(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) not in {1, 2, 3}:
        raise argparse.ArgumentTypeError("--spacing must contain 1, 2, or 3 comma-separated numbers.")
    numbers = [float(part) for part in parts]
    if len(numbers) == 1:
        return [numbers[0], numbers[0], 0.0]
    if len(numbers) == 2:
        return [numbers[0], numbers[1], 0.0]
    return numbers


def first_swarm_config(cfg: dict[str, Any]) -> dict[str, Any]:
    swarm_cfg = cfg.setdefault("swarm", [])
    if isinstance(swarm_cfg, list):
        if not swarm_cfg:
            swarm_cfg.append({"id": "A"})
        return swarm_cfg[0]
    if isinstance(swarm_cfg, dict):
        return swarm_cfg
    raise TypeError("config['swarm'] must be a list or dict.")


def agent_by_name(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(agent.get("name")): agent
        for agent in cfg.get("agents", [])
        if agent.get("type") == "uav" and agent.get("name") is not None
    }


def apply_single_swarm_leader(cfg: dict[str, Any], leader_name: str) -> None:
    """Configure a star formation around any UAV in the single swarm."""

    leader_name = str(leader_name).strip()
    agents = agent_by_name(cfg)
    if leader_name not in agents:
        raise ValueError(f"Configured leader '{leader_name}' is not a UAV in the scenario.")

    swarm_cfg = first_swarm_config(cfg)
    swarm_id = str(swarm_cfg.get("id", "A"))
    members = {
        name: agent
        for name, agent in agents.items()
        if str(agent.get("swarm_id", "")) == swarm_id
    }
    if leader_name not in members:
        raise ValueError(
            f"Configured leader '{leader_name}' does not belong to swarm '{swarm_id}'."
        )

    previous_leader = str(swarm_cfg.get("leader", "")).strip()
    previous_waypoints = copy.deepcopy(
        members.get(previous_leader, {}).get("waypoints", [])
    )
    swarm_cfg["leader"] = leader_name

    random_wp_cfg = cfg.setdefault("random_waypoints", {})
    configured_agents = list(random_wp_cfg.get("agents", []))
    non_swarm_agents = [name for name in configured_agents if name not in members]
    random_wp_cfg["agents"] = [leader_name, *non_swarm_agents]

    for name, agent in members.items():
        if name != leader_name:
            agent["waypoints"] = []
    if not bool(random_wp_cfg.get("enabled", False)):
        leader_waypoints = members[leader_name].get("waypoints", [])
        if not leader_waypoints and previous_waypoints:
            members[leader_name]["waypoints"] = previous_waypoints

    cfg.setdefault("experiment", {})["formation_leader"] = leader_name


def apply_two_pair_swarm_layout(
    cfg: dict[str, Any],
    formation_type: str,
    spacing: list[float],
    formation_seed: int,
    mode: str = "crossing",
) -> None:
    agents = agent_by_name(cfg)
    required = {"drone_0", "drone_1", "drone_2", "drone_3"}
    missing = sorted(required - set(agents.keys()))
    if missing:
        raise ValueError(f"two_pairs swarm layout requires drones {sorted(required)}. Missing: {missing}")

    mode = str(mode or "crossing").strip().lower()
    formation_type = str(formation_type or "trail").strip().lower()

    # Two separated takeoff groups. The two leaders then receive different waypoint sets.
    starts = {
        "drone_0": [0.0, -6.0, 0.0],
        "drone_1": [-0.6, -6.0, 0.0],
        "drone_2": [0.0, 6.0, 0.0],
        "drone_3": [-0.6, 6.0, 0.0],
    }
    for name, start_pos in starts.items():
        agents[name]["start_pos"] = start_pos

    if mode == "parallel":
        wp_a = [[12.0, -6.0, 2.0], [28.0, -6.0, 3.0], [44.0, -6.0, 3.0]]
        wp_b = [[12.0, 6.0, 2.0], [28.0, 6.0, 3.0], [44.0, 6.0, 3.0]]
    elif mode == "diverging":
        wp_a = [[12.0, -6.0, 2.0], [30.0, -14.0, 3.0], [48.0, -18.0, 3.0]]
        wp_b = [[12.0, 6.0, 2.0], [30.0, 14.0, 3.0], [48.0, 18.0, 3.0]]
    elif mode == "crossing":
        wp_a = [[12.0, -6.0, 2.0], [28.0, 4.0, 3.0], [44.0, 10.0, 3.0]]
        wp_b = [[12.0, 6.0, 2.0], [28.0, -4.0, 3.0], [44.0, -10.0, 3.0]]
    else:
        raise ValueError("--two-swarm-mode must be one of: crossing, parallel, diverging.")

    agents["drone_0"]["swarm_id"] = "A"
    agents["drone_1"]["swarm_id"] = "A"
    agents["drone_2"]["swarm_id"] = "B"
    agents["drone_3"]["swarm_id"] = "B"
    agents["drone_0"]["waypoints"] = wp_a
    agents["drone_2"]["waypoints"] = wp_b
    agents["drone_1"]["waypoints"] = []
    agents["drone_3"]["waypoints"] = []

    cfg["swarm"] = [
        {
            "id": "A",
            "leader": "drone_0",
            "min_sep": 0.6,
            "avoid_gain": 0.5,
            "ip": "localhost",
            "formation": {
                "mode": "fixed",
                "type": formation_type,
                "seed": formation_seed,
                "spacing": spacing,
            },
        },
        {
            "id": "B",
            "leader": "drone_2",
            "min_sep": 0.6,
            "avoid_gain": 0.5,
            "ip": "localhost",
            "formation": {
                "mode": "fixed",
                "type": formation_type,
                "seed": formation_seed + 1,
                "spacing": spacing,
            },
        },
    ]

    random_wp_cfg = cfg.setdefault("random_waypoints", {})
    random_wp_cfg["enabled"] = False
    cfg.setdefault("experiment", {})["scenario"] = {
        "type": "two_pairs",
        "mode": mode,
        "formation_type": formation_type,
        "swarm_A": {"leader": "drone_0", "follower": "drone_1", "waypoints": wp_a},
        "swarm_B": {"leader": "drone_2", "follower": "drone_3", "waypoints": wp_b},
    }


def apply_single_plus_isolated_layout(
    cfg: dict[str, Any],
    motion_profile: dict[str, Any] | None = None,
) -> None:
    """Keep swarm A unchanged and add drone_4 as an independent UAV."""

    agents = agent_by_name(cfg)
    fallback_waypoints = [
        [-30.0, -30.0, 3.0],
        [0.0, -30.0, 4.0],
        [30.0, 0.0, 5.0],
        [0.0, 30.0, 4.0],
    ]
    isolated_start = [0.0, -42.0, 0.0]
    if motion_profile and motion_profile.get("enabled", False):
        center = motion_profile["center"]
        amplitude = motion_profile["amplitude"]
        phase = motion_profile["phase"]
        isolated_start = [
            float(center[0]) + float(amplitude[0]) * math.sin(float(phase[0])),
            float(center[1]) + float(amplitude[1]) * math.sin(float(phase[1])),
            0.0,
        ]

    if "drone_4" not in agents:
        if "drone_3" not in agents:
            raise ValueError("single_plus_isolated requires drone_3 as a template for drone_4.")
        template = copy.deepcopy(agents["drone_3"])
        template["name"] = "drone_4"
        template["body_id"] = 4
        template.pop("swarm_id", None)
        template["start_pos"] = isolated_start
        template["waypoints"] = fallback_waypoints
        template["safety_radius"] = 0.25
        template["max_repulsive_force"] = 0.5
        cfg.setdefault("agents", []).append(template)
    else:
        agents["drone_4"].pop("swarm_id", None)
        agents["drone_4"]["start_pos"] = isolated_start
        agents["drone_4"]["waypoints"] = fallback_waypoints

    random_wp_cfg = cfg.setdefault("random_waypoints", {})
    isolated = agent_by_name(cfg)["drone_4"]
    if motion_profile and motion_profile.get("enabled", False):
        isolated["motion_profile"] = copy.deepcopy(motion_profile)
        isolated["waypoints"] = [list(motion_profile["center"])]
        random_wp_cfg["agents"] = ["drone_0"]
        waypoint_description = "deterministic_sinusoidal_motion_profile"
    else:
        isolated.pop("motion_profile", None)
        random_wp_cfg["agents"] = ["drone_0", "drone_4"]
        waypoint_description = "generated_by_random_waypoints_sampler"

    cfg.setdefault("experiment", {})["isolated_drone"] = {
        "name": "drone_4",
        "role": "negative_control",
        "start_pos": isolated_start,
        "waypoints": waypoint_description,
        "motion_profile": copy.deepcopy(motion_profile) if motion_profile else None,
        "expected_adjacency": "zero edges to/from formation swarm",
    }


def isolated_motion_profile_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build the optional deterministic isolated-UAV motion profile from one plan row."""

    motion_type = str(row.get("isolated_motion_type", "")).strip().lower()
    if motion_type in {"", "none", "off", "false", "0"}:
        return None

    def vector(prefix: str, defaults: list[float]) -> list[float]:
        return [
            float(row.get(f"{prefix}_x", defaults[0])),
            float(row.get(f"{prefix}_y", defaults[1])),
            float(row.get(f"{prefix}_z", defaults[2])),
        ]

    return {
        "enabled": True,
        "type": motion_type,
        "center": vector("isolated_motion_center", [0.0, 0.0, 55.0]),
        "amplitude": vector("isolated_motion_amplitude", [48.0, 48.0, 1.0]),
        "period": vector("isolated_motion_period", [80.0, 80.0, 40.0]),
        "phase": vector("isolated_motion_phase", [0.0, -1.57079632679, 0.0]),
        "warmup_s": float(row.get("isolated_motion_warmup_s", 20.0)),
    }


def split_for_block(seed_block: int, n_seeds: int) -> str:
    train_cut = max(1, int(0.7 * n_seeds))
    val_cut = max(train_cut + 1, int(0.85 * n_seeds))
    if seed_block < train_cut:
        return "train"
    if seed_block < val_cut:
        return "val"
    return "test"


def generate_plan(args: argparse.Namespace, start_run_id: int) -> list[dict[str, Any]]:
    if args.perturbation_magnitude <= 0.0:
        raise ValueError("--perturbation-magnitude must be strictly positive.")
    requested_targets = {
        str(target).strip()
        for target in args.perturbation_targets
        if str(target).strip().lower()
        not in {"", "none", "baseline", "off", "false", "0"}
    }
    if args.leader in requested_targets:
        raise ValueError("Perturbation targets must be followers, not the leader.")
    rows: list[dict[str, Any]] = []
    run_id = start_run_id
    for seed_block in range(args.n_seeds):
        for formation in args.formations:
            for target in args.perturbation_targets:
                target = str(target).strip()
                perturbation_enabled = target.lower() not in {"", "none", "baseline", "off", "false", "0"}
                rows.append(
                    {
                        "run_id": run_id,
                        "seed_block": seed_block,
                        "split": split_for_block(seed_block, args.n_seeds),
                        "simulation_seed": args.base_seed + seed_block,
                        "waypoint_seed": args.base_seed + 100_000 + seed_block,
                        "formation_seed": args.base_seed + 200_000 + seed_block,
                        "formation": formation,
                        "leader": args.leader if args.swarm_layout != "two_pairs" else "",
                        "swarm_layout": args.swarm_layout,
                        "two_swarm_mode": args.two_swarm_mode if args.swarm_layout == "two_pairs" else "",
                        "spacing_x": args.spacing[0],
                        "spacing_y": args.spacing[1],
                        "spacing_z": args.spacing[2],
                        "random_waypoints_enabled": int(args.random_waypoints),
                        "max_sim_time_s": args.max_sim_time if args.max_sim_time is not None else "",
                        "deterministic_execution": 1,
                        "intervention_enabled": int(perturbation_enabled),
                        "intervention_target": target if perturbation_enabled else "none",
                        "intervention_start_time": args.perturbation_start_time if perturbation_enabled else "",
                        "intervention_interval_s": args.perturbation_interval if perturbation_enabled else "",
                        "intervention_repeat_until_s": args.max_sim_time if perturbation_enabled else "",
                        "intervention_duration": args.perturbation_duration if perturbation_enabled else "",
                        "intervention_type": args.perturbation_type if perturbation_enabled else "none",
                        "intervention_axis": args.perturbation_axis if perturbation_enabled else "none",
                        "intervention_force_x": args.perturbation_magnitude if perturbation_enabled and args.perturbation_axis == "x" else 0.0,
                        "intervention_force_y": args.perturbation_magnitude if perturbation_enabled and args.perturbation_axis == "y" else 0.0,
                        "intervention_force_z": 0.0,
                    }
                )
                run_id += 1
    return rows


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Plan is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_plan_row(base_config: dict[str, Any], row: dict[str, Any], logs_dir: Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    run_id = int(row["run_id"])
    simulation_seed = int(row["simulation_seed"])
    waypoint_seed = int(row.get("waypoint_seed", simulation_seed + 100_000))
    formation_seed = int(row.get("formation_seed", simulation_seed + 200_000))
    formation_type = row.get("formation", "triangle")
    swarm_layout = str(row.get("swarm_layout", "single")).strip().lower()
    if str(formation_type).strip().lower() in LEGACY_TWO_SWARM_FORMATIONS:
        swarm_layout = "two_pairs"
        formation_type = "trail"
    spacing = [
        float(row.get("spacing_x", 1.0)),
        float(row.get("spacing_y", 1.0)),
        float(row.get("spacing_z", 0.0)),
    ]

    sim_cfg = cfg.setdefault("simulation", {})
    sim_cfg["connect_mode"] = "direct"
    sim_cfg["run_config"] = run_id
    sim_cfg["fixed_run_config"] = True
    sim_cfg["seed"] = simulation_seed
    sim_cfg["seed_add_run_id"] = False
    deterministic_raw = str(row.get("deterministic_execution", "1")).strip().lower()
    sim_cfg["deterministic_execution"] = deterministic_raw in {
        "1",
        "true",
        "yes",
        "y",
    }
    for agent_cfg in cfg.get("agents", []):
        if agent_cfg.get("type") == "uav":
            agent_cfg.setdefault("navigation", {})["synchronous_planning"] = True
    if logs_dir is not None:
        sim_cfg["log_dir"] = str(logs_dir)
    if row.get("max_sim_time_s") not in {None, ""}:
        sim_cfg["max_sim_time"] = float(row["max_sim_time_s"])

    # Persist the pairing contract inside every run summary/config.  Dataset
    # builders should not have to recover causal pairs from a separate CSV that
    # may have been moved or renamed after simulation.
    paired_raw = str(row.get("paired_baseline_run_id", "")).strip()
    paired_baseline_run_id = (
        None
        if paired_raw.lower() in {"", "nan", "none"}
        else int(float(paired_raw))
    )
    seed_block_raw = str(row.get("seed_block", "")).strip()
    cfg.setdefault("experiment", {})["pairing"] = {
        "pair_id": run_id if paired_baseline_run_id is None else paired_baseline_run_id,
        "seed_block": (
            None
            if seed_block_raw.lower() in {"", "nan", "none"}
            else int(float(seed_block_raw))
        ),
        "split": str(row.get("split", "unspecified")),
        "paired_baseline_run_id": paired_baseline_run_id,
        "is_baseline": paired_baseline_run_id is None,
    }
    snapshot_times_raw = str(row.get("snapshot_times_s", "")).strip()
    snapshot_times = []
    if snapshot_times_raw.lower() not in {"", "nan", "none"}:
        try:
            snapshot_times = [float(value) for value in json.loads(snapshot_times_raw)]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("snapshot_times_s must be a JSON list of times.") from exc
    cfg.setdefault("experiment", {})["snapshot_schedule"] = {
        "warmup_s": float(row.get("warmup_s", 0.0) or 0.0),
        "rollout_horizon_s": float(row.get("rollout_horizon_s", 0.0) or 0.0),
        "snapshot_seed": (
            None
            if str(row.get("snapshot_seed", "")).strip().lower() in {"", "nan", "none"}
            else int(float(row["snapshot_seed"]))
        ),
        "times_s": snapshot_times,
    }
    forks_enabled_raw = str(
        row.get("counterfactual_forks_enabled", "0")
    ).strip().lower()
    cfg["counterfactual_forks"] = {
        "enabled": forks_enabled_raw in {"1", "true", "yes", "y"},
        "rollout_horizon_s": float(row.get("rollout_horizon_s", 8.0) or 8.0),
    }

    rwp_cfg = cfg.setdefault("random_waypoints", {})
    if "random_waypoints_enabled" in row:
        rwp_cfg["enabled"] = str(row["random_waypoints_enabled"]).strip().lower() in {"1", "true", "yes", "y"}
    rwp_cfg["seed"] = waypoint_seed
    rwp_cfg["seed_add_run_id"] = False

    if swarm_layout == "two_pairs":
        apply_two_pair_swarm_layout(
            cfg,
            formation_type=formation_type,
            spacing=spacing,
            formation_seed=formation_seed,
            mode=row.get("two_swarm_mode", "crossing"),
        )
    elif swarm_layout in {"single_plus_isolated", "single_isolated", "isolated"}:
        swarm_cfg = first_swarm_config(cfg)
        formation_cfg = swarm_cfg.setdefault("formation", {})
        formation_cfg["mode"] = "fixed"
        formation_cfg["type"] = formation_type
        formation_cfg["seed"] = formation_seed
        formation_cfg["spacing"] = spacing
        apply_single_plus_isolated_layout(
            cfg,
            motion_profile=isolated_motion_profile_from_row(row),
        )
    elif swarm_layout in {"", "single", "one"}:
        swarm_cfg = first_swarm_config(cfg)
        formation_cfg = swarm_cfg.setdefault("formation", {})
        formation_cfg["mode"] = "fixed"
        formation_cfg["type"] = formation_type
        formation_cfg["seed"] = formation_seed
        formation_cfg["spacing"] = spacing
    else:
        raise ValueError(f"Unknown swarm_layout '{swarm_layout}'. Use 'single', 'single_plus_isolated' or 'two_pairs'.")

    leader_name = str(row.get("leader", "")).strip()
    if leader_name:
        if swarm_layout == "two_pairs":
            raise ValueError("A single 'leader' column is ambiguous for the two_pairs layout.")
        apply_single_swarm_leader(cfg, leader_name)

    target = str(row.get("intervention_target", "none")).strip()
    enabled_raw = str(row.get("intervention_enabled", "")).strip().lower()
    intervention_enabled = enabled_raw in {"1", "true", "yes", "y"}
    if enabled_raw == "":
        intervention_enabled = target.lower() not in {"", "none", "baseline", "off", "false", "0"}

    if intervention_enabled:
        targets = [part.strip() for part in target.split(",") if part.strip()]
        force = [
            float(row.get("intervention_force_x", 0.0)),
            float(row.get("intervention_force_y", 0.0)),
            float(row.get("intervention_force_z", 0.0)),
        ]
        active_horizontal_axes = sum(abs(value) > 1e-12 for value in force[:2])
        if abs(force[2]) > 1e-12 or active_horizontal_axes != 1:
            raise ValueError(
                "Interventions must use exactly one horizontal force axis (x or y) "
                "and intervention_force_z must be zero."
            )
        start_time = float(row.get("intervention_start_time", 20.0))
        duration = float(row.get("intervention_duration", 1.0))
        interval_raw = row.get("intervention_interval_s", row.get("intervention_interval", ""))
        interval = float(interval_raw) if interval_raw not in {None, ""} else 0.0
        repeat_until_raw = row.get("intervention_repeat_until_s", "")
        repeat_until = (
            float(repeat_until_raw)
            if repeat_until_raw not in {None, ""}
            else float(sim_cfg.get("max_sim_time", start_time + duration))
        )
        if duration <= 0.0:
            raise ValueError("intervention_duration must be strictly positive.")
        if interval > 0.0 and duration >= interval:
            raise ValueError("intervention_duration must be shorter than intervention_interval_s.")

        event_times = snapshot_times or [start_time]
        if not snapshot_times and interval > 0.0:
            event_times = []
            event_time = start_time
            while event_time < repeat_until - 1e-9:
                event_times.append(event_time)
                event_time += interval

        events = [
            {
                "id": f"run_{run_id}_controlled_push_{'_'.join(targets)}_{event_idx}",
                "targets": targets,
                "start_time": event_time,
                "duration": duration,
                "type": str(row.get("intervention_type", "controlled_push")),
                "force": force,
                "frame": str(row.get("intervention_frame", "world")),
            }
            for event_idx, event_time in enumerate(event_times)
        ]
        cfg["interventions"] = {
            "enabled": True,
            "events": events,
        }
    else:
        interventions_cfg = cfg.setdefault("interventions", {})
        interventions_cfg["enabled"] = False

    return cfg


def run_one(config: dict[str, Any], config_dir: Path) -> None:
    run_id = int(config["simulation"]["run_config"])
    config_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(config_dir / f"run_{run_id}_config.yaml"))

    sim = SimulationManager(config)
    try:
        sim.run()
    finally:
        sim.stop()


def execute_plan_rows(
    base_config: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    *,
    logs_dir: Path,
    plan_path: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate and execute plan rows, returning an auditable batch summary."""

    summary: dict[str, Any] = {
        "plan_csv": str(plan_path),
        "n_runs": len(plan_rows),
        "run_ids": [int(row["run_id"]) for row in plan_rows],
        "formations": sorted({row["formation"] for row in plan_rows}),
        "swarm_layouts": sorted(
            {row.get("swarm_layout", "single") for row in plan_rows}
        ),
        "intervention_targets": sorted(
            {row.get("intervention_target", "none") for row in plan_rows}
        ),
        "dry_run": bool(dry_run),
        "completed_run_ids": [],
        "skipped_run_ids": [],
    }
    print(json.dumps(summary, indent=2))

    # A dry run validates every row by constructing its effective config.  It
    # catches invalid targets, forces, layouts and fork schedules without
    # starting PyBullet.
    if dry_run:
        for row in plan_rows:
            apply_plan_row(base_config, row, logs_dir)
        summary["validated_run_ids"] = summary["run_ids"]
        return summary

    config_dir = logs_dir / "config_save"
    for idx, row in enumerate(plan_rows, start=1):
        run_id = int(row["run_id"])
        summary_path = logs_dir / f"run_{run_id}_summary.json"
        if resume and summary_path.exists():
            print(
                f"\n[Batch] Skip run_id={run_id}: "
                f"{summary_path.name} already exists."
            )
            summary["skipped_run_ids"].append(run_id)
            continue
        print(
            f"\n[Batch] Run {idx}/{len(plan_rows)}: "
            f"run_id={row['run_id']} formation={row['formation']} "
            f"layout={row.get('swarm_layout', 'single')} "
            f"intervention={row.get('intervention_target', 'none')} "
            f"seed={row['simulation_seed']}"
        )
        config = apply_plan_row(base_config, row, logs_dir)
        run_one(config, config_dir)
        summary["completed_run_ids"].append(run_id)

        paired_raw = str(row.get("paired_baseline_run_id", "")).strip()
        if paired_raw and paired_raw.lower() not in {"nan", "none"}:
            baseline_run_id = int(float(paired_raw))
            intervention_time = float(row.get("intervention_start_time", 0.0))
            pair_report = compare_run_pair(
                logs_dir,
                baseline_run_id,
                run_id,
                before_time=intervention_time,
                atol=0.0,
            )
            report_path = logs_dir / f"run_{run_id}_preintervention_check.json"
            report_path.write_text(
                json.dumps(pair_report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if not pair_report["passed"]:
                raise RuntimeError(
                    "Paired runs diverged before the intervention. "
                    f"See {report_path}."
                )
            print(
                "[Paired determinism] PASS: "
                f"run {baseline_run_id} == run {run_id} "
                f"for t < {intervention_time}s"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a seeded UAV experiment batch.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--plan-csv", default=None, help="Existing plan CSV to execute.")
    parser.add_argument("--out-plan", default="logs/seeded_batch_plan.csv")
    parser.add_argument("--dry-run", action="store_true", help="Only write/print the plan; do not run simulations.")
    parser.add_argument("--resume", action="store_true", help="Skip runs whose summary JSON already exists in logs-dir.")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seed blocks.")
    parser.add_argument("--base-seed", type=int, default=424200)
    parser.add_argument("--start-run-id", type=int, default=None)
    parser.add_argument("--formations", nargs="+", default=DEFAULT_FORMATIONS)
    parser.add_argument("--leader", choices=DEFAULT_UAV_NAMES, default="drone_0")
    parser.add_argument(
        "--swarm-layout",
        choices=["single", "single_plus_isolated", "two_pairs"],
        default="single",
        help="single keeps the YAML swarm; single_plus_isolated adds drone_4 outside the swarm; two_pairs creates swarms A=(drone_0,drone_1) and B=(drone_2,drone_3).",
    )
    parser.add_argument(
        "--two-swarm-mode",
        choices=["crossing", "parallel", "diverging"],
        default="crossing",
        help="Waypoint geometry used when --swarm-layout two_pairs.",
    )
    parser.add_argument("--spacing", type=parse_spacing, default=[1.0, 1.0, 0.0])
    parser.add_argument("--max-sim-time", type=float, default=None)
    parser.add_argument("--random-waypoints", action="store_true")
    parser.add_argument(
        "--perturbation-targets",
        nargs="+",
        default=["none"],
        help="Targets to perturb. Include 'none' to generate baseline runs.",
    )
    parser.add_argument("--perturbation-axis", choices=["x", "y"], default="y")
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=0.02,
        help="Weak single-axis horizontal force magnitude in newtons.",
    )
    parser.add_argument("--perturbation-start-time", type=float, default=20.0)
    parser.add_argument("--perturbation-interval", type=float, default=0.0)
    parser.add_argument("--perturbation-duration", type=float, default=1.0)
    parser.add_argument("--perturbation-type", default="controlled_push")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    base_config = load_config(args.config)

    if args.plan_csv:
        plan_rows = read_plan(Path(args.plan_csv))
        plan_path = Path(args.plan_csv)
    else:
        start_run_id = args.start_run_id
        if start_run_id is None:
            start_run_id = next_available_run_id(logs_dir)
        plan_rows = generate_plan(args, start_run_id)
        plan_path = Path(args.out_plan)
        write_plan(plan_path, plan_rows)

    execute_plan_rows(
        base_config,
        plan_rows,
        logs_dir=logs_dir,
        plan_path=plan_path,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
