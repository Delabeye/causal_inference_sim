"""Controlled perturbation suite (GNSS / wind).

Creates separate runs so you can attribute failures to a single cause:
- baseline
- gnss_only (swarm only)
- wind_only (swarm only)
- gnss_wind (swarm only)

Explicit scenario:
- swarm = {drone_0, drone_1, drone_2}
- independent = {drone_3}
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import socket

import numpy as np

from simulator.simulator_manager import SimulationManager
from utilities.config import load_config, save_config


SWARM = {"drone_0", "drone_1", "drone_2"}
INDEPENDENT = {"drone_3"}


def _parse_vec3(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_uav_agents(cfg: Dict) -> List[Dict]:
    agents = cfg.get("agents", [])
    return [a for a in agents if isinstance(a, dict) and a.get("type") == "uav"]


def _apply_connect_mode(cfg: Dict, connect_mode: str) -> None:
    cfg.setdefault("simulation", {})
    cfg["simulation"]["connect_mode"] = connect_mode


def _apply_log_dir(cfg: Dict, log_dir: str) -> None:
    cfg.setdefault("simulation", {})
    cfg["simulation"]["log_dir"] = log_dir


def _apply_time(cfg: Dict, max_sim_time: Optional[float]) -> None:
    if max_sim_time is None:
        return
    cfg.setdefault("simulation", {})
    cfg["simulation"]["max_sim_time"] = float(max_sim_time)


def _find_free_port() -> int:
    """Ask the OS for a free TCP port (best effort)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _apply_ports(cfg: Dict, port_in: int, port_out: int, radar_port: int) -> None:
    """Set swarm proxy ports + radar port, and propagate to UAV radar subscriptions."""
    # Swarm proxy ports
    if isinstance(cfg.get("swarm"), list):
        for s in cfg["swarm"]:
            if isinstance(s, dict):
                s["port_in"] = int(port_in)
                s["port_out"] = int(port_out)
                s.setdefault("ip", "localhost")

    # Radar station port + UAV radar list
    for agent in cfg.get("agents", []) or []:
        if not isinstance(agent, dict):
            continue
        if agent.get("type") == "radar":
            agent["port_out"] = int(radar_port)
            agent.setdefault("ip", "localhost")
        elif agent.get("type") == "uav":
            rad_list = agent.get("radar")
            if isinstance(rad_list, list):
                for r in rad_list:
                    if isinstance(r, dict):
                        r["port"] = int(radar_port)
                        r.setdefault("ip", "localhost")


def _apply_gnss_noise(cfg: Dict, target_names: Sequence[str], pos_std: float, vel_std: float) -> None:
    for agent in _get_uav_agents(cfg):
        name = agent.get("name")
        if name not in target_names:
            continue
        agent.setdefault("sensors", {})
        agent["sensors"].setdefault("gnss", {})
        agent["sensors"]["gnss"]["position_noise_std"] = float(pos_std)
        agent["sensors"]["gnss"]["velocity_noise_std"] = float(vel_std)


def _apply_wind(cfg: Dict, target_names: Sequence[str], wind_mean: List[float], turbulence: float) -> None:
    for agent in _get_uav_agents(cfg):
        name = agent.get("name")
        if name not in target_names:
            continue
        # Preferred structure in YAML
        agent.setdefault("wind", {})
        agent["wind"]["wind_mean"] = list(wind_mean)
        agent["wind"]["turbulence"] = float(turbulence)
        # Backward-compatible flat keys (UAV supports both)
        agent["wind_mean"] = list(wind_mean)
        agent["turbulence"] = float(turbulence)


def _run_one(cfg: Dict) -> None:
    sim = SimulationManager(cfg)
    try:
        sim.run()
    finally:
        sim.stop()


def build_conditions(
    base_cfg: Dict,
    gnss_pos_std: float,
    gnss_vel_std: float,
    wind_mean: List[float],
    wind_turbulence: float,
) -> List[Tuple[str, Dict]]:
    """Return a list of (condition_name, cfg) for the perturbation suite."""
    conditions: List[Tuple[str, Dict]] = []

    # Baseline
    conditions.append(("baseline", copy.deepcopy(base_cfg)))

    # GNSS-only on swarm
    c = copy.deepcopy(base_cfg)
    _apply_gnss_noise(c, sorted(SWARM), gnss_pos_std, gnss_vel_std)
    conditions.append(("gnss_only_swarm", c))

    # Wind-only on swarm
    c = copy.deepcopy(base_cfg)
    _apply_wind(c, sorted(SWARM), wind_mean, wind_turbulence)
    conditions.append(("wind_only_swarm", c))

    # GNSS+Wind on swarm
    c = copy.deepcopy(base_cfg)
    _apply_gnss_noise(c, sorted(SWARM), gnss_pos_std, gnss_vel_std)
    _apply_wind(c, sorted(SWARM), wind_mean, wind_turbulence)
    conditions.append(("gnss_wind_swarm", c))

    return conditions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", default="config.yaml")
    p.add_argument("--out_dir", default="experiments_out")
    p.add_argument("--connect_mode", default="direct", choices=["direct", "gui"])
    p.add_argument("--seeds", default="0", help="Comma-separated list of seeds, e.g. 0,1,2")
    p.add_argument("--max_sim_time", type=float, default=None)

    p.add_argument(
        "--fixed_ports",
        action="store_true",
        help="If set, keep ports from config.yaml (no per-run randomization).",
    )

    p.add_argument("--gnss_pos_std", type=float, default=0.5, help="GNSS position noise std (m) for GNSS-only")
    p.add_argument("--gnss_vel_std", type=float, default=0.25, help="GNSS velocity noise std (m/s) for GNSS-only")

    p.add_argument("--wind_mean", type=str, default="2,0,0", help="Wind mean vec3 as 'x,y,z' (m/s)")
    p.add_argument("--wind_turbulence", type=float, default=20.0)

    p.add_argument(
        "--run_causal_analysis",
        action="store_true",
        help="If set, run analysis/causal_analysis.py after each simulation",
    )
    p.add_argument(
        "--proximity_radius",
        type=float,
        default=2.0,
        help="Only used if --run_causal_analysis is set",
    )

    args = p.parse_args()

    base_cfg = load_config(args.base_config)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    wind_mean = _parse_vec3(args.wind_mean)

    conditions = build_conditions(
        base_cfg,
        gnss_pos_std=args.gnss_pos_std,
        gnss_vel_std=args.gnss_vel_std,
        wind_mean=wind_mean,
        wind_turbulence=args.wind_turbulence,
    )

    manifest = {
        "swarm": sorted(SWARM),
        "independent": sorted(INDEPENDENT),
        "conditions": [name for name, _ in conditions],
        "seeds": seeds,
        "gnss_pos_std": args.gnss_pos_std,
        "gnss_vel_std": args.gnss_vel_std,
        "wind_mean": wind_mean,
        "wind_turbulence": args.wind_turbulence,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_idx = 0
    for seed in seeds:
        for cond_name, cfg in conditions:
            run_dir = out_root / f"{cond_name}" / f"seed_{seed:03d}"
            log_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            cfg = copy.deepcopy(cfg)
            _apply_connect_mode(cfg, args.connect_mode)
            _apply_log_dir(cfg, str(log_dir))
            _apply_time(cfg, args.max_sim_time)
            cfg.setdefault("simulation", {})
            cfg["simulation"]["seed"] = seed

            # IMPORTANT (Windows/ZMQ): randomize ports per run to avoid collisions
            # when running multiple simulations in the same Python process.
            if not args.fixed_ports:
                # Best effort: ask OS for free ports (not reserved).
                # Keep them distinct.
                p1 = _find_free_port()
                p2 = _find_free_port()
                while p2 == p1:
                    p2 = _find_free_port()
                pr = _find_free_port()
                while pr in (p1, p2):
                    pr = _find_free_port()
                _apply_ports(cfg, port_in=p1, port_out=p2, radar_port=pr)
                cfg.setdefault("simulation", {})
                cfg["simulation"]["swarm_port_in"] = int(p1)
                cfg["simulation"]["swarm_port_out"] = int(p2)
                cfg["simulation"]["radar_port_out"] = int(pr)
            run_idx += 1

            save_config(cfg, str(run_dir / "config_used.yaml"))

            meta = {
                "condition": cond_name,
                "seed": seed,
                "log_dir": str(log_dir),
            }
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            print(f"\n[RUN] {cond_name} seed={seed} -> {run_dir}")
            _set_seed(seed)
            _run_one(cfg)

            if args.run_causal_analysis:
                causal_dir = run_dir / "causal_out"
                causal_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "analysis/causal_analysis.py",
                    "--log_dir",
                    str(log_dir),
                    "--output_dir",
                    str(causal_dir),
                    "--swarm_indices",
                    "0,1,2",
                    "--independent_indices",
                    "3",
                    "--proximity_radius",
                    str(args.proximity_radius),
                ]
                print("[ANALYSIS]", " ".join(cmd))
                subprocess.run(cmd, check=False)

    print(f"\nAll runs finished. Outputs in: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
