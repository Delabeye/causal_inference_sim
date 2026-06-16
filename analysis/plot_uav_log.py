#!/usr/bin/env python3
"""Plot UAV trajectories for one simulation run from CSV logs."""

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


LOG_RE = re.compile(r"run_(?P<run>\d+)_drone_(?P<drone>\d+)\.csv$")
REQUIRED_COLUMNS = {"time", "gt_x", "gt_y", "gt_z"}



def find_run_logs(logs_dir: Path, run_id: int) -> list[tuple[int, Path]]:
    """Return sorted (drone_id, path) pairs for a run."""
    matches = []
    for path in logs_dir.glob(f"run_{run_id}_drone_*.csv"):
        match = LOG_RE.match(path.name)
        if match:
            matches.append((int(match.group("drone")), path))
    return sorted(matches, key=lambda item: item[0])


def available_runs(logs_dir: Path) -> list[int]:
    runs = set()
    for path in logs_dir.glob("run_*_drone_*.csv"):
        match = LOG_RE.match(path.name)
        if match:
            runs.add(int(match.group("run")))
    return sorted(runs)


def latest_run(logs_dir: Path) -> int:
    runs = available_runs(logs_dir)
    if not runs:
        raise FileNotFoundError(f"No run logs found in {logs_dir}")
    return runs[-1]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_uav_logs(logs_dir: Path, run_id: int) -> dict[int, list[dict[str, float]]]:
    data = {}
    for drone_id, path in find_run_logs(logs_dir, run_id):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                print(f"Skipping {path}: empty CSV")
                continue

            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                print(f"Skipping {path}: missing columns {sorted(missing)}")
                continue

            rows = [
                {key: to_float(value) for key, value in row.items()}
                for row in reader
            ]

        rows.sort(key=lambda row: row["time"])
        if rows:
            data[drone_id] = rows

    if not data:
        raise FileNotFoundError(f"No valid logs found for run {run_id} in {logs_dir}")
    return data


def column(rows: list[dict[str, float]], name: str) -> list[float]:
    return [row.get(name, math.nan) for row in rows]


def has_columns(rows: list[dict[str, float]], names: set[str]) -> bool:
    return bool(rows) and names.issubset(rows[0].keys())


def load_waypoints(config_path: str | None) -> dict[int, list[list[float]]]:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        print(f"Waypoint config not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    waypoints = {}
    for agent in config.get("agents", []):
        if agent.get("type") != "uav":
            continue

        match = re.search(r"drone_(\d+)$", str(agent.get("name", "")))
        if not match:
            continue

        drone_id = int(match.group(1))
        wp_list = []
        for waypoint in agent.get("waypoints", []) or []:
            if len(waypoint) >= 3:
                wp_list.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
        if wp_list:
            waypoints[drone_id] = wp_list

    return waypoints


def load_run_waypoints(logs_dir: str, run_id: int) -> dict[int, list[list[float]]]:
    path = Path(logs_dir) / f"run_{run_id}_waypoints.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    waypoints = {}
    for name, wp_list in payload.get("waypoints", {}).items():
        match = re.search(r"drone_(\d+)$", str(name))
        if not match:
            continue
        parsed = []
        for waypoint in wp_list or []:
            if len(waypoint) >= 3:
                parsed.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
        if parsed:
            waypoints[int(match.group(1))] = parsed
    return waypoints


def world_urdf_from_config(config_path: str | None) -> Path | None:
    if config_path is None:
        return None

    path = Path(config_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    world = config.get("world", {})
    world_type = world.get("type", "city")
    if world_type == "generated":
        city = world.get("city", {})
        filename = city.get("filename", "city")
        return Path("assets") / f"{filename}.urdf"
    if world_type == "custom" and world.get("filename"):
        return Path(world["filename"])
    return None


def parse_xyz(value: str | None) -> list[float]:
    if not value:
        return [0.0, 0.0, 0.0]
    parts = value.split()
    if len(parts) < 3:
        return [0.0, 0.0, 0.0]
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def load_buildings(
    config_path: str | None,
    urdf_path: str | None = None,
    min_height: float = 1.0,
) -> list[dict[str, float | str]]:
    path = Path(urdf_path) if urdf_path else world_urdf_from_config(config_path)
    if path is None:
        return []
    if not path.exists():
        print(f"Building URDF not found: {path}")
        return []

    tree = ET.parse(path)
    root = tree.getroot()

    joint_origins = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        origin = joint.find("origin")
        if child is not None and child.get("link"):
            joint_origins[child.get("link")] = parse_xyz(origin.get("xyz") if origin is not None else None)

    buildings = []
    for link in root.findall("link"):
        name = link.get("name", "")
        box = link.find("./collision/geometry/box")
        if box is None:
            box = link.find("./visual/geometry/box")
        if box is None or not box.get("size"):
            continue

        sx, sy, sz = parse_xyz(box.get("size"))
        if sz < min_height:
            continue
        if name in {"world_link", "ground_plane"}:
            continue

        cx, cy, cz = joint_origins.get(name, [0.0, 0.0, sz / 2.0])
        buildings.append(
            {
                "name": name,
                "x": cx,
                "y": cy,
                "z": cz,
                "width": sx,
                "length": sy,
                "height": sz,
            }
        )

    return buildings


def waypoint_reach_summary(
    rows: list[dict[str, float]],
    waypoints: list[list[float]],
    threshold: float,
) -> list[tuple[int, float, bool]]:
    gt_x = column(rows, "gt_x")
    gt_y = column(rows, "gt_y")
    gt_z = column(rows, "gt_z")
    summary = []

    for idx, waypoint in enumerate(waypoints):
        wx, wy, wz = waypoint
        distances = [
            math.sqrt((x - wx) ** 2 + (y - wy) ** 2 + (z - wz) ** 2)
            for x, y, z in zip(gt_x, gt_y, gt_z)
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
        ]
        min_dist = min(distances) if distances else math.nan
        summary.append((idx, min_dist, math.isfinite(min_dist) and min_dist <= threshold))

    return summary


def draw_building_edges(ax, building: dict[str, float | str], color: str = "0.35"):
    x = float(building["x"])
    y = float(building["y"])
    z = float(building["z"])
    width = float(building["width"])
    length = float(building["length"])
    height = float(building["height"])

    xmin, xmax = x - width / 2.0, x + width / 2.0
    ymin, ymax = y - length / 2.0, y + length / 2.0
    zmin, zmax = z - height / 2.0, z + height / 2.0
    corners = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        ax.plot(
            [corners[start][0], corners[end][0]],
            [corners[start][1], corners[end][1]],
            [corners[start][2], corners[end][2]],
            color=color,
            linewidth=0.45,
            alpha=0.35,
        )


def horizontal_clearance_to_building(x: float, y: float, building: dict[str, float | str]) -> float:
    dx = abs(x - float(building["x"])) - float(building["width"]) / 2.0
    dy = abs(y - float(building["y"])) - float(building["length"]) / 2.0
    return math.hypot(max(dx, 0.0), max(dy, 0.0))


def building_clearance_summary(
    rows: list[dict[str, float]],
    buildings: list[dict[str, float | str]],
) -> tuple[str, float] | None:
    if not buildings:
        return None

    best_name = ""
    best_clearance = math.inf
    for row in rows:
        x = row.get("gt_x", math.nan)
        y = row.get("gt_y", math.nan)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        for building in buildings:
            clearance = horizontal_clearance_to_building(x, y, building)
            if clearance < best_clearance:
                best_clearance = clearance
                best_name = str(building["name"])

    if not math.isfinite(best_clearance):
        return None
    return best_name, best_clearance


def set_axes_equal_3d(ax):
    """Make 3D axes use comparable scales."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    centers = [
        0.5 * (x_limits[0] + x_limits[1]),
        0.5 * (y_limits[0] + y_limits[1]),
        0.5 * (z_limits[0] + z_limits[1]),
    ]
    radius = 0.5 * max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([max(0.0, centers[2] - radius), centers[2] + radius])


def plot_run(
    run_id: int | None = None,
    logs_dir: str = "logs",
    save_path: str | None = None,
    save: bool = True,
    show: bool = True,
    config_path: str | None = "config.yaml",
    waypoint_threshold: float = 0.5,
    buildings_urdf: str | None = None,
    show_buildings: bool = True,
    building_min_height: float = 1.0,
):
    logs_path = Path(logs_dir)
    if run_id is None:
        run_id = latest_run(logs_path)

    data = load_uav_logs(logs_path, run_id)
    if config_path is None:
        waypoints_by_drone = {}
    else:
        waypoints_by_drone = load_run_waypoints(logs_dir, run_id) or load_waypoints(config_path)
    buildings = load_buildings(config_path, buildings_urdf, building_min_height) if show_buildings else []
    print(f"Loaded run {run_id}: {len(data)} drone log(s)")
    if buildings:
        print(f"Loaded {len(buildings)} building(s)")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"UAV trajectories - run {run_id}", fontsize=14)

    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_z = fig.add_subplot(2, 2, 3)
    ax_err = fig.add_subplot(2, 2, 4)

    summary_rows = []
    waypoint_rows = []
    building_rows = []

    building_label_added = False
    for building in buildings:
        x = float(building["x"])
        y = float(building["y"])
        width = float(building["width"])
        length = float(building["length"])
        rect = Rectangle(
            (x - width / 2.0, y - length / 2.0),
            width,
            length,
            facecolor="0.35",
            edgecolor="0.15",
            alpha=0.22,
            linewidth=0.6,
            label="buildings" if not building_label_added else None,
        )
        ax_xy.add_patch(rect)
        building_label_added = True

    for color_index, (drone_id, rows) in enumerate(data.items()):
        color = plt.cm.tab10(color_index % 10)
        label = f"drone_{drone_id}"
        time = column(rows, "time")
        gt_x = column(rows, "gt_x")
        gt_y = column(rows, "gt_y")
        gt_z = column(rows, "gt_z")

        ax_3d.plot(gt_x, gt_y, gt_z, color=color, linewidth=1.8, label=label)
        ax_3d.scatter(gt_x[0], gt_y[0], gt_z[0], color=color, marker="o", s=25)
        ax_3d.scatter(gt_x[-1], gt_y[-1], gt_z[-1], color=color, marker="x", s=35)

        ax_xy.plot(gt_x, gt_y, color=color, linewidth=1.8, label=label)
        ax_xy.scatter(gt_x[0], gt_y[0], color=color, marker="o", s=25)
        ax_xy.scatter(gt_x[-1], gt_y[-1], color=color, marker="x", s=35)

        drone_waypoints = waypoints_by_drone.get(drone_id, [])
        if drone_waypoints:
            wp_x = [wp[0] for wp in drone_waypoints]
            wp_y = [wp[1] for wp in drone_waypoints]
            wp_z = [wp[2] for wp in drone_waypoints]
            ax_3d.plot(wp_x, wp_y, wp_z, color=color, linestyle=":", linewidth=1.2, alpha=0.85)
            ax_3d.scatter(wp_x, wp_y, wp_z, color=color, marker="D", s=45, label=f"{label} waypoints")
            ax_xy.plot(wp_x, wp_y, color=color, linestyle=":", linewidth=1.2, alpha=0.85)
            ax_xy.scatter(wp_x, wp_y, color=color, marker="D", s=45)

            for idx, (wx, wy, wz) in enumerate(drone_waypoints):
                ax_3d.text(wx, wy, wz, f"W{idx}", color=color, fontsize=8)
                ax_xy.annotate(f"W{idx}", (wx, wy), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)

            for wp_idx, min_dist, reached in waypoint_reach_summary(rows, drone_waypoints, waypoint_threshold):
                waypoint_rows.append((label, wp_idx, min_dist, reached))

        if has_columns(rows, {"target_x", "target_y"}):
            ax_xy.plot(
                column(rows, "target_x"),
                column(rows, "target_y"),
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )

        ax_z.plot(time, gt_z, color=color, linewidth=1.6, label=label)
        if has_columns(rows, {"target_z"}):
            ax_z.plot(time, column(rows, "target_z"), color=color, linestyle="--", linewidth=0.9, alpha=0.55)

        if has_columns(rows, {"tracking_error_mag"}):
            tracking_error = column(rows, "tracking_error_mag")
        elif has_columns(rows, {"target_x", "target_y", "target_z"}):
            tracking_error = [
                math.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)
                for tx, ty, tz, x, y, z in zip(
                    column(rows, "target_x"),
                    column(rows, "target_y"),
                    column(rows, "target_z"),
                    gt_x,
                    gt_y,
                    gt_z,
                )
            ]
        else:
            tracking_error = None

        if tracking_error is not None:
            ax_err.plot(time, tracking_error, color=color, linewidth=1.6, label=label)
            finite_errors = [err for err in tracking_error if math.isfinite(err)]
            if finite_errors:
                summary_rows.append((label, sum(finite_errors) / len(finite_errors), max(finite_errors)))

        building_clearance = building_clearance_summary(rows, buildings)
        if building_clearance is not None:
            building_rows.append((label, building_clearance[0], building_clearance[1]))

    ax_3d.set_title("3D trajectory")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.legend()
    set_axes_equal_3d(ax_3d)

    ax_xy.set_title("XY trajectory")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, linestyle=":")
    ax_xy.legend()

    ax_z.set_title("Altitude over time")
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("Z [m]")
    ax_z.grid(True, linestyle=":")
    ax_z.legend()

    ax_err.set_title("Tracking error")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [m]")
    ax_err.grid(True, linestyle=":")
    ax_err.legend()

    fig.tight_layout()

    if save:
        output = Path(save_path) if save_path else Path("analysis") / f"run_{run_id}_trajectories.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved plot to {output}")

    if summary_rows:
        print("\nTracking error summary")
        print("drone      mean [m]    max [m]")
        for label, mean_err, max_err in summary_rows:
            print(f"{label:<9} {mean_err:>8.3f} {max_err:>10.3f}")

    if waypoint_rows:
        print(f"\nWaypoint reach summary (threshold={waypoint_threshold:.2f} m)")
        print("drone      waypoint  min_dist [m]  reached")
        for label, wp_idx, min_dist, reached in waypoint_rows:
            status = "yes" if reached else "no"
            print(f"{label:<9} W{wp_idx:<7} {min_dist:>11.3f}  {status}")

    if building_rows:
        print("\nBuilding clearance summary")
        print("drone      closest_building  min_xy_clearance [m]")
        for label, building_name, clearance in building_rows:
            print(f"{label:<9} {building_name:<16} {clearance:>18.3f}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot all drone trajectories for one run from logs/run_<id>_drone_<id>.csv"
    )
    parser.add_argument("run", nargs="?", type=int, help="Run id to plot. Defaults to the latest run.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing CSV logs.")
    parser.add_argument("--config", default="config.yaml", help="YAML config used to draw UAV waypoints.")
    parser.add_argument("--no-waypoints", action="store_true", help="Do not draw configured waypoints.")
    parser.add_argument("--waypoint-threshold", type=float, default=0.5, help="Distance threshold in meters for waypoint reached summary.")
    parser.add_argument("--buildings-urdf", help="Optional URDF path used to draw buildings. Defaults to the world URDF from --config.")
    parser.add_argument("--no-buildings", action="store_true", help="Do not draw buildings from the world URDF.")
    parser.add_argument("--building-min-height", type=float, default=1.0, help="Minimum box height in meters to treat an URDF link as a building.")
    parser.add_argument("--save", help="Optional output image path, for example analysis/run_40_trajectories.png.")
    parser.add_argument("--no-save", action="store_true", help="Do not save the plot image.")
    parser.add_argument("--no-show", action="store_true", help="Save/compute without opening a plot window.")
    parser.add_argument("--list-runs", action="store_true", help="List available run ids and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_runs:
        print("Available runs:", " ".join(map(str, available_runs(Path(args.logs_dir)))))
    else:
        plot_run(
            run_id=args.run,
            logs_dir=args.logs_dir,
            save_path=None if args.no_save else args.save,
            save=not args.no_save,
            show=not args.no_show,
            config_path=None if args.no_waypoints else args.config,
            waypoint_threshold=args.waypoint_threshold,
            buildings_urdf=args.buildings_urdf,
            show_buildings=not args.no_buildings,
            building_min_height=args.building_min_height,
        )
