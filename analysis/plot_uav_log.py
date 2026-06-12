#!/usr/bin/env python3
"""Plot UAV trajectories for one simulation run from CSV logs."""

import argparse
import csv
import math
import re
from pathlib import Path
import matplotlib.pyplot as plt


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


def plot_run(run_id: int | None = None, logs_dir: str = "logs", save_path: str | None = None, show: bool = True):
    logs_path = Path(logs_dir)
    if run_id is None:
        run_id = latest_run(logs_path)

    data = load_uav_logs(logs_path, run_id)
    print(f"Loaded run {run_id}: {len(data)} drone log(s)")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"UAV trajectories - run {run_id}", fontsize=14)

    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_z = fig.add_subplot(2, 2, 3)
    ax_err = fig.add_subplot(2, 2, 4)

    summary_rows = []

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

    ax_3d.set_title("3D trajectory")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.legend()
    set_axes_equal_3d(ax_3d)

    ax_xy.set_title("XY trajectory")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.axis("equal")
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

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved plot to {output}")

    if summary_rows:
        print("\nTracking error summary")
        print("drone      mean [m]    max [m]")
        for label, mean_err, max_err in summary_rows:
            print(f"{label:<9} {mean_err:>8.3f} {max_err:>10.3f}")

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
    parser.add_argument("--save", help="Optional output image path, for example analysis/run_40_trajectories.png.")
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
            save_path=args.save,
            show=not args.no_show,
        )
