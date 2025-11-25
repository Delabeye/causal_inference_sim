#!/usr/bin/env python3
"""
plot_uav_log.py

Simple log analysis/plot script for the UAV simulation.

Usage examples:
    python analysis/plot_uav_log.py
    python analysis/plot_uav_log.py logs/drone_0_log.csv
    python analysis/plot_uav_log.py --file logs/drone_0_log.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


def load_log(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Log file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError("CSV log must contain a 't' column (time).")
    return df


def has_valid_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    """Return True if all columns exist and are not all NaN."""
    for c in cols:
        if c not in df.columns:
            return False
    all_nan = df[cols].isna().all().all()
    return not all_nan


def plot_trajectories_3d(df: pd.DataFrame):
    """Plot 3D trajectories: true, GPS, EKF (if available)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("3D Trajectories")

    # True trajectory
    if has_valid_columns(df, ["x_true", "y_true", "z_true"]):
        ax.plot(df["x_true"], df["y_true"], df["z_true"], label="True")

    # GPS trajectory
    if has_valid_columns(df, ["x_gps", "y_gps", "z_gps"]):
        ax.plot(df["x_gps"], df["y_gps"], df["z_gps"], linestyle="--", label="GPS")

    # EKF trajectory
    if has_valid_columns(df, ["x_ekf", "y_ekf", "z_ekf"]):
        ax.plot(df["x_ekf"], df["y_ekf"], df["z_ekf"], linestyle=":", label="EKF")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.grid(True)


def plot_position_errors(df: pd.DataFrame):
    """Plot norm of position error for GPS and EKF."""
    t = df["t"].values

    fig, ax = plt.subplots()
    fig.suptitle("Position Error Norms")

    # GPS error
    if has_valid_columns(df, ["x_true", "y_true", "z_true",
                              "x_gps", "y_gps", "z_gps"]):
        dx = df["x_true"] - df["x_gps"]
        dy = df["y_true"] - df["y_gps"]
        dz = df["z_true"] - df["z_gps"]
        err_gps = np.sqrt(dx**2 + dy**2 + dz**2)
        ax.plot(t, err_gps, label="||True - GPS||")

    # EKF error
    if has_valid_columns(df, ["x_true", "y_true", "z_true",
                              "x_ekf", "y_ekf", "z_ekf"]):
        dx = df["x_true"] - df["x_ekf"]
        dy = df["y_true"] - df["y_ekf"]
        dz = df["z_true"] - df["z_ekf"]
        err_ekf = np.sqrt(dx**2 + dy**2 + dz**2)
        ax.plot(t, err_ekf, label="||True - EKF||")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m]")
    ax.grid(True)
    ax.legend()


def plot_distance_to_waypoint(df: pd.DataFrame):
    """Plot distance to current waypoint index over time."""
    if "dist_to_wp" not in df.columns:
        return

    t = df["t"].values
    dist = df["dist_to_wp"].values

    fig, ax = plt.subplots()
    fig.suptitle("Distance to Active Waypoint")
    ax.plot(t, dist)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.grid(True)

    # Overlay waypoint index as a step plot if available
    if "wp_index" in df.columns:
        ax2 = ax.twinx()
        ax2.step(t, df["wp_index"], where="post", alpha=0.5)
        ax2.set_ylabel("Waypoint index")


def plot_control_commands(df: pd.DataFrame):
    """Plot PID acceleration commands over time."""
    if not has_valid_columns(df, ["ax_cmd", "ay_cmd", "az_cmd"]):
        return

    t = df["t"].values

    fig, ax = plt.subplots()
    fig.suptitle("PID Acceleration Commands")

    ax.plot(t, df["ax_cmd"], label="ax_cmd")
    ax.plot(t, df["ay_cmd"], label="ay_cmd")
    ax.plot(t, df["az_cmd"], label="az_cmd")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration command [m/s^2]")
    ax.grid(True)
    ax.legend()


def plot_attitude_commands(df: pd.DataFrame):
    """Plot commanded roll, pitch, yaw over time."""
    if not has_valid_columns(df, ["roll_cmd", "pitch_cmd", "yaw_cmd"]):
        return

    t = df["t"].values

    fig, ax = plt.subplots()
    fig.suptitle("Commanded Attitude (roll/pitch/yaw)")

    ax.plot(t, df["roll_cmd"], label="roll_cmd")
    ax.plot(t, df["pitch_cmd"], label="pitch_cmd")
    ax.plot(t, df["yaw_cmd"], label="yaw_cmd")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [rad]")
    ax.grid(True)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot UAV simulation logs.")
    # Optionnel : --file / -f
    parser.add_argument(
        "--file",
        "-f",
        dest="file",
        type=str,
        default=None,
        help="Path to CSV log file.",
    )
    # Argument positionnel optionnel (pour permettre: python plot_uav_log.py logs/drone_0_log.csv)
    parser.add_argument(
        "file_positional",
        nargs="?",
        default=None,
        help="Path to CSV log file (positional).",
    )

    args = parser.parse_args()

    # Priorité : argument positionnel, puis --file, puis défaut
    if args.file_positional is not None:
        csv_path = args.file_positional
    elif args.file is not None:
        csv_path = args.file
    else:
        csv_path = "logs/drone_0_log.csv"

    df = load_log(csv_path)
    print(f"Loaded log file: {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Create plots
    plot_trajectories_3d(df)
    plot_position_errors(df)
    plot_distance_to_waypoint(df)
    plot_control_commands(df)
    plot_attitude_commands(df)

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()
