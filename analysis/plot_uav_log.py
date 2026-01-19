#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

def draw_covariance_ellipse(ax, mean_x, mean_y, cov_xx, cov_yy, cov_xy, n_std=3.0, **kwargs):
    """
    Draws a covariance ellipse centered at (mean_x, mean_y).
    """
    try:
        if np.isnan([mean_x, mean_y, cov_xx, cov_yy, cov_xy]).any(): return
        
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width = 2 * n_std * np.sqrt(np.maximum(vals, 0)[0])
        height = 2 * n_std * np.sqrt(np.maximum(vals, 0)[1])
        
        ell = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=theta, **kwargs)
        ax.add_patch(ell)
    except Exception as e:
        pass

def plot_log(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return

    print(f"Loading log: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return

    required_cols = ['x_true', 'x_gps', 'x_ekf', 't']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns! Found: {list(df.columns)}")
        return

    # Fill NaN
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    t = df['t']

    # --- SMOOTHING EKF (Rolling Mean) ---
    window_size = 15
    df['x_ekf_smooth'] = df['x_ekf'].rolling(window=window_size, min_periods=1, center=True).mean()
    df['y_ekf_smooth'] = df['y_ekf'].rolling(window=window_size, min_periods=1, center=True).mean()
    df['z_ekf_smooth'] = df['z_ekf'].rolling(window=window_size, min_periods=1, center=True).mean()

    # =========================================================================
    # FIGURE 1: 3D TRAJECTORY
    # =========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Trajectory (XYZ)")

    ax.plot(df['x_true'], df['y_true'], df['z_true'], 'k-', linewidth=2, label="Ground Truth")
    
    step = 5 if len(df) > 500 else 1
    ax.scatter(df['x_gps'][::step], df['y_gps'][::step], df['z_gps'][::step], 
               c='green', marker='x', s=15, alpha=0.3, label="GPS Measurements")
    
    ax.plot(df['x_ekf_smooth'], df['y_ekf_smooth'], df['z_ekf_smooth'], 'b--', linewidth=2, label="EKF (Smooth)")
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    # =========================================================================
    # FIGURE 2: 2D TRAJECTORY & ERROR (Clean)
    # =========================================================================
    fig2, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: XY Plane ---
    ax_xy.plot(df['x_true'], df['y_true'], 'k-', label="Ground Truth")
    ax_xy.scatter(df['x_gps'], df['y_gps'], s=5, c='green', alpha=0.2, marker='x', label="GPS")
    ax_xy.plot(df['x_ekf_smooth'], df['y_ekf_smooth'], 'b--', linewidth=1.5, label="EKF Smooth")
    
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("XY Trajectory")
    ax_xy.axis("equal")
    ax_xy.grid(True)
    ax_xy.legend()

    # --- Plot 2: Position Error ---
    err_gps = np.sqrt((df['x_gps'] - df['x_true'])**2 + (df['y_gps'] - df['y_true'])**2 + (df['z_gps'] - df['z_true'])**2)
    err_ekf = np.sqrt((df['x_ekf_smooth'] - df['x_true'])**2 + (df['y_ekf_smooth'] - df['y_true'])**2 + (df['z_ekf_smooth'] - df['z_true'])**2)
    
    ax_err.plot(t, err_gps, color='green', alpha=0.2, label="GPS Error (Raw)")
    ax_err.plot(t, err_ekf, color='blue', linewidth=2, label="EKF Error (Smooth)")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("3D Position Error [m]")
    ax_err.set_title("Estimation Error")
    ax_err.grid(True)
    ax_err.legend()

    # =========================================================================
    # FIGURE 3: COVARIANCE ZOOM (2D XY) - Separate Figure
    # =========================================================================
    if "P_xx" in df.columns:
        fig3 = plt.figure(figsize=(12, 10))
        fig3.suptitle("XY Covariance Zoom (Last 4 instants)")
        gps_var = 2.0**2  # Visual constant matching new EKF noise (2.0m)
       
        # Select last 4 valid points
        indices = df.index[-4:]
        
        for i, idx in enumerate(indices):
            ax = fig3.add_subplot(2, 2, i+1)
            row = df.loc[idx]
           
            # Ground Truth
            ax.plot(row["x_true"], row["y_true"], 'ko', label="Truth")
           
            # GPS + Ellipse
            if not np.isnan(row["x_gps"]):
                ax.plot(row["x_gps"], row["y_gps"], 'gx', label="GPS")
                draw_covariance_ellipse(ax, row["x_gps"], row["y_gps"], gps_var, gps_var, 0.0,
                                        n_std=2.0, edgecolor='green', linestyle='--', facecolor='none', label="GPS Uncert.")
           
            # EKF (Raw) + Ellipse
            ax.plot(row["x_ekf"], row["y_ekf"], 'b+', markersize=10, label="EKF")
            draw_covariance_ellipse(ax, row["x_ekf"], row["y_ekf"],
                                    row["P_xx"], row["P_yy"], row["P_xy"],
                                    n_std=2.0, edgecolor='blue', facecolor='blue', alpha=0.15, label="EKF Uncert.")
           
            ax.set_title(f"t = {row['t']:.3f} s")
            ax.axis('equal')
            ax.grid(True, linestyle=':')
           
            # Zoom logic based on uncertainty
            cx, cy = row["x_true"], row["y_true"]
            ax.set_xlim(cx - 5.0, cx + 5.0)
            ax.set_ylim(cy - 5.0, cy + 5.0)
           
            if i == 0: ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

    # --- METRICS ---
    rmse_gps = np.sqrt(np.mean(err_gps**2))
    rmse_ekf = np.sqrt(np.mean(err_ekf**2))
    print("=" * 40)
    print(f" PERFORMANCE REPORT")
    print("=" * 40)
    print(f" GPS RMSE (Raw)    : {rmse_gps:.4f} m")
    print(f" EKF RMSE (Smooth) : {rmse_ekf:.4f} m")
    if rmse_gps > 0:
        improvement = (1 - rmse_ekf/rmse_gps) * 100
        print(f" Improvement       : {improvement:.1f}%")
    print("=" * 40)

if __name__ == "__main__":
    default_log = os.path.join("logs", "drone_0.csv")
    log_path = sys.argv[1] if len(sys.argv) > 1 else default_log
    plot_log(log_path)

