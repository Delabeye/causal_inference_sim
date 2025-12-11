import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.patches import Ellipse
# Import necessary for 3D plotting
from mpl_toolkits.mplot3d import Axes3D 

def draw_covariance_ellipse(ax, mean_x, mean_y, cov_xx, cov_yy, cov_xy, n_std=3.0, **kwargs):
    """Draws a covariance ellipse (2D only)."""
    try:
        if np.isnan([mean_x, mean_y, cov_xx, cov_yy, cov_xy]).any(): return None
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
        return ell
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="logs/drone_0_log.csv")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    df = pd.read_csv(args.file)
    print(f"Loading {len(df)} points.")

    # Propagate GPS to avoid gaps (Forward Fill)
    cols_gps = [c for c in df.columns if 'gps' in c]
    if cols_gps:
        df[cols_gps] = df[cols_gps].ffill()

    # Valid EKF data
    df_ekf = pd.DataFrame()
    if "x_ekf" in df.columns:
        df_ekf = df.dropna(subset=["x_ekf", "y_ekf", "z_ekf"])

    # =========================================================================
    # FIGURE 1: 3D TRAJECTORY (XYZ) - SMOOTHED & VISIBLE
    # =========================================================================
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title("3D Trajectory (XYZ): Ground Truth vs EKF Estimation")

    # 1. Ground Truth (Thick, opaque black line)
    ax1.plot(df["x_true"], df["y_true"], df["z_true"], 'k-', lw=2.5, alpha=1.0, label="Ground Truth")
    
    # Start/End Markers
    ax1.text(df["x_true"].iloc[0], df["y_true"].iloc[0], df["z_true"].iloc[0], " START", color='green', fontweight='bold')
    ax1.text(df["x_true"].iloc[-1], df["y_true"].iloc[-1], df["z_true"].iloc[-1], " END", color='red', fontweight='bold')

    # 2. GPS (Green points, downsampled for readability)
    if "x_gps" in df.columns:
        step_gps = 50
        ax1.scatter(df["x_gps"][::step_gps], df["y_gps"][::step_gps], df["z_gps"][::step_gps], 
                   c='green', marker='x', s=30, alpha=0.4, label="GPS Measurements")

    # 3. SMOOTHED EKF (Blue line)
    if not df_ekf.empty:
        # --- SMOOTHING ---
        # Apply a rolling mean over a window (e.g., 50 points)
        # This removes high-frequency visual noise ("lines in all directions")
        window_size = 50
        ekf_smooth = df_ekf[["x_ekf", "y_ekf", "z_ekf"]].rolling(window=window_size, min_periods=1, center=True).mean()
        
        ax1.plot(ekf_smooth["x_ekf"], ekf_smooth["y_ekf"], ekf_smooth["z_ekf"], 
                 'b-', lw=2.0, alpha=0.8, label=f"EKF")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (Altitude m)")
    ax1.legend()
    
    # Aspect ratio adjustment
    try:
        ax1.set_box_aspect((np.ptp(df["x_true"]), np.ptp(df["y_true"]), np.ptp(df["z_true"])))
    except:
        pass 

    # =========================================================================
    # FIGURE 2: ERRORS (3D Distance)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title("3D Position Error (Euclidean)")
    
    if "x_gps" in df.columns:
        d_g = df.dropna(subset=["x_gps"])
        err_gps = np.sqrt((d_g["x_true"]-d_g["x_gps"])**2 + 
                          (d_g["y_true"]-d_g["y_gps"])**2 + 
                          (d_g["z_true"]-d_g["z_gps"])**2)
        ax2.plot(d_g["t"], err_gps, 'g.', alpha=0.2, label="GPS Error (3D)")
    
    if not df_ekf.empty:
        err_ekf = np.sqrt((df_ekf["x_true"]-df_ekf["x_ekf"])**2 + 
                          (df_ekf["y_true"]-df_ekf["y_ekf"])**2 + 
                          (df_ekf["z_true"]-df_ekf["z_ekf"])**2)
        
        # Smooth the error for readability as well
        err_ekf_smooth = err_ekf.rolling(window=20, min_periods=1).mean()
        
        ax2.plot(df_ekf["t"], err_ekf, 'b-', lw=0.5, alpha=0.2)
        ax2.plot(df_ekf["t"], err_ekf_smooth, 'b-', lw=2, label="EKF Error (3D, Mean)")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")

    # =========================================================================
    # FIGURE 3: COVARIANCE ZOOM (2D XY)
    # =========================================================================
    if not df_ekf.empty and "P_xx" in df.columns:
        fig3 = plt.figure(figsize=(12, 10))
        fig3.suptitle("XY Covariance Zoom (Last 4 instants)")
        gps_var = 0.3**2 
        
        indices = df_ekf.index[-4:]
        for i, idx in enumerate(indices):
            ax = fig3.add_subplot(2, 2, i+1)
            row = df.loc[idx]
            
            ax.plot(row["x_true"], row["y_true"], 'ko', label="Truth")
            
            if not np.isnan(row["x_gps"]):
                ax.plot(row["x_gps"], row["y_gps"], 'gx', label="GPS")
                draw_covariance_ellipse(ax, row["x_gps"], row["y_gps"], gps_var, gps_var, 0.0, 
                                      n_std=3.0, edgecolor='green', linestyle='--', facecolor='none', label="GPS Uncert.")
            
            ax.plot(row["x_ekf"], row["y_ekf"], 'b+', markersize=10, label="EKF")
            draw_covariance_ellipse(ax, row["x_ekf"], row["y_ekf"], 
                                  row["P_xx"], row["P_yy"], row["P_xy"], 
                                  n_std=3.0, edgecolor='blue', facecolor='blue', alpha=0.15, label="EKF Uncert.")
            
            ax.set_title(f"t = {row['t']:.3f} s")
            ax.axis('equal')
            ax.grid(True, linestyle=':')
            
            # Zoom
            cx, cy = row["x_true"], row["y_true"]
            ax.set_xlim(cx - 1.0, cx + 1.0)
            ax.set_ylim(cy - 1.0, cy + 1.0)
            
            if i == 0: ax.legend(loc='best', fontsize='small')

    plt.show()

if __name__ == "__main__":
    main()