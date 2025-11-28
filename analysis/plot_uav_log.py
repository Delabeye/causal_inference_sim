import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.patches import Ellipse
# Import nécessaire pour la 3D
from mpl_toolkits.mplot3d import Axes3D 

def draw_covariance_ellipse(ax, mean_x, mean_y, cov_xx, cov_yy, cov_xy, n_std=3.0, **kwargs):
    """Dessine une ellipse de covariance (2D uniquement)."""
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
        print(f"Fichier introuvable: {args.file}")
        return

    df = pd.read_csv(args.file)
    print(f"Chargement de {len(df)} points.")

    # Propagation GPS pour éviter les trous
    cols_gps = [c for c in df.columns if 'gps' in c]
    if cols_gps:
        df[cols_gps] = df[cols_gps].ffill()

    # Données EKF valides
    df_ekf = pd.DataFrame()
    if "x_ekf" in df.columns:
        df_ekf = df.dropna(subset=["x_ekf", "y_ekf", "z_ekf"])

    # =========================================================================
    # FIGURE 1 : TRAJECTOIRE 3D (XYZ) - LISSÉE
    # =========================================================================
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title("Trajectoire 3D (XYZ) : Vérité vs Estimation EKF (Lissée)")

    # 1. Vérité Terrain (Ligne noire fine)
    ax1.plot(df["x_true"], df["y_true"], df["z_true"], 'k-', lw=1, alpha=0.5, label="Vérité")
    
    # Marqueurs Start/End
    ax1.text(df["x_true"].iloc[0], df["y_true"].iloc[0], df["z_true"].iloc[0], " START", color='green', fontweight='bold')
    ax1.text(df["x_true"].iloc[-1], df["y_true"].iloc[-1], df["z_true"].iloc[-1], " END", color='red', fontweight='bold')

    # 2. GPS (Points verts, sous-échantillonnés pour la lisibilité)
    if "x_gps" in df.columns:
        # On affiche 1 point sur 50 pour ne pas noyer le graphique 3D
        step_gps = 50
        ax1.scatter(df["x_gps"][::step_gps], df["y_gps"][::step_gps], df["z_gps"][::step_gps], 
                   c='green', marker='x', s=20, alpha=0.3, label="Mesures GPS")

    # 3. EKF LISSÉ (Ligne bleue)
    if not df_ekf.empty:
        # --- LISSAGE ---
        # On applique une moyenne glissante sur une fenêtre (ex: 50 points ~ 0.2s)
        # cela supprime le bruit haute fréquence visuel ("traits dans tous les sens")
        window_size = 50
        ekf_smooth = df_ekf[["x_ekf", "y_ekf", "z_ekf"]].rolling(window=window_size, min_periods=1, center=True).mean()
        
        ax1.plot(ekf_smooth["x_ekf"], ekf_smooth["y_ekf"], ekf_smooth["z_ekf"], 
                 'b-', lw=2.5, label=f"EKF (Lissé)")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (Altitude m)")
    ax1.legend()
    
    # Ajustement des axes pour ratio réaliste
    try:
        ax1.set_box_aspect((np.ptp(df["x_true"]), np.ptp(df["y_true"]), np.ptp(df["z_true"])))
    except:
        pass 

    # =========================================================================
    # FIGURE 2 : ERREURS (3D Distance)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title("Erreur de Position 3D (Euclidienne)")
    
    if "x_gps" in df.columns:
        d_g = df.dropna(subset=["x_gps"])
        err_gps = np.sqrt((d_g["x_true"]-d_g["x_gps"])**2 + 
                          (d_g["y_true"]-d_g["y_gps"])**2 + 
                          (d_g["z_true"]-d_g["z_gps"])**2)
        ax2.plot(d_g["t"], err_gps, 'g.', alpha=0.2, label="Err GPS 3D")
    
    if not df_ekf.empty:
        # On trace l'erreur brute (non lissée) pour voir la vraie performance, 
        # ou l'erreur lissée si on préfère. Ici erreur brute pour honnêteté scientifique.
        err_ekf = np.sqrt((df_ekf["x_true"]-df_ekf["x_ekf"])**2 + 
                          (df_ekf["y_true"]-df_ekf["y_ekf"])**2 + 
                          (df_ekf["z_true"]-df_ekf["z_ekf"])**2)
        
        # Lissage visuel de l'erreur aussi pour la lisibilité
        err_ekf_smooth = err_ekf.rolling(window=20, min_periods=1).mean()
        
        ax2.plot(df_ekf["t"], err_ekf, 'b-', lw=0.5, alpha=0.3) # Brute légère
        ax2.plot(df_ekf["t"], err_ekf_smooth, 'b-', lw=2, label="Err EKF 3D (Moyenne)")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Erreur (m)")

    # =========================================================================
    # FIGURE 3 : ZOOM COVARIANCE (2D XY) - inchangé (car 2D est lisible)
    # =========================================================================
    if not df_ekf.empty and "P_xx" in df.columns:
        fig3 = plt.figure(figsize=(12, 10))
        fig3.suptitle("Zoom Covariance XY (4 derniers instants)")
        gps_var = 0.3**2 
        
        indices = df_ekf.index[-4:]
        for i, idx in enumerate(indices):
            ax = fig3.add_subplot(2, 2, i+1)
            row = df.loc[idx]
            
            ax.plot(row["x_true"], row["y_true"], 'ko', label="Vrai")
            
            if not np.isnan(row["x_gps"]):
                ax.plot(row["x_gps"], row["y_gps"], 'gx', label="GPS")
                draw_covariance_ellipse(ax, row["x_gps"], row["y_gps"], gps_var, gps_var, 0.0, 
                                      n_std=3.0, edgecolor='green', linestyle='--', facecolor='none')
            
            ax.plot(row["x_ekf"], row["y_ekf"], 'b+', markersize=10, label="EKF")
            draw_covariance_ellipse(ax, row["x_ekf"], row["y_ekf"], 
                                  row["P_xx"], row["P_yy"], row["P_xy"], 
                                  n_std=3.0, edgecolor='blue', facecolor='blue', alpha=0.15)
            
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