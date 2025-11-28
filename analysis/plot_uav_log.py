import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.patches import Ellipse

def draw_covariance_ellipse(ax, mean_x, mean_y, cov_xx, cov_yy, cov_xy, n_std=3.0, color='red', alpha=0.3):
    """
    Dessine une ellipse de covariance à n_std sigma.
    """
    # Construction de la matrice de covariance 2x2
    cov = np.array([[cov_xx, cov_xy], 
                    [cov_xy, cov_yy]])
    
    # Valeurs propres et vecteurs propres
    vals, vecs = np.linalg.eigh(cov)
    
    # L'ordre des eigenvalues donne l'angle et la taille
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Angle en degrés
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Largeur et hauteur (diamètre = 2 * n_std * écart-type)
    # Ecart-type = sqrt(valeur propre)
    width, height = 2 * n_std * np.sqrt(vals)
    
    ell = Ellipse(xy=(mean_x, mean_y),
                  width=width, height=height,
                  angle=theta, color=color, alpha=alpha)
    ax.add_patch(ell)


def main():
    parser = argparse.ArgumentParser(description="Trace les logs du drone avec Ellipses EKF.")
    parser.add_argument("--file", type=str, default="logs/drone_0_log.csv", help="Chemin du CSV")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Fichier introuvable : {args.file}")
        return

    df = pd.read_csv(args.file)
    print(f"Chargement de {len(df)} lignes.")

    # Nettoyage des données (supprimer les lignes sans EKF si besoin)
    # df = df.dropna(subset=['x_ekf', 'P_xx'])

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- PLOT 1 : TRAJECTOIRE X-Y ---
    ax_traj = axs[0]
    ax_traj.set_title("Trajectoire 2D : Vérité vs GPS vs EKF")
    
    # Vérité Terrain
    ax_traj.plot(df["x_true"], df["y_true"], 'k-', linewidth=2, label="Vérité Terrain")
    
    # GPS (points)
    ax_traj.plot(df["x_gps"], df["y_gps"], 'g.', markersize=2, alpha=0.3, label="GPS (Mesure)")
    
    # EKF
    ax_traj.plot(df["x_ekf"], df["y_ekf"], 'b--', linewidth=1.5, label="EKF (Fusion)")

    # Dessin des ellipses EKF
    # On ne dessine pas à chaque pas de temps pour ne pas saturer le graphe
    step = 50 # Dessine une ellipse toutes les 50 frames (~0.2s)
    
    if "P_xx" in df.columns:
        indices = range(0, len(df), step)
        first_ellipse = True
        for i in indices:
            row = df.iloc[i]
            if np.isnan(row["x_ekf"]) or np.isnan(row["P_xx"]):
                continue
                
            label = "Incertitude 3$\sigma$" if first_ellipse else None
            draw_covariance_ellipse(
                ax_traj, 
                row["x_ekf"], row["y_ekf"], 
                row["P_xx"], row["P_yy"], row["P_xy"], 
                n_std=3.0, 
                color='blue', 
                alpha=0.1
            )
            first_ellipse = False
            # Point central de l'ellipse
            # ax_traj.plot(row["x_ekf"], row["y_ekf"], 'b+', markersize=5, alpha=0.5)

    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.axis('equal')
    ax_traj.grid(True)

    # --- PLOT 2 : ERREURS ---
    ax_err = axs[1]
    ax_err.set_title("Erreur de Position dans le temps")
    
    time = df["t"]
    
    # Calcul des erreurs distance euclidienne
    err_gps = np.sqrt((df["x_true"] - df["x_gps"])**2 + (df["y_true"] - df["y_gps"])**2)
    err_ekf = np.sqrt((df["x_true"] - df["x_ekf"])**2 + (df["y_true"] - df["y_ekf"])**2)
    
    ax_err.plot(time, err_gps, 'g-', alpha=0.4, label="Erreur GPS")
    ax_err.plot(time, err_ekf, 'b-', linewidth=2, label="Erreur EKF")
    
    ax_err.set_xlabel("Temps (s)")
    ax_err.set_ylabel("Erreur (m)")
    ax_err.legend()
    ax_err.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()