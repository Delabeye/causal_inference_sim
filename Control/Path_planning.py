from tracemalloc import start
import numpy as np

from utilities.utilities import euclidean_distance_3d

class PathPlanning:
    def __init__(self):
        pass
    
    def plan_path_lidar(self, start: np.ndarray, goal: np.ndarray,
                    obstacles: list, checkpoints_num: int,
                    min_clearance: float,
                    repel_gain: float = 1.2,
                    max_adjust: float = 2.0) -> list:
        """
        Path planning simplifié avec évitement multi-obstacles basé sur des points LIDAR.
        Approche = interpolation linéaire + répulsion douce + correction cumulative + lissage.

        start : np.array [x,y,z]
        goal  : np.array [x,y,z]
        obstacles : liste de points 3D détectés (x,y,z)
        checkpoints_num : nombre de checkpoints intermédiaires
        min_clearance : distance minimale de sécurité autour des obstacles
        repel_gain : facteur d'intensité de répulsion (1.0 = normal)
        max_adjust : magnitude max de déplacement d’un checkpoint pour stabilité
        """

        path = []
        path.append(start)

    # -------------------------
    # 1. Génération initiale des checkpoints
    # -------------------------
        checkpoints = []
        for i in range(1, checkpoints_num + 1):
            t = i / (checkpoints_num + 1)
            checkpoint = start + t * (goal - start)
            checkpoints.append(checkpoint)

    # -------------------------
    # 2. Ajustement des checkpoints avec répulsion multi-obstacles
    # -------------------------
        corrected_checkpoints = []

        for cp in checkpoints:
            correction = np.zeros(3)

            for obs in obstacles:
                d = np.linalg.norm(cp - obs)

                if d < min_clearance:
                # direction de répulsion
                    direction = (cp - obs) / (d + 1e-6)

                # intensité inversement proportionnelle à la distance
                    repel_strength = repel_gain * (min_clearance - d)

                # contribution au vecteur global de correction
                    correction += direction * repel_strength

        # -------------------------
        # 3. Limite la magnitude (stabilité)
        # -------------------------
            corr_norm = np.linalg.norm(correction)
            if corr_norm > max_adjust:
                correction = (correction / corr_norm) * max_adjust

            new_cp = cp + correction
            corrected_checkpoints.append(new_cp)

    # -------------------------
    # 4. Lissage léger (beaucoup plus stable)
    # -------------------------
        smoothed = []
        alpha = 0.35  # poids du smoothing (0 = pas de lissage, 1 = très lisse)

        smoothed.append(corrected_checkpoints[0])  # premier fixe
        for i in range(1, len(corrected_checkpoints)):
            sm = alpha * corrected_checkpoints[i] + (1 - alpha) * smoothed[i - 1]
            smoothed.append(sm)

    # -------------------------
    # 5. Construction finale du chemin
    # -------------------------
        for c in smoothed:
            path.append(c)

        path.append(goal)

        return path

