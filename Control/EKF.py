import numpy as np
import pybullet as p

class GPSEKF:
    """
    EKF Fusion GPS + IMU utilisant la Covariance Intersection (CI).
    
    - Prédiction : Modèle dynamique piloté par l'IMU (accéléromètre).
    - Correction : Fusion robuste (CI) avec le GPS.
    """

    def __init__(self, dt: float,
                 q_pos: float = 0.05,
                 q_vel: float = 0.10,
                 r_pos: float = 0.1,
                 r_vel: float = 0.05,
                 accel_noise_std: float = 0.2) -> None:
        self.dt = float(dt)

        # État : [px, py, pz, vx, vy, vz]
        self.x = np.zeros(6, dtype=float)
        
        # Incertitude initiale
        self.P = np.eye(6, dtype=float) * 1.0

        # Paramètres de bruit
        self.accel_noise_std = float(accel_noise_std)
        
        # Matrice Q de base (petite valeur pour la stabilité numérique)
        self.Q = np.eye(6) * 1e-6

        # Matrice R (Incertitude GPS)
        self.R = np.diag(
            [r_pos ** 2, r_pos ** 2, r_pos ** 2,
             r_vel ** 2, r_vel ** 2, r_vel ** 2]
        )
        
        # Pré-calcul de l'inverse de R pour la CI (optimisation)
        self.R_inv = np.linalg.inv(self.R)

        # Matrice de transition F (partie cinématique simple)
        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self._initialized = False
        # Vecteur gravité (Z vers le haut)
        self.g_vector = np.array([0, 0, 9.81])

    def _init_state(self, meas_pos, meas_vel) -> None:
        """Initialise l'état sur la première mesure GPS."""
        self.x[0:3] = meas_pos
        self.x[3:6] = meas_vel
        # On initialise la covariance avec celle du GPS
        self.P = self.R.copy()
        self._initialized = True

    def predict(self, imu_accel=None, orientation_q=None) -> None:
        """
        Prédiction basée sur l'IMU.
        imu_accel : [ax, ay, az] (m/s^2) force spécifique (Body Frame)
        orientation_q : [x, y, z, w] quaternion du drone
        """
        dt = self.dt
        
        # Si pas d'IMU, repli sur modèle vitesse constante
        if imu_accel is None or orientation_q is None:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q * 100 
            return

        # 1. Rotation de l'accélération du repère Body vers World
        rot_mat = np.array(p.getMatrixFromQuaternion(orientation_q)).reshape(3, 3)
        
        # Accélération cinématique = Rot(f_imu) - g
        # (Note: sensor.py simule f = a - g, donc a = f + g. 
        # Mais attention aux conventions de signe. Ici on suppose f_imu contient la réaction +g quand posé).
        # Ajustement standard : Acc_Monde = Rot * Acc_Body - Gravité
        acc_world = rot_mat @ np.array(imu_accel) - self.g_vector

        # 2. Propagation de l'état (Lois de Newton)
        # Position += v*dt + 0.5*a*dt^2
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc_world * dt**2
        # Vitesse += a*dt
        self.x[3:6] += acc_world * dt

        # 3. Propagation de la covariance (Q basée sur bruit accéléro)
        G = np.zeros((6, 3))
        G[0:3, :] = 0.5 * dt**2 * np.eye(3)
        G[3:6, :] = dt * np.eye(3)
        
        # Matrice de bruit injectée par l'IMU
        Q_imu = G @ (self.accel_noise_std**2 * np.eye(3)) @ G.T
        
        self.P = self.F @ self.P @ self.F.T + Q_imu + self.Q

    def _optimize_omega(self, P_inv, R_inv):
        """Trouve le omega optimal (0..1) minimisant la trace."""
        best_omega = 0.5
        min_trace = float('inf')
        
        # Grille de recherche simple
        for omega in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            # P_ci_inv = w * P_inv + (1-w) * R_inv
            P_ci_inv_candidate = omega * P_inv + (1 - omega) * R_inv
            
            try:
                # On inverse pour avoir P_ci et on regarde sa trace
                P_ci_candidate = np.linalg.inv(P_ci_inv_candidate)
                tr = np.trace(P_ci_candidate)
                if tr < min_trace:
                    min_trace = tr
                    best_omega = omega
            except np.linalg.LinAlgError:
                continue
                
        return best_omega

    def update(self, z) -> None:
        """Mise à jour via Covariance Intersection (CI)."""
        z = np.asarray(z, dtype=float).reshape(6)

        # 1. Inversion de la covariance actuelle (Information Matrix)
        try:
            P_inv = np.linalg.inv(self.P)
        except np.linalg.LinAlgError:
            P_inv = np.eye(6) * 1e3

        # 2. Optimisation de Omega
        omega = self._optimize_omega(P_inv, self.R_inv)

        # 3. Fusion
        P_ci_inv = omega * P_inv + (1 - omega) * self.R_inv
        self.P = np.linalg.inv(P_ci_inv)

        weighted_state = omega * (P_inv @ self.x) + (1 - omega) * (self.R_inv @ z)
        self.x = self.P @ weighted_state

    def step(self, meas_pos, meas_vel, imu_accel=None, orientation_q=None):
        meas_pos = np.asarray(meas_pos, dtype=float).reshape(3)
        meas_vel = np.asarray(meas_vel, dtype=float).reshape(3)

        if not self._initialized:
            self._init_state(meas_pos, meas_vel)

        # Prédiction (IMU)
        self.predict(imu_accel, orientation_q)

        # Correction (CI avec GPS)
        z = np.concatenate([meas_pos, meas_vel])
        self.update(z)

        return self.x[0:3].copy(), self.x[3:6].copy()