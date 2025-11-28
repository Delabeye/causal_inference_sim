import numpy as np
import pybullet as p

class GPSEKF:
    """
    EKF avec Fusion IMU (Prédiction) + GPS (Correction).
    Version stable avec fusion simplifiée (Covariance Intersection pondérée fixe).
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
        
        # Covariance initiale
        self.P = np.eye(6, dtype=float) * 1.0

        # Paramètres de bruit
        self.accel_noise_std = float(accel_noise_std)
        self.Q = np.eye(6) * 1e-6

        self.R = np.diag(
            [r_pos ** 2, r_pos ** 2, r_pos ** 2,
             r_vel ** 2, r_vel ** 2, r_vel ** 2]
        )
        
        # Pré-calcul de l'inverse de R
        try:
            self.R_inv = np.linalg.inv(self.R)
        except np.linalg.LinAlgError:
            self.R_inv = np.eye(6)

        # Matrice de transition F
        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self._initialized = False
        self.g_vector = np.array([0, 0, 9.81])

    def _init_state(self, meas_pos, meas_vel) -> None:
        """Initialise l'état sur la première mesure GPS."""
        if np.isnan(meas_pos).any() or np.isnan(meas_vel).any():
            return
        self.x[0:3] = meas_pos
        self.x[3:6] = meas_vel
        self.P = self.R.copy()
        self._initialized = True

    def predict(self, imu_accel=None, orientation_q=None) -> None:
        """Prédiction basée sur l'IMU."""
        dt = self.dt
        
        if imu_accel is None or orientation_q is None or np.isnan(imu_accel).any():
            # Modèle vitesse constante si pas d'IMU
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q 
            return

        # Rotation Body -> World
        rot_mat = np.array(p.getMatrixFromQuaternion(orientation_q)).reshape(3, 3)
        acc_world = rot_mat @ np.array(imu_accel) - self.g_vector

        # Propagation État
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc_world * dt**2
        self.x[3:6] += acc_world * dt

        # Propagation Covariance
        G = np.zeros((6, 3))
        G[0:3, :] = 0.5 * dt**2 * np.eye(3)
        G[3:6, :] = dt * np.eye(3)
        
        Q_imu = G @ (self.accel_noise_std**2 * np.eye(3)) @ G.T
        self.P = self.F @ self.P @ self.F.T + Q_imu + self.Q

    def update(self, z) -> None:
        """Mise à jour avec mesure GPS (Covariance Intersection simplifiée)."""
        z = np.asarray(z, dtype=float).reshape(6)
        if np.isnan(z).any(): return

        try:
            P_inv = np.linalg.inv(self.P)
        except np.linalg.LinAlgError:
            P_inv = np.eye(6) * 0.1

        # Fusion simple avec poids fixe 0.5 pour robustesse
        omega = 0.5
        P_ci_inv = omega * P_inv + (1 - omega) * self.R_inv
        
        try:
            self.P = np.linalg.inv(P_ci_inv)
        except np.linalg.LinAlgError:
            self.P = np.eye(6) * 1.0

        weighted_state = omega * (P_inv @ self.x) + (1 - omega) * (self.R_inv @ z)
        self.x = self.P @ weighted_state

    def step(self, meas_pos, meas_vel, imu_accel=None, orientation_q=None):
        """Cycle complet : Prédiction (IMU) -> Correction (GPS)."""
        meas_pos = np.asarray(meas_pos, dtype=float).reshape(3)
        meas_vel = np.asarray(meas_vel, dtype=float).reshape(3)

        if not self._initialized:
            self._init_state(meas_pos, meas_vel)
            return self.x[0:3].copy(), self.x[3:6].copy()

        # 1. Prédiction
        self.predict(imu_accel, orientation_q)

        # 2. Correction
        z = np.concatenate([meas_pos, meas_vel])
        self.update(z)

        return self.x[0:3].copy(), self.x[3:6].copy()