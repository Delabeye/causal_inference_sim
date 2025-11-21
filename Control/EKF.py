import numpy as np


class GPSEKF:
    """
    EKF simple pour estimer position + vitesse à partir du GPS.

    État : x = [px, py, pz, vx, vy, vz]^T
    Modèle : position = position + dt * vitesse (vitesse constante)
    Mesure : z = [px, py, pz, vx, vy, vz]^T (GPS bruité)
    """

    def __init__(self, dt: float,
                 q_pos: float = 0.05,
                 q_vel: float = 0.10,
                 r_pos: float = 0.1,
                 r_vel: float = 0.05) -> None:
        self.dt = float(dt)

        # État et covariance
        self.x = np.zeros(6, dtype=float)   # [px, py, pz, vx, vy, vz]
        self.P = np.eye(6, dtype=float) * 1e-3

        # Bruits processus (Q) et mesure (R)
        q_pos = float(q_pos)
        q_vel = float(q_vel)
        r_pos = float(r_pos)
        r_vel = float(r_vel)

        self.Q = np.diag(
            [q_pos ** 2, q_pos ** 2, q_pos ** 2,
             q_vel ** 2, q_vel ** 2, q_vel ** 2]
        )
        self.R = np.diag(
            [r_pos ** 2, r_pos ** 2, r_pos ** 2,
             r_vel ** 2, r_vel ** 2, r_vel ** 2]
        )

        # Matrice de transition F (modèle : vitesse constante)
        dt = self.dt
        self.F = np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Mesure directe de tout l'état
        self.H = np.eye(6, dtype=float)

        self._initialized = False

    # ------------------------------------------------------------------
    def _init_state(self, meas_pos, meas_vel) -> None:
        """Initialise l'état à partir de la première mesure GPS."""
        meas_pos = np.asarray(meas_pos, dtype=float).reshape(3)
        meas_vel = np.asarray(meas_vel, dtype=float).reshape(3)

        self.x[0:3] = meas_pos
        self.x[3:6] = meas_vel
        self.P = np.eye(6, dtype=float) * 1e-1
        self._initialized = True

    # ------------------------------------------------------------------
    def predict(self) -> None:
        """Étape de prédiction de l’EKF."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, z) -> None:
        """Étape de correction de l’EKF."""
        z = np.asarray(z, dtype=float).reshape(6)

        # Innovation
        y = z - self.H @ self.x

        # Covariance de l’innovation
        S = self.H @ self.P @ self.H.T + self.R

        # Gain de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Mise à jour
        self.x = self.x + K @ y
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    # ------------------------------------------------------------------
    def step(self, meas_pos, meas_vel):
        """
        Appel complet : prédit puis corrige avec la mesure GPS.
        Retourne (pos_est, vel_est).
        """
        meas_pos = np.asarray(meas_pos, dtype=float).reshape(3)
        meas_vel = np.asarray(meas_vel, dtype=float).reshape(3)

        if not self._initialized:
            self._init_state(meas_pos, meas_vel)

        # Prédiction
        self.predict()

        # Mise à jour
        z = np.concatenate([meas_pos, meas_vel])
        self.update(z)

        pos_est = self.x[0:3].copy()
        vel_est = self.x[3:6].copy()
        return pos_est, vel_est
