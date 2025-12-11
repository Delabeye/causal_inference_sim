import numpy as np
import pybullet as p

class GPSEKF:
    """EKF Standard avec fusion IMU + GPS + LIDAR (Altitude)."""

    def __init__(self, dt: float,
                 q_pos: float = 0.05,
                 q_vel: float = 0.10,
                 r_pos: float = 0.5,
                 r_vel: float = 0.1,
                 accel_noise_std: float = 0.2) -> None:
        self.dt = float(dt)
        self.x = np.zeros(6, dtype=float)
        self.P = np.eye(6, dtype=float) * 1.0

        # Matrices de bruit
        self.Q = np.eye(6, dtype=float)
        self.Q[0:3, 0:3] *= q_pos**2
        self.Q[3:6, 3:6] *= q_vel**2
        
        self.R = np.diag([r_pos**2]*3 + [r_vel**2]*3)
        
        # Modèle
        self.F = np.eye(6, dtype=float)
        self.F[0,3] = dt; self.F[1,4] = dt; self.F[2,5] = dt
        
        self._initialized = False
        self.g_vector = np.array([0, 0, 9.81])

    def _init_state(self, meas_pos, meas_vel) -> None:
        if np.isnan(meas_pos).any() or np.isnan(meas_vel).any(): return
        self.x[0:3] = meas_pos
        self.x[3:6] = meas_vel
        self.P = self.R.copy()
        self._initialized = True

    def predict(self, imu_accel=None, orientation_q=None) -> None:
        dt = self.dt
        if imu_accel is None or orientation_q is None or np.isnan(imu_accel).any():
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q 
            return

        rot_mat = np.array(p.getMatrixFromQuaternion(orientation_q)).reshape(3, 3)
        acc_world = rot_mat @ np.array(imu_accel) - self.g_vector

        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc_world * dt**2
        self.x[3:6] += acc_world * dt
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z) -> None:
        z = np.asarray(z, dtype=float).reshape(6)
        if np.isnan(z).any(): return
        
        H = np.eye(6)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        
        try: K = self.P @ H.T @ np.linalg.inv(S)
        except: return

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_height(self, z_meas, R_alt=0.01):
        """Fusionne la mesure d'altitude du LIDAR (très précis)."""
        if np.isnan(z_meas): return
        
        # Mesure scalaire z (index 2 dans le vecteur d'état)
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        
        y = z_meas - self.x[2]
        S = H @ self.P @ H.T + R_alt
        
        try: K = self.P @ H.T @ np.linalg.inv(S)
        except: return
        
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def step(self, meas_pos, meas_vel, imu_accel=None, orientation_q=None):
        meas_pos = np.asarray(meas_pos, dtype=float).reshape(3)
        meas_vel = np.asarray(meas_vel, dtype=float).reshape(3)

        if not self._initialized:
            self._init_state(meas_pos, meas_vel)
            return self.x[0:3].copy(), self.x[3:6].copy()

        self.predict(imu_accel, orientation_q)
        z = np.concatenate([meas_pos, meas_vel])
        self.update(z)
        return self.x[0:3].copy(), self.x[3:6].copy()
