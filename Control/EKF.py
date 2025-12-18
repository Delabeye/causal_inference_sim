import numpy as np

class GPSEKF:
    def __init__(self, dt):
        self.dt = dt
        # État : [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # Matrice de Transition (Partie cinématique simple x = x + v*dt)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Incertitude initiale
        self.P = np.eye(6) * 1.0
        
        # Bruit du processus (Q)
        # On réduit le bruit sur la vitesse car l'IMU va nous donner l'info précise
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.01 
        self.Q[3:6, 3:6] *= 0.05 

        # Matrices de Mesure GPS (Position + Vitesse)
        self.H = np.eye(6)
        self.R = np.eye(6)
        self.R[0:3, 0:3] *= 2.0   # Confiance GPS Position (Faible)
        self.R[3:6, 3:6] *= 0.5   # Confiance GPS Vitesse (Moyenne)

        self.GRAVITY = np.array([0, 0, 9.81])

    def _quat_to_rot_matrix(self, q):
        """Convertit un quaternion [x, y, z, w] en matrice de rotation 3x3"""
        x, y, z, w = q
        # Formule standard
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R

    def predict(self, imu_acc_body, orientation_quat):
        """
        Prédiction aidée par l'IMU.
        imu_acc_body : [ax, ay, az] mesuré par l'IMU (m/s^2)
        orientation_quat : [x, y, z, w] orientation actuelle
        """
        # 1. Rotation de l'accélération : Repère Drone -> Repère Monde
        # L'IMU donne l'accélération dans le repère du drone
        R = self._quat_to_rot_matrix(orientation_quat)
        acc_world = R @ imu_acc_body
        
        # 2. Retrait de la gravité
        # L'accéléromètre mesure aussi la gravité (9.81 vers le haut quand à plat).
        # On doit la soustraire pour obtenir l'accélération de mouvement pure.
        acc_linear = acc_world - self.GRAVITY
        
        # 3. Prédiction Physique (Lois de Newton)
        # Position future = Pos + Vel*dt + 0.5*Acc*dt^2
        # Vitesse future  = Vel + Acc*dt
        
        # D'abord on applique la partie Vitesse (F*x)
        self.x = self.F @ self.x
        
        # Ensuite on ajoute la partie Accélération (Matrice de contrôle B*u)
        # Terme position (0.5 * a * dt^2)
        self.x[0:3] += 0.5 * acc_linear * (self.dt**2)
        # Terme vitesse (a * dt)
        self.x[3:6] += acc_linear * self.dt
        
        # 4. Mise à jour de la covariance
        # On ajoute du bruit car l'IMU n'est pas parfaite
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, pos_meas, vel_meas):
        """Correction standard avec GPS"""
        z = np.hstack([pos_meas, vel_meas])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except:
            return self.x[:3], self.x[3:6]
        
        self.x = self.x + (K @ y)
        I = np.eye(6)
        self.P = (I - (K @ self.H)) @ self.P
        
        return self.x[:3], self.x[3:6]