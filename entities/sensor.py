import numpy as np
import pybullet as p


class Sensor:
    """Classe de base pour les capteurs."""
    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée.")


class GPSSensor(Sensor):
    """
    Capteur GPS simple :
      - position_noise_std : écart-type du bruit sur la position (m)
      - velocity_noise_std : écart-type du bruit sur la vitesse (m/s)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.pos_noise_std = float(config.get("position_noise_std", 0.0))
        self.vel_noise_std = float(config.get("velocity_noise_std", 0.0))

        if self.pos_noise_std < 0.0:
            self.pos_noise_std = 0.0
        if self.vel_noise_std < 0.0:
            self.vel_noise_std = 0.0

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Renvoie (position_mesurée, vitesse_mesurée) avec bruit gaussien.
        """
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)

        meas_pos = ground_truth_position + pos_noise
        meas_vel = ground_truth_velocity + vel_noise

        return meas_pos, meas_vel


class IMUSensor(Sensor):
    """
    IMU simplifiée :
      - accéléromètre : mesure la force spécifique en repère body (a - g)
      - gyroscope : mesure la vitesse angulaire en repère body

    config :
      - accel_noise_std : écart-type bruit accel (m/s^2)
      - gyro_noise_std  : écart-type bruit gyro (rad/s)
      - gravity         : norme de la gravité (par défaut 9.81)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.accel_noise_std = float(config.get("accel_noise_std", 0.0))
        self.gyro_noise_std = float(config.get("gyro_noise_std", 0.0))
        self.gravity = float(config.get("gravity", 9.81))

        self._prev_vel = np.zeros(3, dtype=float)
        self._initialized = False

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_orientation: np.ndarray,
        ground_truth_velocity: np.ndarray,
        ground_truth_ang_vel: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Renvoie (specific_force_body, gyro_body).

        - specific_force_body ≈ R^T (a_world - g_world)
        - gyro_body ≈ vitesse angulaire fournie par PyBullet (approx)
        """
        if dt <= 0.0:
            dt = 1e-6

        v = np.array(ground_truth_velocity, dtype=float)

        if not self._initialized:
            a_world = np.zeros(3, dtype=float)
            self._initialized = True
        else:
            a_world = (v - self._prev_vel) / dt

        self._prev_vel = v

        # Gravité en repère monde
        g_world = np.array([0.0, 0.0, -self.gravity], dtype=float)

        # Force spécifique en repère monde
        f_world = a_world - g_world

        # Rotation monde -> body via quaternion
        q = ground_truth_orientation  # [x, y, z, w]
        rot_mat = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        f_body = rot_mat.T @ f_world

        # Vitesse angulaire (on la prend telle quelle, approx body)
        omega_body = np.array(ground_truth_ang_vel, dtype=float)

        # Bruits
        if self.accel_noise_std > 0.0:
            f_body = f_body + np.random.normal(0.0, self.accel_noise_std, 3)
        if self.gyro_noise_std > 0.0:
            omega_body = omega_body + np.random.normal(0.0, self.gyro_noise_std, 3)

        return f_body, omega_body

class LidarSensor(Sensor):
    """
    Capteur LIDAR qui lit ses paramètres depuis un objet config.    
    """
    def __init__(self, config):

        """
        Initialise le capteur LIDAR.
        'config' est un dictionnaire, par ex:
        {'max_distance': 100.0, 'angle_resolution': 1.0}
        """
        super().__init__()
        
        # Lit les paramètres depuis l'objet config
        self.max_distance = float(config.get('max_distance', 100.0))
        self.angle_resolution = float(config.get('angle_resolution', 1.0))

        if self.max_distance <= 0:
            self.max_distance = 100.0
        if self.angle_resolution <= 0:
            self.angle_resolution = 1.0
        
    def measure(self, sensor_position: np.ndarray):
        """
        Simule un lidar 2D à 360° autour du capteur.
        sensor_position : np.array([x, y, z])
        """
        obstacles_positions = []
        num_measurements = int(360 / self.angle_resolution)

        for i in range(num_measurements):
            angle_deg = i * self.angle_resolution
            angle_rad = np.radians(angle_deg)

            # direction dans le plan XY
            direction = np.array([
                np.cos(angle_rad),
                np.sin(angle_rad),
                0.0
                ])

            # point final du rayon
            ray_end = sensor_position + direction * self.max_distance

            result = p.rayTest(
                sensor_position.tolist(),
                ray_end.tolist()
            )

            # result structure :
            # (objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal)

            hit_id = result[0]

            if hit_id != -1:
                obstacles_positions.append(np.array(result[3]))
    
        return obstacles_positions  