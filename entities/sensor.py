import math
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

class LidarSensor:
    def __init__(self, config: dict):
        """
        Lidar paramétrable avec un champ de vision (FOV) conique.
        """
        self.max_distance = float(config.get("max_distance", 5.0))
        self.angle_resolution = float(config.get("angle_resolution", 2.0))
        
        # FOV en degrés, convertis en demi-angles
        fov_h = float(config.get("fov_horizontal", 90.0))
        fov_v = float(config.get("fov_vertical", 30.0))
        
        self.half_fov_h = fov_h / 2.0
        self.half_fov_v = fov_v / 2.0
        
        # Pré-génération des vecteurs de rayons dans le repère LOCAL du drone
        # Axe X = Devant, Y = Gauche, Z = Haut
        self.local_rays = self._generate_local_rays()
        print(f"[LidarSensor] Initialisé : {len(self.local_rays)} rayons (FOV H:{fov_h}°, V:{fov_v}°)")

    def _generate_local_rays(self):
        rays = []
        # On balaie de gauche à droite (-fov_h/2 à +fov_h/2)
        for az in np.arange(-self.half_fov_h, self.half_fov_h, self.angle_resolution):
            # On balaie de bas en haut (-fov_v/2 à +fov_v/2)
            for el in np.arange(-self.half_fov_v, self.half_fov_v, self.angle_resolution):
                
                # Conversion degrés -> radians
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Coordonnées sphériques vers Cartésiennes (X est devant)
                # x = cos(el) * cos(az)
                # y = cos(el) * sin(az)
                # z = sin(el)
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                
                # Normalisation (juste par sécurité)
                v = np.array([x, y, z])
                v = v / np.linalg.norm(v)
                rays.append(v)
        
        return np.array(rays)

    def measure(self, position, roll, yaw, pitch):
        """
        Effectue un raycast par lot (batch) dans PyBullet.
        Args:
            position: [x, y, z] du drone
            roll, yaw, pitch: orientation actuelle en radians
        Returns:
            points: Liste de np.array [x, y, z] des impacts détectés
        """
        # 1. Calcul de la matrice de rotation du drone
        # PyBullet utilise l'ordre [roll, pitch, yaw] pour les quaternions Euler
        orn_q = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(orn_q)
        
        # Transformation de la matrice plate (9,) en (3,3)
        R = np.array(rot_matrix).reshape(3, 3)
        
        # 2. Rotation de tous les rayons locaux vers le monde
        # Formule: Ray_Monde = R * Ray_Local
        # Optimisation vectorielle : (N,3) dot (3,3) -> (N,3)
        # Note: on utilise transpose pour aligner les dimensions correctement
        world_rays_dir = self.local_rays @ R.T 
        
        # 3. Préparation des positions de départ et d'arrivée
        num_rays = len(world_rays_dir)
        ray_froms = np.tile(position, (num_rays, 1))
        ray_tos = ray_froms + world_rays_dir * self.max_distance
        
        # 4. Raycast PyBullet (Batch = très rapide)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 5. Filtrage des impacts
        detected_points = []
        for i, res in enumerate(results):
            # res structure: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
            hit_id = res[0]
            if hit_id >= 0: # Si on a touché un objet (id >= 0)
                hit_pos = np.array(res[3])
                detected_points.append(hit_pos)
                
        return detected_points