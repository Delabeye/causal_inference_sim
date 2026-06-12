import math
import numpy as np
import pybullet as p


class Sensor:
    """
    Sensor class for measuring physical quantities.
    A base class that defines the interface for sensor implementations.
    Subclasses must implement the measure method to provide specific
    measurement functionality.
    Attributes:
        None
    Methods:
        measure(*args, **kwargs): Measure physical quantities.
            Must be implemented by subclasses.
    Raises:
        NotImplementedError: If measure method is called on base Sensor class.
    Example:
        >>> class TemperatureSensor(Sensor):
        ...     def measure(self, *args, **kwargs):
        ...         return 25.5
        >>> sensor = TemperatureSensor()
        >>> sensor.measure()
        25.5
    """
    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("The 'measure' method must be implemented.")


class GNSSensor(Sensor):
    """
    Simple GPS Sensor:
    - position_noise_std : standard deviation of position noise (m)
    - velocity_noise_std : standard deviation of velocity noise (m/s)
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
        Returns (measured_position, measured_velocity) with Gaussian noise.
        """
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)

        meas_pos = ground_truth_position + pos_noise
        meas_vel = ground_truth_velocity + vel_noise

        return meas_pos, meas_vel


# Dans entities/sensor.py

class IMUSensor:
    def __init__(self, config_dict=None):
        self.last_vel = np.zeros(3)
        self.dt = 1/100 # Supposé
        self.g_vector = np.array([0, 0, 9.81])

    def measure(self, vel, orn_q):
        """
        Simule un accéléromètre + gyroscope
        Retourne : acc_body, gyro_body
        """
        # 1. Calcul Accélération Monde (a = dv/dt)
        # (C'est une approximation discrete)
        acc_world = (np.array(vel) - self.last_vel) / self.dt
        self.last_vel = np.array(vel)
        
        # 2. Ajout de la "pesanteur ressentie" (Proper Acceleration)
        # Un accéléromètre mesure (a - g). Comme g pointe vers le bas (-9.81), 
        # a_mesure = a_monde - (-9.81) = a_monde + 9.81
        acc_proper_world = acc_world + self.g_vector
        
        # 3. Rotation vers Body Frame (Monde -> Drone)
        # On utilise la matrice inverse de rotation
        R_world_to_body = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3,3).T
        acc_body = R_world_to_body @ acc_proper_world
        
        # Ajout de bruit (facultatif)
        acc_body += np.random.normal(0, 0.1, 3) # Bruit blanc
        
        # Gyro (Vitesse angulaire, ici on met 0 ou la vraie pour simplifier)
        gyro_body = np.zeros(3) 
        
        return acc_body, gyro_body

class LidarSensor:
    """
    LidarSensor
    A configurable LiDAR sensor simulator that emulates a conical Field of View (FOV) 
    sensor mounted on a drone. It generates ray vectors in the local drone frame and 
    performs batch raycasting in PyBullet to detect obstacles and environmental features.
    The sensor operates in the drone's local coordinate system where:
    - X-axis: Forward direction
    - Y-axis: Left direction
    - Z-axis: Up direction
    Attributes:
        max_distance (float): Maximum detection range in meters (default: 5.0)
        angle_resolution (float): Angular resolution between rays in degrees (default: 2.0)
        half_fov_h (float): Half of the horizontal field of view in degrees
        half_fov_v (float): Half of the vertical field of view in degrees
        local_rays (np.ndarray): Pre-generated array of ray direction vectors in the drone's 
                                 local frame, shape (N, 3) where N is the number of rays
    Methods:
        _generate_local_rays(): Generates all ray vectors in the local drone frame by sweeping
                               horizontally and vertically within the FOV bounds, converting
                               spherical coordinates to Cartesian unit vectors.
        measure(position, roll, yaw, pitch): Performs batch raycasting from the drone's 
                                             current position and orientation to detect 
                                             obstacles in the environment using PyBullet.
    """
    def __init__(self, config: dict):
        """
        Configurable Lidar with a conical Field of View (FOV).
        """
        self.max_distance = float(config.get("max_distance", 5.0))
        self.angle_resolution = float(config.get("angle_resolution", 2.0))
        
        # FOV in degrees, converted to half-angles
        fov_h = float(config.get("fov_horizontal", 90.0))
        fov_v = float(config.get("fov_vertical", 30.0))
        
        self.half_fov_h = fov_h / 2.0
        self.half_fov_v = fov_v / 2.0
        
        # Pre-generation of ray vectors in the LOCAL frame of the drone
        # X-axis = Forward, Y = Left, Z = Up
        self.local_rays = self._generate_local_rays()
        print(f"[LidarSensor] Initialized : {len(self.local_rays)} rays (FOV H:{fov_h}°, V:{fov_v}°)")

    def _generate_local_rays(self):
        rays = []
        # Sweep from left to right (-fov_h/2 to +fov_h/2)
        for az in np.arange(-self.half_fov_h, self.half_fov_h, self.angle_resolution):
            # Sweep from bottom to top (-fov_v/2 to +fov_v/2)
            for el in np.arange(-self.half_fov_v, self.half_fov_v, self.angle_resolution):
                
                # Convert degrees -> radians
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Spherical to Cartesian coordinates (X is forward)
                # x = cos(el) * cos(az)
                # y = cos(el) * sin(az)
                # z = sin(el)
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                
                # Normalization (safety check)
                v = np.array([x, y, z])
                v = v / np.linalg.norm(v)
                rays.append(v)
        
        return np.array(rays)

    def measure(self, position, roll, yaw, pitch):
        """
        Performs a batch raycast in PyBullet.
        Args:
            position: [x, y, z] of the drone
            roll, yaw, pitch: current orientation in radians
        Returns:
            points: List of np.array [x, y, z] of detected impacts
        """
        # 1. Calculate drone rotation matrix
        # PyBullet uses [roll, pitch, yaw] order for Euler quaternions
        orn_q = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(orn_q)
        
        # Reshape flat matrix (9,) to (3,3)
        R = np.array(rot_matrix).reshape(3, 3)
        
        # 2. Rotate all local rays to world frame
        # Formula: Ray_World = R * Ray_Local
        # Vector optimization: (N,3) dot (3,3) -> (N,3)
        # Note: using transpose to align dimensions correctly
        world_rays_dir = self.local_rays @ R.T 
        
        # 3. Prepare start and end positions
        num_rays = len(world_rays_dir)
        ray_froms = np.tile(position, (num_rays, 1))
        ray_tos = ray_froms + world_rays_dir * self.max_distance
        
        # 4. PyBullet Raycast (Batch = very fast)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 5. Filter impacts
        detected_points = []
        for _, res in enumerate(results):
            # res structure: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
            hit_id = res[0]
            if hit_id >= 0: # If an object was hit (id >= 0)
                hit_pos = np.array(res[3])
                if hit_pos[2] > 0.01:  # Filter to avoid points too close
                    hit_id = [round(coord, 3) for coord in hit_pos]
                    detected_points.append(hit_pos)
                
        return detected_points
    
class HeightSensor:
    def __init__(self, physics_client_id, noise_std=0.01, max_range=4.0):
        """
        Simule un capteur de distance orienté vers le bas (Down-facing Lidar/Sonar).
        """
        self.client_id = physics_client_id
        self.noise_std = noise_std
        self.max_range = max_range

    def measure(self, pos, orn_q):
        """
        Retourne la distance mesurée vers le sol (ou None si hors de portée).
        """
        # 1. Calcul du vecteur direction (Le capteur pointe vers le "Bas" du drone)
        # En repère monde, le bas du drone change si le drone penche (Roll/Pitch)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3, 3)
        # Le vecteur "Bas" dans le repère du drone est [0, 0, -1]
        # On le tourne dans le repère monde
        down_vec_world = rot_mat @ np.array([0, 0, -1])

        # 2. Raycast (Tir du rayon)
        start = np.array(pos)
        end = start + (down_vec_world * self.max_range)
        
        results = p.rayTest(start, end, physicsClientId=self.client_id)
        # results[0] contient [objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal]
        
        hit_fraction = results[0][2]
        
        # 3. Traitement
        if hit_fraction == 1.0: # Rien touché
            return None # Trop haut pour le capteur
        
        # Distance réelle = hit_fraction * max_range
        dist = hit_fraction * self.max_range
        
        # Ajout du bruit
        dist += np.random.normal(0, self.noise_std)
        
        # Protection valeurs négatives
        return max(0.0, dist)