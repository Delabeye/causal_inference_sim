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
    """Capteur GPS simple."""
    def __init__(self, config: dict):
        super().__init__()
        self.pos_noise_std = float(config.get("position_noise_std", 0.0))
        self.vel_noise_std = float(config.get("velocity_noise_std", 0.0))
        if self.pos_noise_std < 0.0: self.pos_noise_std = 0.0
        if self.vel_noise_std < 0.0: self.vel_noise_std = 0.0

    def measure(self, ground_truth_position: np.ndarray, ground_truth_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)
        return ground_truth_position + pos_noise, ground_truth_velocity + vel_noise


class IMUSensor(Sensor):
    """IMU simplifiée."""
    def __init__(self, config: dict):
        super().__init__()
        self.accel_noise_std = float(config.get("accel_noise_std", 0.0))
        self.gyro_noise_std = float(config.get("gyro_noise_std", 0.0))
        self.gravity = float(config.get("gravity", 9.81))
        self._prev_vel = np.zeros(3, dtype=float)
        self._initialized = False

    def measure(self, gt_pos, gt_orn, gt_vel, gt_ang_vel, dt: float):
        if dt <= 0.0: dt = 1e-6
        v = np.array(gt_vel, dtype=float)
        if not self._initialized:
            a_world = np.zeros(3, dtype=float)
            self._initialized = True
        else:
            a_world = (v - self._prev_vel) / dt
        self._prev_vel = v

        g_world = np.array([0.0, 0.0, -self.gravity], dtype=float)
        f_world = a_world - g_world
        q = gt_orn
        rot_mat = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        f_body = rot_mat.T @ f_world
        omega_body = np.array(gt_ang_vel, dtype=float)

        if self.accel_noise_std > 0.0: f_body += np.random.normal(0.0, self.accel_noise_std, 3)
        if self.gyro_noise_std > 0.0: omega_body += np.random.normal(0.0, self.gyro_noise_std, 3)

        return f_body, omega_body


class LidarSensor(Sensor):
    """
    Capteur LIDAR capable de scanner l'environnement ET de mesurer l'altitude.
    """
    def __init__(self, max_distance: float, angle_resolution: float):
        super().__init__()
        self.max_distance = max_distance
        self.angle_resolution = angle_resolution

        if self.max_distance <= 0: self.max_distance = 100.0
        if self.angle_resolution <= 0: self.angle_resolution = 1.0
        
    def measure(self, sensor_position: np.ndarray, roll: float, yaw: float, pitch: float) -> list[np.ndarray]:
        """Simule un lidar 2D à 360° autour du capteur (Plan XY)."""
        obstacles_positions = []
        num_measurements = int(360 / self.angle_resolution)
        
        # Copie pour ne pas modifier l'original
        pos = sensor_position.copy()
        pos[2] += 0.01 

        for i in range(num_measurements):
            angle_deg = i * self.angle_resolution
            angle_rad = np.radians(angle_deg)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            
            # Matrices de rotation
            Rz = np.array([[math.cos(yaw), -math.sin(yaw),0], [math.sin(yaw), math.cos(yaw),0], [0,0,1]])
            Ry = np.array([[math.cos(pitch),0, math.sin(pitch)], [0,1,0], [-math.sin(pitch),0, math.cos(pitch)]])
            Rx = np.array([[1,0,0], [0,math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])
            R = Rz @ Ry @ Rx
            
            ray_end = pos + R @ direction * self.max_distance
            result = p.rayTest(pos.tolist(), ray_end.tolist())
            
            if result[0][1] != -1:
                obstacles_positions.append(np.array(result[0][3], dtype=float))
    
        return obstacles_positions

    def measure_altitude(self, sensor_position: np.ndarray, orientation_q: np.ndarray) -> float:
        """
        NOUVEAU : Mesure l'altitude Z via un rayon vers le bas (Body Frame).
        Retourne l'altitude Z projetée ou NaN si hors de portée.
        """
        # 1. Vecteur "Bas" dans le repère du drone (Body)
        vec_down_body = np.array([0, 0, -1])
        
        # 2. Rotation vers le repère Monde
        rot_mat = np.array(p.getMatrixFromQuaternion(orientation_q)).reshape(3, 3)
        vec_down_world = rot_mat @ vec_down_body
        
        # 3. Lancer du rayon
        start = sensor_position
        end = sensor_position + vec_down_world * self.max_distance
        
        result = p.rayTest(start.tolist(), end.tolist())
        hit_fraction = result[0][2]
        
        if hit_fraction < 1.0:
            # Distance mesurée (en oblique si le drone est penché)
            slant_range = hit_fraction * self.max_distance
            
            # 4. Correction d'angle pour avoir la hauteur verticale Z
            # On projette la distance mesurée sur l'axe Z vertical du monde
            # cos(theta) = produit scalaire entre le rayon et la verticale (0,0,-1)
            # Comme le rayon est vec_down_world (unitaire), cos_theta = -vec_down_world[2]
            cos_tilt = -vec_down_world[2]
            
            if cos_tilt > 0:
                z_altitude = slant_range * cos_tilt
                return z_altitude
                
        return float('nan')
