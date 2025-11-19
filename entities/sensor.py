import numpy as np
import pybullet as p

class Sensor:
    """Classe de base pour les capteurs (identique à votre PDF )."""
    def __init__(self):
        pass
        
    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée par la sous-classe")

class GPSSensor(Sensor):
    """
    Capteur GPS qui lit ses paramètres depuis un objet config.
    """
    def __init__(self, config):
        """
        Initialise le capteur GPS.
        'config' est un dictionnaire, par ex:
        {'position_noise_std': 0.05, 'velocity_noise_std': 0.02}
        """
        super().__init__()
        
        # Lit les paramètres depuis l'objet config
        self.pos_noise_std = float(config.get('position_noise_std', 0.0))
        self.vel_noise_std = float(config.get('velocity_noise_std', 0.0))

        if self.pos_noise_std < 0:
            self.pos_noise_std = 0
        if self.vel_noise_std < 0:
            self.vel_noise_std = 0
            
    def measure(self,
                ground_truth_position: np.ndarray,
                ground_truth_velocity: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        
        # (Code identique à votre PDF )
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)
        
        measured_position = ground_truth_position + pos_noise
        measured_velocity = ground_truth_velocity + vel_noise
        
        return measured_position, measured_velocity
    

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