import numpy as np

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