import numpy as np

class Sensor:
    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("The 'measure' method must be implemented by the subclass")


class GPSSensor(Sensor):
    def __init__(self, 
                 position_noise_std: float = 0.05, 
                 velocity_noise_std: float = 0.02 
                ):
        
        super().__init__()
        self.pos_noise_std = float(position_noise_std)
        self.vel_noise_std = float(velocity_noise_std)
        
        if self.pos_noise_std < 0:
            self.pos_noise_std = 0
        if self.vel_noise_std < 0:
            self.vel_noise_std = 0

    def measure(self, 
                ground_truth_position: np.ndarray, 
                ground_truth_velocity: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)

        measured_position = ground_truth_position + pos_noise
        measured_velocity = ground_truth_velocity + vel_noise
        
        return measured_position, measured_velocity