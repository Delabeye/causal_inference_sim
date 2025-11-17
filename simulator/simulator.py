import utilities.config as cfg 
import entites.obstacles as obs
import entites.sensor as sns

class Simulator:

    def __init__(self):
        pass

    def run(self):
        pass

    def reset(self):
        pass

    def add_sensor(self, entity):

        sensors = [sns.GNSS_Sensor(identifier=i,
                                   position_noise_std=cfg.gnss_position_noise_std[i],
                                   velocity_noise_std=cfg.gnss_velocity_noise_std[i]
                                   ) for i in range(cfg.Num_GNSS_sensors)]
        return sensors

