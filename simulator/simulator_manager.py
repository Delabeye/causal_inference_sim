import utilities.config as cfg 
import entites.drone as drn
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


    def add_drone(self, entity):
        
        drones = [drn.Drone(identifier=i,
                        coord=cfg.Drone_coords[i],
                        angle=cfg.Drone_angles[i],
                        speed=cfg.Drone_speeds[i],
                        mass=cfg.Drone_mass[i],
                        drag=cfg.Drone_drag_coefficient[i],
                        max_speed=cfg.Max_drone_speed[i],
                        max_acceleration=cfg.Max_drone_acceleration[i],
                        sensors=cfg.Sensors[i]
                        ) for i in range(cfg.Num_drones)]
