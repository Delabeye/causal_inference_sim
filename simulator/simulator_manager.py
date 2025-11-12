import pybullet as p
import pybullet_data
import utilities.config as cfg 
import entites.drone as drn
import entites.obstacles as obs
import entites.sensor as sns




class World:
    def __init__(self, physics_client_id):
        self.p = p
        self.physics_client_id = physics_client_id
        self.obstacle_ids =

    def load_basic_environment(self):
        """Charge le plan de base et définit la physique par défaut."""
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                         physicsClientId=self.physics_client_id)
        
        # Définir la gravité et le pas de temps
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        self.p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id) # Pas manuel
        
        # Charger la surface du sol
        self.p.loadURDF("plane.urdf", , useFixedBase=1, 
                          physicsClientId=self.physics_client_id)
        
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