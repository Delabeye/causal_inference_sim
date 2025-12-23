import json
import pybullet as p
import numpy as np
from entities.agent import Agent
import zmq

class RadarStation(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.name = self.config.get("name", "Radar")
        self.dt = dt
        self.physics_client_id = physics_client_id
        self.position_noise_std = float(self.config.get("position_noise_std", 0.0))
        self.velocity_noise_std = float(self.config.get("velocity_noise_std", 0.0))
        self.bodyId = self.config.get("bodyId", 1000)   
        # 1. Configuration Physique

        start_pos = self.config.get("pos", [0, 0, 0])
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = self.config.get("urdf_path", "assets/radar.urdf")
        super().__init__(urdf_path, start_pos, start_orn, physics_client_id, dt)
        
        # Rendre l'objet statique (Masse = 0)
        p.changeDynamics(self.bodyId, -1, mass=0, physicsClientId=self.physics_client_id)
        
        # Couleur distinctive (Rouge pour un radar "ennemi" ou de surveillance)
        p.changeVisualShape(self.bodyId, -1, rgbaColor=[0.8, 0, 0, 1], physicsClientId=self.physics_client_id)

        # 2. Configuration du Capteur
        self.detection_range = float(self.config.get("range", 5.0)) # Rayon en mètres
        self.detected_agents = []
        
        # Référence vers la liste des cibles (sera remplie par le Manager)
        self.targets = [] 

    def setup_network(self, ip, port_pub,_):
        """Configure la radio du drone (ZeroMQ)"""
        self.zmq_ctx = zmq.Context()
        self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.IDENTITY, self.name)  
        self.pub_socket.connect(f"tcp://{ip}:{port_pub}")
        
    def publish_detection(self):
        """Publie les agents détectés via la radio (ZeroMQ)"""
        if not self.detected_agents:
            return  # Rien à publier
        
        message = {
            "radar_name": self.name,
            "detected_agents": [self.detected_agents[0], self.detected_agents[1], self.detected_agents[2]],
            "timestamp": p.getRealTimeSimulation(physicsClientId=self.physics_client_id)
        }
        self.pub_socket.send_string("State" + json.dumps(message))

    def think_and_act(self):
        """
        Boucle principale du radar : Scan de l'environnement
        """
        self.detected_agents = []

        for agent in self.targets:
            # On ne se détecte pas soi-même
            if agent.bodyId == self.bodyId:
                continue
            
            # Récupérer position de la cible
            target_pos, _ = p.getBasePositionAndOrientation(agent.bodyId, physicsClientId=self.physics_client_id)
            target_pos = np.array(target_pos)
            target_vel, _ = p.getBaseVelocity(agent.bodyId, physicsClientId=self.physics_client_id)
            # Calcul distance
            dist = np.linalg.norm(target_pos - self.pos)
            
            if dist <= self.detection_range:
                pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
                vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)
                self.detected_agents.append((agent.name, target_pos+pos_noise, target_vel+vel_noise))
                # Action lors de la détection (Log, Alerte, etc.)
                self.publish_detection()
                
                