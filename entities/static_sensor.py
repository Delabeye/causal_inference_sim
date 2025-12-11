import pybullet as p
import numpy as np
from entities.agent import Agent

class RadarStation(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.name = self.config.get("name", "Radar")
        self.dt = dt
        
        # 1. Configuration Physique
        # On utilise une forme visuelle simple (cylindre ou cube)
        start_pos = self.config.get("pos", [0, 0, 0])
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = self.config.get("urdf_path", "assets/cube.urdf")
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

    def set_targets(self, agents_list):
        """Reçoit la liste des agents à surveiller"""
        self.targets = agents_list

    def think_and_act(self):
        """
        Boucle principale du radar : Scan de l'environnement
        """
        self.detected_agents = []
        my_pos, _ = p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
        my_pos = np.array(my_pos)

        # DEBUG: Dessiner la zone de détection (sphère filaire rouge)
        # Note: PyBullet n'a pas de "drawSphere" simple persistant, on peut utiliser des lignes
        # Pour l'instant, on affiche juste dans la console.

        for agent in self.targets:
            # On ne se détecte pas soi-même
            if agent.bodyId == self.bodyId:
                continue
            
            # Récupérer position de la cible
            target_pos, _ = p.getBasePositionAndOrientation(agent.bodyId, physicsClientId=self.physics_client_id)
            target_pos = np.array(target_pos)
            
            # Calcul distance
            dist = np.linalg.norm(target_pos - my_pos)
            
            if dist <= self.detection_range:
                self.detected_agents.append(agent.name)
                # Action lors de la détection (Log, Alerte, etc.)
                print(f"[RADAR '{self.name}'] 🚨 DÉTECTION : '{agent.name}' à {dist:.2f}m !")
                
                