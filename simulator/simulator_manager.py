import pybullet as p
import pybullet_data
import time
import numpy as np

from environment.world import World
from entites.uav import UAV
# Importer les définitions d'obstacles 
from entites.obstacles import CubeObstacle, SphericalObstacle, CylindricalObstacle

class SimulationManager:
    """
    Classe centrale d'orchestration.
    Construit et exécute la simulation en se basant sur un objet config.
    """
    def __init__(self, config):
        self.config = config
        self.dt = self.config['simulation']['dt']
        
        # 1. Connexion à PyBullet (paramétrée)
        mode = p.GUI if self.config['simulation']['connect_mode'] == 'gui' else p.DIRECT
        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("N'a pas pu se connecter au client PyBullet.")
            
        print(f"Connecté au client PyBullet avec l'ID: {self.physics_client_id}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config['physics']['gravity']) # Paramétré
        
        self.agents = []
        self.setpoints = {}
        
        # 2. Initialiser l'environnement
        self.world = World(self.physics_client_id)
        self.world.load_basic_environment()
        
        # 3. Charger le Scénario (lit la config)
        self.load_scenario()

    def load_scenario(self):
        """
        Charge les agents et les obstacles en lisant l'objet config.
        """
        print("Chargement du scénario depuis la configuration...")
        
        # --- Charger les Obstacles depuis la config ---
        for obs_config in self.config['world']['obstacles']:
            if obs_config['type'] == 'cube':
                obstacle = CubeObstacle(center=obs_config['center'],
                                        length=obs_config['length'],
                                        width=obs_config['width'],
                                        height=obs_config['height'])
                self.world.add_cube_obstacle(obstacle)
            
            elif obs_config['type'] == 'sphere':
                obstacle = SphericalObstacle(center=obs_config['center'],
                                             radius=obs_config['radius'])
                self.world.add_sphere_obstacle(obstacle)
            
            elif obs_config['type'] == 'cylinder':
                obstacle = CylindricalObstacle(center=obs_config['center'],
                                               radius=obs_config['radius'],
                                               height=obs_config['height'])
                self.world.add_cylindrical_obstacle(obstacle)
        
        # --- Charger les Agents depuis la config ---
        for agent_config in self.config['agents']:
            if agent_config['type'] == 'uav':
                
                start_orn_q = p.getQuaternionFromEuler(agent_config['start_orn_euler'])
                
                # Crée l'agent en lui passant son propre bloc de config
                agent = UAV(
                    config=agent_config, # Passe tout le bloc de config de l'agent
                    physics_client_id=self.physics_client_id,
                    dt=self.dt
                ) [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                
                self.agents.append(agent)
                self.setpoints[agent.bodyId] = np.array(agent_config['setpoint'])
                
            # elif agent_config['type'] == 'ugv':
            #     # agent = UGV(config=agent_config,...)
            #     pass

        print(f"Scénario chargé : {len(self.agents)} agents, {len(self.world.obstacle_ids)} obstacles.")

    def run(self):
        """
        Exécute la boucle de simulation principale.
        """
        sim_time = 0.0
        max_time = self.config['simulation']['max_sim_time']
        
        while sim_time < max_time:
            
            # 1. Penser (Think) [Image 1]
            for agent in self.agents:
                setpoint = self.setpoints.get(agent.bodyId, np.zeros(3))
                agent.think_and_act(setpoint)

            # 2. Agir (Act) 
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 3. Log (Étape 5)
            # self.logger.log_tick(sim_time, self.agents)

            time.sleep(self.dt)
            sim_time += self.dt
                
    def stop(self):
        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(physicsClientId=self.physics_client_id)