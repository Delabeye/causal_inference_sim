# simulator/simulator_manager.py
import pybullet as p
import pybullet_data
import time
import numpy as np

from environment.world import World
from entities.uav import UAV
from entities.obstacles import CubeObstacle, SphericalObstacle, CylindricalObstacle


class SimulationManager:
    def __init__(self, config):
        self.config = config
        self.dt = self.config['simulation']['dt']

        # --- Connexion PyBullet robuste (accepte 'GUI', 'gui ', etc.) ---
        mode_str = str(self.config['simulation']['connect_mode']).strip().lower()
        mode = p.GUI if mode_str == 'gui' else p.DIRECT

        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("N'a pas pu se connecter au client PyBullet.")

        print(f"Connecté au client PyBullet avec l'ID: {self.physics_client_id}")

        # Chemin des données (plane.urdf, etc.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config['physics']['gravity'])

        # Monde + sol
        self.world = World(self.physics_client_id)
        self.world.load_basic_environment()

        # --- IMPORTANT : mettre la caméra sur la zone des obstacles ---
        # Les obstacles de ton YAML sont autour de (0,0,~1), (2,0,0.5), (-2,1,1), (0,3,1.5)
        p.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.0, 1.5, 1.0],
        )

        self.agents = []
        self.setpoints = {}

        # Chargement du scénario (obstacles + agents)
        self.load_scenario()
