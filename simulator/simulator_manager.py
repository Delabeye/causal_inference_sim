import pybullet as p
import pybullet_data
import time
import numpy as np

from environment.world import World
from entities.uav import UAV
from entities.obstacles import CubeObstacle, SphericalObstacle, CylindricalObstacle


class SimulationManager:
    """
    Classe centrale d'orchestration.
    Construit et exécute la simulation en se basant sur un objet config.
    """

    def __init__(self, config):
        self.config = config
        self.dt = self.config["simulation"]["dt"]

        # 1. Connexion à PyBullet (paramétrée)
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        mode = p.GUI if mode_str == "gui" else p.DIRECT

        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("N'a pas pu se connecter au client PyBullet.")

        print(f"Connecté au client PyBullet avec l'ID: {self.physics_client_id}")

        # Chemin des ressources + gravité
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config["physics"]["gravity"])

        # 2. Initialiser l'environnement (sol, gravité)
        self.world = World(self.physics_client_id)
        self.world.load_basic_environment()

        # 3. Optionnel : régler la caméra pour bien voir les obstacles
        # Tes obstacles YAML sont vers :
        #   cube     : [2, 0, 0.5]
        #   sphère   : [-2, 1, 1]
        #   cylindre : [0, 3, 1.5]
        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, 1.5, 1.0],
            )

        self.agents = []
        self.setpoints = {}

        # 4. Charger le scénario complet (obstacles + agents)
        self.load_scenario()

    # ------------------------------------------------------------------
    # Chargement du scénario
    # ------------------------------------------------------------------
    def load_scenario(self):
        """
        Charge les agents et les obstacles en lisant l'objet config.
        """
        print("Chargement du scénario depuis la configuration...")

        # --- Obstacles ---
        world_cfg = self.config.get("world", {})
        for obs_config in world_cfg.get("obstacles", []):
            otype = obs_config.get("type")

            if otype == "cube":
                obstacle = CubeObstacle(
                    center=obs_config["center"],
                    length=obs_config["length"],
                    width=obs_config["width"],
                    height=obs_config["height"],
                )
                self.world.add_cube_obstacle(obstacle)

            elif otype == "sphere":
                obstacle = SphericalObstacle(
                    center=obs_config["center"],
                    radius=obs_config["radius"],
                )
                self.world.add_sphere_obstacle(obstacle)

            elif otype == "cylinder":
                obstacle = CylindricalObstacle(
                    center=obs_config["center"],
                    radius=obs_config["radius"],
                    height=obs_config["height"],
                )
                self.world.add_cylindrical_obstacle(obstacle)

        # --- Agents (UAV) ---
        for agent_config in self.config.get("agents", []):
            if agent_config.get("type") == "uav":
                # Le constructeur UAV lit tout dans son bloc config
                agent = UAV(
                    config=agent_config,
                    physics_client_id=self.physics_client_id,
                    dt=self.dt,
                )
                self.agents.append(agent)
                self.setpoints[agent.bodyId] = np.array(agent_config["setpoint"])

        print(
            f"Scénario chargé : {len(self.agents)} agents, "
            f"{len(self.world.obstacle_ids)} obstacles."
        )

    # ------------------------------------------------------------------
    # Boucle principale de simulation
    # ------------------------------------------------------------------
    def run(self):
        """
        Exécute la boucle de simulation principale.
        """
        sim_time = 0.0
        max_time = self.config["simulation"]["max_sim_time"]

        while sim_time < max_time:
            # 1. Penser (Think)
            for agent in self.agents:
                setpoint = self.setpoints.get(agent.bodyId, np.zeros(3))
                agent.think_and_act(setpoint)

            # 2. Avancer la physique
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 3. (Optionnel) logging ici

            time.sleep(self.dt)
            sim_time += self.dt

    # ------------------------------------------------------------------
    # Arrêt propre
    # ------------------------------------------------------------------
    def stop(self):
        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(physicsClientId=self.physics_client_id)
