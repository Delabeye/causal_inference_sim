# simulator/simulator_manager.py
import time
import numpy as np
import pybullet as p
import pybullet_data

from environment.world import World
from entities.uav import UAV
from entities.obstacles import CubeObstacle, SphericalObstacle, CylindricalObstacle


class SimulationManager:
    """
    Orchestration de la simulation :
      - connexion à PyBullet
      - création du monde (sol + obstacles)
      - création des UAV depuis config.yaml
      - boucle de simulation
    """

    def __init__(self, config: dict):
        self.config = config
        self.dt = float(self.config["simulation"]["dt"])

        # 1. Connexion PyBullet
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        mode = p.GUI if mode_str == "gui" else p.DIRECT

        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("Impossible de se connecter à PyBullet.")

        print(f"Connecté à PyBullet, client_id={self.physics_client_id}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config["physics"]["gravity"], physicsClientId=self.physics_client_id)

        # 2. Monde (sol + obstacles)
        self.world = World(self.physics_client_id)
        self.world.load_basic_environment()

        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, 1.0, 1.0],
                physicsClientId=self.physics_client_id,
            )

        self.agents: list[UAV] = []

        # 3. Charger scénario
        self.load_scenario()

    # ------------------------------------------------------------------
    def load_scenario(self):
        print("Chargement du scénario...")

        # Obstacles
        world_cfg = self.config.get("world", {})
        for obs_cfg in world_cfg.get("obstacles", []):
            otype = obs_cfg.get("type")

            if otype == "cube":
                obstacle = CubeObstacle(
                    center=obs_cfg["center"],
                    length=obs_cfg["length"],
                    width=obs_cfg["width"],
                    height=obs_cfg["height"],
                )
                self.world.add_cube_obstacle(obstacle)

            elif otype == "sphere":
                obstacle = SphericalObstacle(
                    center=obs_cfg["center"],
                    radius=obs_cfg["radius"],
                )
                self.world.add_sphere_obstacle(obstacle)

            elif otype == "cylinder":
                obstacle = CylindricalObstacle(
                    center=obs_cfg["center"],
                    radius=obs_cfg["radius"],
                    height=obs_cfg["height"],
                )
                self.world.add_cylindrical_obstacle(obstacle)

        # Drones
        for agent_cfg in self.config.get("agents", []):
            if agent_cfg.get("type") == "uav":
                uav = UAV(
                    config=agent_cfg,
                    physics_client_id=self.physics_client_id,
                    dt=self.dt,
                )
                self.agents.append(uav)

        for objective in self.config.get("objectives", []):
            agent_id = objective.get("agent_body_id")
            if objective.get("type") == "reach_setpoint":
                if agent_id is None:
                    raise ValueError("Objective of type 'reach_setpoint' is missing 'agent_body_id'.")
                setpoint = np.array(objective.get("setpoint", [0.0, 0.0, 0.0]), dtype=float)
                self.setpoints[agent_id] = setpoint

        print(
            f"Scénario chargé : {len(self.agents)} drones, "
            f"{len(self.world.obstacle_ids)} obstacles."
        )

    # ------------------------------------------------------------------
    def run(self):
        """
        Boucle principale de simulation.
        Chaque UAV lit sa cible dans self.target_pos.
        """
        sim_time = 0.0
        max_time = float(self.config["simulation"]["max_sim_time"])

        while (
            sim_time < max_time
            and p.isConnected(self.physics_client_id)
        ):
            # 1. Contrôle des drones
            for uav in self.agents:
                uav.think_and_act()

            # 2. Avancer la physique
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 3. Real time
            time.sleep(self.dt)
            sim_time += self.dt

    # ------------------------------------------------------------------
    def stop(self):
        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(self.physics_client_id)
