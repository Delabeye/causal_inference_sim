# simulator/simulator_manager.py
import time
import numpy as np
import pybullet as p
import pybullet_data

from environment.world import World
from entities.uav import UAV
from entities.obstacles import CubeObstacle, SphericalObstacle, CylindricalObstacle
from swarm.swarm import Swarm
from entities.static_sensor import RadarStation

class SimulationManager:
    """
    Orchestration de la simulation :
      - connexion à PyBullet
      - création du monde (sol + obstacles)
      - création des UAV depuis config.yaml
      - création éventuelle d'un essaim (Swarm) selon la config
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
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )

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

        # Liste de tous les agents (UAV + radars)
        self.agents: list[UAV | RadarStation] = []

        # Liste des essaims (on n'en crée qu'un, mais on garde une liste)
        self.swarms: list[Swarm] = []
        
        # Liste des radars
        self.radars: list[RadarStation] = []
        
        # 3. Charger scénario (obstacles + drones + objectifs éventuels)
        self.load_scenario()

        # 4. Créer un essaim si demandé dans la config
        self._create_swarm_from_config()

    # ------------------------------------------------------------------
    def load_scenario(self):
        print("Chargement du scénario...")
        known_obstacles_agent = []
        self.known_obstacles_config = self.config.get("world", {}).get("obstacles", [])
        # Obstacles
        for obs_cfg in self.known_obstacles_config:
            otype = obs_cfg.get("type")
            if otype == "cube":
                obstacle = CubeObstacle(
                    center=obs_cfg["center"],
                    length=obs_cfg["length"],
                    width=obs_cfg["width"],
                    height=obs_cfg["height"],
                    obstacle_id=obs_cfg["id"]
                )
                self.world.add_cube_obstacle(obstacle)

            elif otype == "sphere":
                obstacle = SphericalObstacle(
                    center=obs_cfg["center"],
                    radius=obs_cfg["radius"],
                    obstacle_id=obs_cfg["id"]
                )
                self.world.add_sphere_obstacle(obstacle)

            elif otype == "cylinder":
                obstacle = CylindricalObstacle(
                    center=obs_cfg["center"],
                    radius=obs_cfg["radius"],
                    height=obs_cfg["height"],
                    obstacle_id=obs_cfg["id"]
                )
                self.world.add_cylindrical_obstacle(obstacle)
        for obs in self.known_obstacles_config :
            if obs.get("knowledge", False) == True :
                known_obstacles_agent.append(obs)
        # Drones
        for agent_cfg in self.config.get("agents", []):
            if agent_cfg.get("type") == "uav":
                uav = UAV(
                    config=agent_cfg,
                    physics_client_id=self.physics_client_id,
                    dt=self.dt,
                    known_obstacles_config=known_obstacles_agent,
                )
                self.agents.append(uav)

            elif agent_cfg.get("type") == "radar":
                radar = RadarStation(config=agent_cfg, physics_client_id=self.physics_client_id, dt=self.dt)
                self.agents.append(radar) # On l'ajoute à la boucle principale pour le think_and_act
                self.radars.append(radar)

        # Objectifs (ancienne mécanique, on la garde pour compatibilité)
        for objective in self.config.get("objectives", []):
            agent_id = objective.get("agent")
            if objective.get("type") == "reach_position":
                if agent_id is None:
                    raise ValueError(
                        "Objective of type 'reach_position' is missing 'agent'."
                    )
                setpoint = np.array(
                    objective.get("target_pos", [0.0, 0.0, 0.0]),
                    dtype=float,
                )

                # assign target to the matching agent (by index or by name)
                if isinstance(agent_id, int):
                    if 0 <= agent_id < len(self.agents):
                        agent = self.agents[agent_id]
                        if hasattr(agent, "set_target_pos"):
                            agent.set_target_pos(setpoint)
                        elif hasattr(agent, "set_target_position"):
                            agent.set_target_position(setpoint)
                        elif hasattr(agent, "set_target"):
                            agent.set_target(setpoint)
                        else:
                            setattr(agent, "target_pos", setpoint)
                    else:
                        raise ValueError(
                            f"Agent index {agent_id} out of range for objective."
                        )
                else:
                    # agent_id est un nom
                    found = False
                    for agent in self.agents:
                        if getattr(agent, "name", None) == agent_id:
                            if hasattr(agent, "set_target_pos"):
                                agent.set_target_pos(setpoint)
                            elif hasattr(agent, "set_target_position"):
                                agent.set_target_position(setpoint)
                            elif hasattr(agent, "set_target"):
                                agent.set_target(setpoint)
                            else:
                                setattr(agent, "target_pos", setpoint)
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"No agent with name '{agent_id}' found for objective."
                        )

        print(
            f"Scénario chargé : {len(self.agents)} drones, "
            f"{len(self.world.obstacle_ids)} obstacles."
        )

    # ------------------------------------------------------------------
    def _create_swarm_from_config(self):
        """
        Crée un essaim en fonction de la section 'swarm' du config.yaml.

        config.yaml :
          swarm:
            enabled: true/false
            leader: "drone_0"
            min_sep: 0.6
            avoid_gain: 0.5
        """
        swarm_cfg = self.config.get("swarm", {})
        enabled = bool(swarm_cfg.get("enabled", False))

        if not enabled:
            print("[Swarm] Essaim désactivé dans la config.")
            return

        uavs = [a for a in self.agents if isinstance(a, UAV)]
        if len(uavs) < 2:
            print("[Swarm] Moins de 2 UAV, aucun essaim créé.")
            return

        leader_name = swarm_cfg.get("leader", None)
        min_sep = float(swarm_cfg.get("min_sep", 0.6))
        avoid_gain = float(swarm_cfg.get("avoid_gain", 0.5))

        swarm = Swarm(
            uavs,
            leader_name=leader_name,
            min_sep=min_sep,
            avoid_gain=avoid_gain,
        )
        self.swarms.append(swarm)
        print(
            f"[Swarm] Essaim activé (leader='{swarm.leader.name}', "
            f"{len(swarm.followers)} follower(s))."
        )

    # ------------------------------------------------------------------
    def run(self):
        """
        Boucle principale de simulation.
        """
        sim_time = 0.0
        max_time = float(self.config["simulation"]["max_sim_time"])

        while sim_time < max_time and p.isConnected(self.physics_client_id):
            # 1. Mise à jour des essaims (leader/followers) si activés
            for swarm in self.swarms:
                swarm.update()

            # 2. Contrôle de chaque drone
            for uav in self.agents:
                uav.think_and_act()

            # 3. Avancer la physique
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 4. Real time
            time.sleep(self.dt)
            sim_time += self.dt

    # ------------------------------------------------------------------
    def stop(self):
        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(self.physics_client_id)
            for swarm in self.swarms:
                swarm.cleanup()