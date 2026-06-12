# simulator/simulator_manager.py
import time
import numpy as np
import pybullet as p
import pybullet_data
import json
import os
import re

from environment.world import World
from entities.uav import UAV
from swarm.swarm import Swarm
from entities.static_sensor import RadarStation
from Control.Path_planning import HeightmapAStar 
from swarm.swarmnetwork import SwarmNetwork

class SimulationManager:
    """SimulationManager
    Orchestrates a PyBullet-based multi-agent simulation with UAVs and radar stations.
    Responsibilities:
        - Establish connection to PyBullet physics engine (GUI or DIRECT mode)
        - Initialize simulation world (ground plane, obstacles, buildings)
        - Load scenario configuration (agents, objectives, obstacles)
        - Create and manage UAVs and RadarStation agents
        - Instantiate swarms with leader-follower dynamics if enabled
        - Execute main simulation loop with physics stepping and agent control
        - Handle resource cleanup and disconnection
    Attributes:
        config (dict): Complete simulation configuration from config.yaml
        dt (float): Physics simulation timestep (seconds)
        physics_client_id (int): PyBullet client identifier
        world (World): PyBullet world manager handling ground and obstacles
        agents (list[UAV | RadarStation]): All active agents in the simulation
        swarms (list[Swarm]): Swarm formations with leader-follower behavior
        radars (list[RadarStation]): Dedicated reference to radar agents
        planner (HeightmapAStar): Path planning algorithm using heightmap from buildings
        obstacles_config (dict): Configuration for world obstacles and buildings
    Methods:
        load_scenario(): Initialize agents, obstacles, and objectives from config
        _create_swarm_from_config(): Instantiate swarm formation if enabled
        run(): Main simulation loop (physics stepping + agent control)
        stop(): Cleanup and disconnect from PyBullet
    Configuration:
        Requires config dict with sections: simulation, physics, world, agents, objectives, swarm
        Supports both indexed (int) and named (str) agent references in objectives
    """


    def __init__(self, config: dict):
        self.config = config

        # On allume le proxy pour toutes les simulation
        self.network = SwarmNetwork(port_in=5556, port_out=5557)
        self.network.init_proxy()

        # Auto-détection de la run 
        log_dir = "logs"
        next_run_id = 0
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                match = re.search(r'run_(\d+)', filename)
                if match:
                    run_id = int(match.group(1))
                    if run_id >= next_run_id:
                        next_run_id = run_id + 1

        self.config["simulation"]["run_config"] = next_run_id
        for agent_cfg in self.config.get("agents", []):
            agent_cfg["run_config"] = next_run_id


        self.sim_time = 0.0
        self.dt = float(self.config["simulation"]["dt"])
        # 1. Connexion PyBullet
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        mode = p.GUI if mode_str == "gui" else p.DIRECT
        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("Impossible de se connecter à PyBullet.")
        
        # Pour le bon fonctionnement des logs d'entrainement
        self.is_gui_mode = True if mode_str == "gui" else False
        # Gerer la pause la simulation
        self.is_paused = False

        print(f"Connecté à PyBullet, client_id={self.physics_client_id}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )

        # 2. Monde (sol + obstacles)
        self.world = World(self.physics_client_id)

        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, 1.0, 1.0],
                physicsClientId=self.physics_client_id,
            )

            # Boutton pause et quit
            # self.btn_pause = p.addUserDebugParameter("Pause / Play", 1, -1, 1, physicsClientId = self.physics_client_id)
            # self.btn_quit = p.addUserDebugParameter("Quit Simulation", 1, -1, 1, physicsClientId = self.physics_client_id)

            # self.count_pause_clicks = 0
            # self.count_quit_click = 0

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
        
        self.obstacles_config = self.config.get("world", {})
        print(self.obstacles_config)
        # Obstacles
        res=self.obstacles_config.get("res",0.25)
        world_type = self.obstacles_config.get("type","city")
        if world_type == "generated":
            """Charge le sol + règle la physique."""
            obstacles=self.world.generate_city_urdf(self.obstacles_config.get("city",{}))

            p.loadURDF(
            "assets/city.urdf",  # <--- Votre nouveau fichier
            basePosition=[0, 0, 0],
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
            )
            self.planner = HeightmapAStar(
            self.obstacles_config.get("Astar",{}),
            resolution=res, 
            )
            self.planner.build_from_buildings(obstacles)
        
        if world_type == "custom":
            world_file = self.obstacles_config.get("filename")
            p.loadURDF(
            world_file,  # <--- Votre nouveau fichier
            basePosition=[0, 0, 0],
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
            )
            self.planner = HeightmapAStar(
            self.obstacles_config.get("Astar",{}),
            resolution=res 
            )
            self.planner.custom_heightmap()
        # Drones
        for agent_cfg in self.config.get("agents", []):
            
            if agent_cfg.get("type") == "radar":
                radar = RadarStation(config=agent_cfg, physics_client_id=self.physics_client_id, dt=self.dt)
                self.agents.append(radar) # On l'ajoute à la boucle principale pour le think_and_act
                self.radars.append(radar)

            elif agent_cfg.get("type") == "uav":
                uav = UAV(
                    config=agent_cfg,
                    physics_client_id=self.physics_client_id,
                    dt=self.dt,
                    known_obstacles_config=obstacles,
                    planner=self.planner,
                    world_type = world_type
                )
                self.agents.append(uav)

        for radar in self.radars:
            radar.targets = [a for a in self.agents if isinstance(a, UAV)]
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
        Crée les essaims en associant les drones par 'swarm_id' 
        et en appliquant la config spécifique définie dans le YAML.
        """
        # 1. Chargement des configs d'essaims (YAML)
        # Le YAML est une liste : [{id: "A", ...}, {id: "B", ...}]
        raw_swarm_configs = self.config.get("swarm", [])
        
        # On convertit en dictionnaire pour accès rapide : { "A": {config}, "B": {config} }
        swarm_configs_map = {}
        if isinstance(raw_swarm_configs, list):
            for cfg in raw_swarm_configs:
                sid = str(cfg.get("id"))
                swarm_configs_map[sid] = cfg
        elif isinstance(raw_swarm_configs, dict):
            # Cas où il n'y a qu'un seul essaim défini sans tiret
            sid = str(raw_swarm_configs.get("id", "default"))
            swarm_configs_map[sid] = raw_swarm_configs

        # 2. Regroupement des Drones (Code existant)
        swarms_groups = {}
        uavs = [a for a in self.agents if isinstance(a, UAV)]
        
        for uav in uavs:
            s_id = uav.config.get("swarm_id", None)
            if s_id is not None:
                s_id = str(s_id)
                if s_id not in swarms_groups:
                    swarms_groups[s_id] = []
                swarms_groups[s_id].append(uav)

        # 3. Création des objets Swarm
        if not swarms_groups:
            print("[Swarm] Aucun 'swarm_id' trouvé sur les drones.")
            return

        for s_id, members in swarms_groups.items():
            if len(members) < 2:
                print(f"[Swarm] Groupe '{s_id}' : Trop petit (<2). Ignoré.")
                continue

            # --- ICI EST LA CORRECTION ---
            # On récupère la config spécifique à cet ID (ex: "A")
            # Si pas de config trouvée dans 'swarm:', on utilise {} (valeurs par défaut)
            specific_cfg = swarm_configs_map.get(s_id, {})

            print(f"[Swarm] Création groupe '{s_id}' avec config : {specific_cfg}")

            # Extraction des paramètres spécifiques
            leader_name = specific_cfg.get("leader", None) # Le nom du drone leader
            min_sep = float(specific_cfg.get("min_sep", 0.6))
            avoid_gain = float(specific_cfg.get("avoid_gain", 0.5))
            
            # Paramètres Réseau
            ip = specific_cfg.get("ip", "localhost")

            # Création de l'instance
            new_swarm = Swarm(
                agents=members,
                leader_name=leader_name,
                formation_body_offsets = None,
                min_sep=min_sep,
                avoid_gain=avoid_gain,
                port_in=self.network.port_in,
                port_out=self.network.port_out,
                ip=ip
            )
            self.swarms.append(new_swarm)
        
    

    # ------------------------------------------------------------------
    def save_state(self, filename: str):
        
        os.makedirs("saves", exist_ok=True)
        filepath = os.path.join("saves", f"{filename}")
        if os.path.exists(filepath): os.remove(filepath)

        bullet_file = f"{filepath}.bullet"
        json_file = f"{filepath}.json"

        try:
            state_id = p.saveState()
            p.saveBullet(bullet_file, physicsClientId = self.physics_client_id)
        except p.error as e:
            print(f"Erreur lors du chargement: {e}")
            return
        
        print(f"Etat physique sauvegardé dans : {filepath}")


        logic_state = {
            "sim_time": self.sim_time, 
            "agents": {}
        
        }
        # On récupère l'état de chaque agent
        for idx, agent in enumerate(self.agents):
            if hasattr(agent, "get_state"):
                logic_state["agents"][idx] = agent.get_state()

        with open(json_file, 'w') as f:
            json.dump(logic_state, f, indent=4)

        print(f"Sauvegarde réussie : {bullet_file} et {json_file}")

        
    
    # ------------------------------------------------------------------
    def load_state(self, filepath: str):

        bullet_file = f"{filepath}.bullet"
        json_file = f"{filepath}.json"

        try:
            p.restoreState(filename = bullet_file, physicsClientId = self.physics_client_id)
        except p.error as e:
            print(f"Erreur lors du chargement : {e}")
            return
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                logic_state = json.load(f)
            
            self.sim_time = logic_state.get("sim_time", 0.0)

            saved_agents = logic_state.get("agents", {})
            for str_idx, agent_state in saved_agents.items():
                idx = int(str_idx)
                if idx < len(self.agents):
                    agent = self.agents[idx]
                    if hasattr(agent, "load_state"):
                        agent.load_state(agent_state)

            print(f"Chargement de la save réussi")
        else:
            print(f"Echec du chargment de : {json_file}. Seule la physique a été chargée.")
    

    
    # ------------------------------------------------------------------
    def run(self):
        """
        Boucle principale de simulation.
        """
        max_time = float(self.config["simulation"]["max_sim_time"])
        save_compteur = 0

        while self.sim_time < max_time and p.isConnected(self.physics_client_id):

            # Keyboard button, keys est un dictionnaire
            keys = p.getKeyboardEvents(physicsClientId = self.physics_client_id)

            # Touche Espace
            if keys.get(32) == 1:
                self.is_paused = not self.is_paused
                print(f"Pause via Clavier : {self.is_paused}")
            
            # Appuyer sur q en AZERTY
            if keys.get(97) == 1:
                print("Arrêt de la simulation")
                break

            # Sauvegarder avec s
            if keys.get(115) == 1:
                print(f"Sauvegarde {save_compteur} à l'instant t={self.sim_time}s")
                save_file_path = f"run_{self.config["simulation"]["run_config"]}_save_state_{save_compteur}"
                # Seulement mettre le nom du fichier
                self.save_state(save_file_path)
                save_compteur += 1

            # GUI button pause/play and quit simulation
            # if self.is_gui_mode:
            #     pause_click = p.readUserDebugParameter(self.btn_pause, physicsClientId = self.physics_client_id)
            #     quit_clicks = p.readUserDebugParameter(self.btn_quit, physicsClientId = self.physics_client_id)

            #     if quit_clicks > self.count_quit_click:
            #         print("\n Button QUIT pressed. Stopping the simulation")
            #         break
                
            #     if pause_click > self.count_pause_clicks:
            #         self.is_paused = not self.is_paused
            #         self.count_pause_clicks = pause_click
            #         etat = "PAUSED" if self.is_paused else "PLAYING"
            #         print(f"\n Simulation {etat} (t={round(self.sim_time, 2)}s)")


            # 1. Mise à jour des essaims (leader/followers) si activés
            if not self.is_paused:
                for swarm in self.swarms:
                    swarm.update()

                # 2. Contrôle de chaque drone
                for agent in self.agents:
                    if agent.type=="uav":
                        agent.think_and_act()

                    elif agent.type=="radar":
                        if self.sim_time > agent.radar_period + agent.radar_last_time:
                            agent.radar_last_time = self.sim_time
                            agent.think_and_act(self.sim_time)
                # 3. Avancer la physique
                p.stepSimulation(physicsClientId=self.physics_client_id)
                self.sim_time += self.dt



                # 4. Real time
            if self.is_gui_mode:
                time.sleep(self.dt)


            

    # ------------------------------------------------------------------
    def stop(self):
        for agent in self.agents:
            if agent.type == "uav":
                if not os.path.exists(agent.log_file):
                    agent.write_csv()

        for swarm in self.swarms:
                swarm.cleanup()

        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(self.physics_client_id)

        self.network.stop_proxy()



        
    # ------------------------------------------------------------------
    def it_stop(self, ind: int):

        for agent in self.agents:
            if agent.type == "uav":
                if not os.path.exists(agent.log_file):
                    agent.write_csv()

        for swarm in self.swarms:
                swarm.cleanup()
        

        self.config["simulation"]["run_config"] = ind
        for agent_cfg in self.config.get("agents", []):
            agent_cfg["run_config"] = ind

        if p.isConnected(self.physics_client_id):
            print(f"Reset de PyBullet pour la fin de la run {ind}.")

        # return(self.config)
        
    
    def initialisation(self):
        self.sim_time = 0.0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )

        # 2. Monde (sol + obstacles)
        self.world = World(self.physics_client_id)

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


    def reset(self):
        p.resetSimulation(self.physics_client_id)
        self.initialisation()
        
        
        
    
    def disconnect(self):
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)
            print("Déconnection de PyBullet")
        self.network.stop_proxy()
