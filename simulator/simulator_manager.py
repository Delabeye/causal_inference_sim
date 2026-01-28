# simulator/simulator_manager.py
import time
import numpy as np
import pybullet as p
import pybullet_data

from environment.world import World
from entities.uav import UAV
from swarm.swarm import Swarm
from entities.static_sensor import RadarStation
from Control.Path_planning import HeightmapAStar 

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
        """
        Initializes the manager and connects to the physics engine.

        Args:
            config (dict): The complete configuration tree defining simulation, 
                   physics, world, and agent parameters.

        Raises:
            ConnectionError: If a connection to the PyBullet server cannot be established.
        """
        self.config = config
        self.dt = float(self.config["simulation"]["dt"])

        # 1. PyBullet Connection
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        mode = p.GUI if mode_str == "gui" else p.DIRECT
        self.physics_client_id = p.connect(mode)
        
        if self.physics_client_id < 0:
            raise ConnectionError("Unable to connect to PyBullet.")

        print(f"Connected to PyBullet, client_id={self.physics_client_id}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )

        # 2. World setup (ground + obstacles)
        self.world = World(self.physics_client_id)

        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, 1.0, 1.0],
                physicsClientId=self.physics_client_id,
            )

        # List of all agents (UAVs + radars)
        self.agents: list[UAV | RadarStation] = []

        # List of swarms (currently only one is created, but we maintain a list for scalability)
        self.swarms: list[Swarm] = []
        
        # List of radars
        self.radars: list[RadarStation] = []
        
        # 3. Load scenario (obstacles + drones + potential objectives)
        self.load_scenario()

        # 4. Create a swarm if requested in the configuration
        self._create_swarm_from_config()

    # ------------------------------------------------------------------
    def load_scenario(self):
        """
        Constructs the simulation environment and populates it with agents.

        This method performs the following sequential operations:
        1. Environment Setup: Generates a procedural city or loads a custom URDF.
        2. Planner Initialization: Builds a 2.5D heightmap for the path-planning engine.
        3. Agent Spawning: Instantiates Radar stations and UAVs based on config definitions.
        4. Objective Assignment: Links high-level goals (e.g., target positions) to 
        specific agents by name or index.
        """
        print("Scenario loading...")
        
        self.obstacles_config = self.config.get("world", {})
        print(self.obstacles_config)
        # Obstacles
        res=self.obstacles_config.get("res",0.25)
        world_type = self.obstacles_config.get("type","city")
        if world_type == "generated":
            #load buildings from procedural generation
            obstacles=self.world.generate_city_urdf(self.obstacles_config.get("city",{}))

            p.loadURDF(
            "assets/city.urdf",  
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
            world_file,  
            basePosition=[0, 0, 0],
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
            )
            self.planner = HeightmapAStar(
            self.obstacles_config.get("Astar",{}),
            resolution=res, 
            )
            self.planner.custom_heightmap()
        # UAVs and Radars
        # Optional per-run log directory (propagated to UAV configs)
        sim_log_dir = None
        if isinstance(self.config, dict):
            sim_log_dir = self.config.get("simulation", {}).get("log_dir", None)
        for agent_cfg in self.config.get("agents", []):

            # Propagate global log directory to each UAV config (if provided)
            if sim_log_dir and isinstance(agent_cfg, dict) and agent_cfg.get("type") == "uav":
                agent_cfg["log_dir"] = sim_log_dir
            
            if agent_cfg.get("type") == "radar":
                radar = RadarStation(config=agent_cfg, physics_client_id=self.physics_client_id, dt=self.dt)
                self.agents.append(radar) # It is added to the main loop for think_and_act.
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
                    # agent_id is a string (name)
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
            f"Scenario loaded : {len(self.agents)-len(self.radars)} UAVs, "
            f"{len(self.world.obstacle_ids)} obstacles."
        )

    # ------------------------------------------------------------------
    def _create_swarm_from_config(self):
        """
        Organizes UAV agents into swarms based on shared identifiers.

        Parses the swarm configuration section to assign leaders, separation margins, 
        and network parameters (IP/Ports for inter-agent communication). This method 
        groups agents by their 'swarm_id' and instantiates Swarm controllers to 
        manage collective dynamics.
        """
        # 1. Loading swarm configurations (YAML)
        # YAML is a list: [{id: ‘A’, ...}, {id: ‘B’, ...}]
        raw_swarm_configs = self.config.get("swarm", [])
        
        # We convert it into a dictionary for quick access: { ‘A’: {config}, ‘B’: {config} }
        swarm_configs_map = {}
        if isinstance(raw_swarm_configs, list):
            for cfg in raw_swarm_configs:
                sid = str(cfg.get("id"))
                swarm_configs_map[sid] = cfg
        elif isinstance(raw_swarm_configs, dict):
            # Cases where there is only one defined swarm without a hyphen
            sid = str(raw_swarm_configs.get("id", "default"))
            swarm_configs_map[sid] = raw_swarm_configs

        # 2. Grouping of Drones (Existing Code)
        swarms_groups = {}
        uavs = [a for a in self.agents if isinstance(a, UAV)]
        
        for uav in uavs:
            s_id = uav.config.get("swarm_id", None)
            if s_id is not None:
                s_id = str(s_id)
                if s_id not in swarms_groups:
                    swarms_groups[s_id] = []
                swarms_groups[s_id].append(uav)

        # 3. Creation of swarms
        if not swarms_groups:
            print("[Swarm] No “swarm_id” found on the drones.")
            return

        for s_id, members in swarms_groups.items():
            if len(members) < 2:
                print(f"[Swarm] Group “{s_id}”: Too small (<2). Ignored.")
                continue

            # Retrieve the configuration specific to this ID (e.g. ‘A’)
            # If no configuration is found in “swarm:”, use {} (default values)
            specific_cfg = swarm_configs_map.get(s_id, {})

            print(f"[Swarm] Group “{s_id}” created with configuration: {specific_cfg}")

            # Extraction of specific parameters
            leader_name = specific_cfg.get("leader", None) # Leader's name
            min_sep = float(specific_cfg.get("min_sep", 0.6))
            avoid_gain = float(specific_cfg.get("avoid_gain", 0.5))
            
            # Network parameters
            port_in = int(specific_cfg.get("port_in", 5556))
            port_out = int(specific_cfg.get("port_out", 5557))
            ip = specific_cfg.get("ip", "localhost")

            # Creation of the instance
            new_swarm = Swarm(
                agents=members,
                leader_name=leader_name,
                formation_body_offsets = None,
                min_sep=min_sep,
                avoid_gain=avoid_gain,
                port_in=port_in,
                port_out=port_out,
                ip=ip
            )
            self.swarms.append(new_swarm)
        

    # ------------------------------------------------------------------
    def run(self):
        """
        Executes the main simulation loop until completion.

        The loop follows a fixed-step synchronization pattern:
        1. Swarm Update: Resolves leader-follower logic and inter-agent distances.
        2. Agent Logic: Triggers the 'think_and_act' cycle for UAVs and periodic 
        scans for Radar stations.
        3. Physics Step: Advances the PyBullet world by 'dt'.
        4. Timing: Regulates execution speed to maintain consistency with 'dt'.

        Note:
            The loop terminates if 'max_sim_time' is reached or if the physics 
            server is disconnected.
        """
        sim_time = 0.0
        max_time = float(self.config['simulation']['max_sim_time'])

        while sim_time < max_time and p.isConnected(self.physics_client_id):
            # 1. Update swarms (leader/followers logic) if enabled
            for swarm in self.swarms:
                swarm.update()

            # 2. Process logic for each agent
            for agent in self.agents:
                if agent.type == 'uav':
                    agent.think_and_act()

                elif agent.type == 'radar':
                    # Check if the radar's refresh period has elapsed
                    if sim_time > agent.radar_period + agent.radar_last_time:
                        agent.radar_last_time = sim_time
                        agent.think_and_act(sim_time)
            
            # 3. Advance the physics engine step
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 4. Synchronize with real-time or maintain fixed time step
            time.sleep(self.dt)
            sim_time += self.dt

    # ------------------------------------------------------------------
    def stop(self):
        """Gracefully terminates the simulation and releases resources.

        Disconnects from the PyBullet server and triggers cleanup routines for 
        active swarms (e.g., closing network sockets or logging files).
        """
        if p.isConnected(self.physics_client_id):
            print("Disconnecting from PyBullet.")
            p.disconnect(self.physics_client_id)
            for swarm in self.swarms:
                swarm.cleanup()

    
