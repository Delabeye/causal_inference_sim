# simulator/simulator_manager.py
import time
import numpy as np
import pybullet as p
import pybullet_data
import json
import os
import re
import random
from dataclasses import replace

from environment.world import World
from entities.uav import UAV
from swarm.swarm import Swarm
from entities.static_sensor import RadarStation
from Control.Path_planning import HeightmapAStar 
from swarm.swarmnetwork import SwarmNetwork
from simulator.interventions import (
    InterventionRuntime,
    load_intervention_events,
    validate_intervention_targets,
)
from simulator.learning_trace_logger import LearningTraceLogger
from simulator.relational_ground_truth_logger import RelationalGroundTruthLogger
from simulator.state_checkpoint import SimulationCheckpoint

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

        # Auto-détection de la run, sauf pour les batchs expérimentaux seedés
        # qui doivent pouvoir fixer explicitement le run_id.
        sim_cfg = self.config.setdefault("simulation", {})
        self.deterministic_execution = bool(
            sim_cfg.get("deterministic_execution", False)
        )
        # ZeroMQ is intentionally bypassed for paired experiments. Its delivery
        # depends on wall-clock thread scheduling and made two identically seeded
        # DIRECT runs diverge before an intervention. In deterministic mode the
        # Swarm object uses a simulation-time message queue instead.
        self.network = None
        if not self.deterministic_execution:
            self.network = SwarmNetwork(
                port_in=int(sim_cfg.get("port_in", 5556)),
                port_out=int(sim_cfg.get("port_out", 5557)),
            )
        self.log_dir = str(sim_cfg.get("log_dir", "logs"))
        if bool(sim_cfg.get("fixed_run_config", False)):
            next_run_id = int(sim_cfg.get("run_config", 0))
        else:
            next_run_id = 0
            if os.path.exists(self.log_dir):
                for filename in os.listdir(self.log_dir):
                    match = re.search(r'run_(\d+)', filename)
                    if match:
                        run_id = int(match.group(1))
                        if run_id >= next_run_id:
                            next_run_id = run_id + 1

        self.config["simulation"]["run_config"] = next_run_id
        shared_formation_control = self.config.get("formation_control", {})
        if shared_formation_control is None:
            shared_formation_control = {}
        if not isinstance(shared_formation_control, dict):
            raise ValueError("formation_control must be a YAML mapping.")
        for agent_cfg in self.config.get("agents", []):
            agent_cfg["run_config"] = next_run_id
            agent_cfg["log_dir"] = self.log_dir
            agent_cfg["deterministic_communication"] = (
                self.deterministic_execution
            )
            agent_formation_control = agent_cfg.get("formation_control", {})
            if agent_formation_control is None:
                agent_formation_control = {}
            if not isinstance(agent_formation_control, dict):
                raise ValueError(
                    f"formation_control for agent {agent_cfg.get('name')} "
                    "must be a YAML mapping."
                )
            agent_cfg["formation_control"] = {
                **shared_formation_control,
                **agent_formation_control,
            }

        self.effective_seed = None
        base_seed = self.config["simulation"].get("seed", None)
        if base_seed is not None:
            seed_offset = next_run_id if bool(self.config["simulation"].get("seed_add_run_id", True)) else 0
            self.effective_seed = int(base_seed) + seed_offset
            random.seed(self.effective_seed)
            np.random.seed(self.effective_seed)
            self.config["simulation"]["effective_seed"] = self.effective_seed
            print(f"[Seed] run={next_run_id}, effective_seed={self.effective_seed}")
        if self.deterministic_execution:
            seed_root = int(
                self.effective_seed if self.effective_seed is not None else 0
            )
            for agent_cfg in self.config.get("agents", []):
                agent_name = str(agent_cfg.get("name", agent_cfg.get("type", "agent")))
                stable_offset = sum(
                    (idx + 1) * ord(char) for idx, char in enumerate(agent_name)
                )
                agent_cfg["deterministic_seed"] = seed_root + stable_offset

        self.sim_time = 0.0
        self.dt = float(self.config["simulation"]["dt"])
        # 1. Connexion PyBullet
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        mode = p.GUI if mode_str == "gui" else p.DIRECT
        connect_options = "--numThreads=1" if self.deterministic_execution else ""
        self.physics_client_id = p.connect(mode, options=connect_options)
        if self.physics_client_id < 0:
            raise ConnectionError("Impossible de se connecter à PyBullet.")
        
        self.is_gui_mode = mode_str == "gui"

        print(f"Connecté à PyBullet, client_id={self.physics_client_id}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )
        p.setTimeStep(self.dt, physicsClientId=self.physics_client_id)
        if self.deterministic_execution:
            p.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self.physics_client_id,
            )
        self.contact_dynamics_cfg = self.config.get("physics", {}).get("contact", {})

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

        self.agents: list[UAV | RadarStation] = []
        self.swarms: list[Swarm] = []
        self.formation_run_metadata = {}
        self.intervention_events = load_intervention_events(
            self.config.get("interventions", {})
        )
        self.intervention_runtime = InterventionRuntime(self.intervention_events)
        self._last_intervention_applications = []
        fork_cfg = self.config.get("counterfactual_forks", {}) or {}
        self.counterfactual_forks_enabled = bool(fork_cfg.get("enabled", False))
        self.counterfactual_fork_horizon = float(
            fork_cfg.get(
                "rollout_horizon_s",
                self.config.get("experiment", {})
                .get("snapshot_schedule", {})
                .get("rollout_horizon_s", 8.0),
            )
        )
        self.counterfactual_fork_records = []
        self.counterfactual_fork_manifest_path = os.path.join(
            self.log_dir,
            f"run_{next_run_id}_counterfactual_forks.json",
        )
        self.relational_logger = None
        self.learning_trace_logger = None
        self._last_relational_control_indices = None
        self._run_artifacts_finalized = False
        self.radars: list[RadarStation] = []

        self.load_scenario()
        self._create_swarm_from_config()
        self._initialize_run_loggers()

    def _initialize_run_loggers(self):
        run_id = int(self.config["simulation"].get("run_config", 0))
        uavs = [agent for agent in self.agents if isinstance(agent, UAV)]
        validate_intervention_targets(
            self.intervention_events,
            (agent.name for agent in uavs),
        )
        self.relational_logger = RelationalGroundTruthLogger(
            log_dir=self.log_dir,
            run_id=run_id,
            physics_client_id=self.physics_client_id,
        )
        self.learning_trace_logger = LearningTraceLogger(
            log_dir=self.log_dir,
            run_id=run_id,
            uavs=uavs,
            events=self.intervention_events,
        )
        self._last_relational_control_indices = tuple(
            (agent.name, int(getattr(agent, "control_update_index", 0)))
            for agent in uavs
        )
        self._run_artifacts_finalized = False

    def _log_control_step_artifacts(self):
        uavs = [agent for agent in self.agents if isinstance(agent, UAV)]
        current_indices = tuple(
            (agent.name, int(getattr(agent, "control_update_index", 0)))
            for agent in uavs
        )
        if current_indices == self._last_relational_control_indices:
            return
        self._last_relational_control_indices = current_indices
        log_time = max((float(agent._sim_time) for agent in uavs), default=float(self.sim_time))
        if self.relational_logger is not None:
            self.relational_logger.log_step(log_time, uavs)
        if self.learning_trace_logger is not None:
            self.learning_trace_logger.log_step(
                log_time,
                self._last_intervention_applications,
            )

    def _apply_controlled_interventions(self):
        uavs = [agent for agent in self.agents if isinstance(agent, UAV)]
        self._last_intervention_applications = self.intervention_runtime.apply(
            self.sim_time,
            uavs,
        )

    def _advance_simulation_step(
        self,
        *,
        apply_interventions: bool,
        log_main_artifacts: bool,
    ) -> None:
        if apply_interventions:
            self._apply_controlled_interventions()
        else:
            self._last_intervention_applications = []
            for agent in self.agents:
                if isinstance(agent, UAV):
                    agent.clear_intervention_state()

        for swarm in self.swarms:
            swarm.update()
        for agent in self.agents:
            if isinstance(agent, UAV):
                agent.think_and_act()
            elif isinstance(agent, RadarStation):
                if self.sim_time > agent.radar_period + agent.radar_last_time:
                    agent.radar_last_time = self.sim_time
                    agent.think_and_act(self.sim_time)

        p.stepSimulation(physicsClientId=self.physics_client_id)
        if log_main_artifacts:
            self._log_control_step_artifacts()
        self.sim_time += self.dt

    @staticmethod
    def _control_indices(uavs):
        return tuple(
            (agent.name, int(getattr(agent, "control_update_index", 0)))
            for agent in uavs
        )

    def _run_counterfactual_branch(
        self,
        event,
        event_index: int,
        checkpoint: SimulationCheckpoint,
    ) -> None:
        checkpoint.restore()
        actual_time = float(self.sim_time)
        branch_event = replace(
            event,
            start_time=actual_time,
            end_time=actual_time + event.duration,
        )
        branch_runtime = InterventionRuntime([branch_event])
        original_runtime = self.intervention_runtime
        uavs = [agent for agent in self.agents if isinstance(agent, UAV)]
        artifact_stem = f"run_{self.config['simulation']['run_config']}_fork_{event_index:03d}"
        branch_logger = LearningTraceLogger(
            log_dir=self.log_dir,
            run_id=int(self.config["simulation"]["run_config"]),
            uavs=uavs,
            events=[branch_event],
            artifact_stem=artifact_stem,
        )
        branch_last_indices = self._control_indices(uavs)
        branch_error = None
        try:
            self.intervention_runtime = branch_runtime
            for uav in uavs:
                uav.logging_enabled = False
            branch_end = actual_time + self.counterfactual_fork_horizon
            while self.sim_time < branch_end - 1e-12:
                self._advance_simulation_step(
                    apply_interventions=True,
                    log_main_artifacts=False,
                )
                current_indices = self._control_indices(uavs)
                if current_indices == branch_last_indices:
                    continue
                branch_last_indices = current_indices
                log_time = max(
                    (float(agent._sim_time) for agent in uavs),
                    default=float(self.sim_time),
                )
                branch_logger.log_step(
                    log_time,
                    self._last_intervention_applications,
                )
        except Exception as exc:
            branch_error = exc
        finally:
            branch_logger.close()
            self.intervention_runtime = original_runtime

        if branch_error is not None:
            raise branch_error
        self.counterfactual_fork_records.append(
            {
                "fork_index": event_index,
                "event": event.as_dict(),
                "scheduled_time": event.start_time,
                "actual_snapshot_time": actual_time,
                "rollout_horizon_s": self.counterfactual_fork_horizon,
                "baseline_trace": str(self.learning_trace_logger.path),
                "intervention_trace": str(branch_logger.path),
                "event_catalog": str(branch_logger.event_catalog_path),
                "checkpoint_backend": "pybullet_state_plus_python_logic",
            }
        )

    def _write_counterfactual_fork_manifest(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        payload = {
            "run": int(self.config["simulation"].get("run_config", 0)),
            "matrix_convention": "row=receiver,column=sender",
            "parent_branch": "baseline_without_interventions",
            "forks": self.counterfactual_fork_records,
        }
        with open(
            self.counterfactual_fork_manifest_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2)

    def _run_with_counterfactual_forks(self) -> None:
        if not self.deterministic_execution or self.is_gui_mode:
            raise RuntimeError(
                "counterfactual_forks requires DIRECT mode and deterministic_execution=true."
            )
        if self.counterfactual_fork_horizon <= 0.0:
            raise ValueError("counterfactual fork rollout_horizon_s must be positive.")

        max_time = float(self.config["simulation"]["max_sim_time"])
        events = sorted(self.intervention_events, key=lambda item: item.start_time)
        for event_index, event in enumerate(events):
            if event.start_time + self.counterfactual_fork_horizon > max_time + 1e-9:
                raise ValueError(
                    f"Fork '{event.id}' does not fit before max_sim_time={max_time}."
                )

        snapshots = []
        next_event = 0
        while self.sim_time < max_time and p.isConnected(self.physics_client_id):
            while (
                next_event < len(events)
                and self.sim_time + 1e-12 >= events[next_event].start_time
            ):
                snapshots.append(
                    (
                        next_event,
                        events[next_event],
                        SimulationCheckpoint.capture(self),
                    )
                )
                next_event += 1
            self._advance_simulation_step(
                apply_interventions=False,
                log_main_artifacts=True,
            )

        final_baseline = SimulationCheckpoint.capture(self)
        try:
            for event_index, event, checkpoint in snapshots:
                self._run_counterfactual_branch(event, event_index, checkpoint)
        finally:
            self.intervention_runtime = InterventionRuntime(self.intervention_events)
            final_baseline.restore()
            final_baseline.release()
            for _, _, checkpoint in snapshots:
                checkpoint.release()

        self._write_counterfactual_fork_manifest()
        self._finalize_run_artifacts()

    # ------------------------------------------------------------------
    def _point_building_clearance_xy(self, point_xy, buildings):
        best = float("inf")
        px, py = float(point_xy[0]), float(point_xy[1])

        for building in buildings:
            center = building.get("center", [0.0, 0.0, 0.0])
            bx, by = float(center[0]), float(center[1])
            width = float(building.get("width", 0.0))
            length = float(building.get("length", 0.0))

            dx = abs(px - bx) - width / 2.0
            dy = abs(py - by) - length / 2.0
            clearance = float(np.hypot(max(dx, 0.0), max(dy, 0.0)))
            best = min(best, clearance)

        return best if np.isfinite(best) else 100.0

    def _sample_random_waypoint(self, rng, bounds, altitude_range, buildings, min_clearance, max_attempts, map_margin):
        x_min, x_max = bounds.get("x", [-60.0, 60.0])
        y_min, y_max = bounds.get("y", [-60.0, 60.0])
        z_min, z_max = altitude_range
        x_min = float(x_min) + map_margin
        x_max = float(x_max) - map_margin
        y_min = float(y_min) + map_margin
        y_max = float(y_max) - map_margin

        if x_min >= x_max or y_min >= y_max:
            raise ValueError("random_waypoints.map_margin is too large for the configured world bounds.")

        for _ in range(max_attempts):
            candidate = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    rng.uniform(float(z_min), float(z_max)),
                ],
                dtype=float,
            )
            if self._point_building_clearance_xy(candidate[:2], buildings) >= min_clearance:
                return candidate

        raise RuntimeError(
            "Impossible de générer un waypoint hors bâtiment. "
            "Réduis min_clearance ou élargis world_bounds."
        )

    def _apply_random_waypoints(self, buildings):
        rwp_cfg = self.config.get("random_waypoints", {})
        if not rwp_cfg or not rwp_cfg.get("enabled", False):
            return

        run_id = int(self.config["simulation"]["run_config"])
        base_seed = rwp_cfg.get("seed", None)
        add_run_id = bool(rwp_cfg.get("seed_add_run_id", True))
        seed_offset = run_id if add_run_id else 0
        rng_seed = run_id if base_seed is None else int(base_seed) + seed_offset
        rwp_cfg["effective_seed"] = rng_seed
        rng = random.Random(rng_seed)

        count = int(rwp_cfg.get("count", 3))
        min_clearance = float(rwp_cfg.get("min_building_clearance", 2.0))
        map_margin = float(rwp_cfg.get("map_margin", 8.0))
        max_attempts = int(rwp_cfg.get("max_attempts", 5000))
        altitude_range = rwp_cfg.get("altitude_range", [1.0, 10.0])
        target_agents = rwp_cfg.get("agents", ["drone_0"])
        if isinstance(target_agents, str):
            target_agents = [target_agents]

        astar_cfg = self.obstacles_config.get("Astar", {})
        bounds = rwp_cfg.get("world_bounds", astar_cfg.get("world_bounds", {"x": [-60, 60], "y": [-60, 60]}))

        generated = {}
        for agent_cfg in self.config.get("agents", []):
            if agent_cfg.get("type") != "uav":
                continue
            name = agent_cfg.get("name")
            if name not in target_agents:
                continue

            waypoints = [
                self._sample_random_waypoint(
                    rng=rng,
                    bounds=bounds,
                    altitude_range=altitude_range,
                    buildings=buildings,
                    min_clearance=min_clearance,
                    max_attempts=max_attempts,
                    map_margin=map_margin,
                ).round(3).tolist()
                for _ in range(count)
            ]
            agent_cfg["waypoints"] = waypoints
            generated[name] = waypoints

        if generated:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"run_{run_id}_waypoints.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "run": run_id,
                        "seed": rng_seed,
                        "min_building_clearance": min_clearance,
                        "map_margin": map_margin,
                        "waypoints": generated,
                    },
                    handle,
                    indent=2,
                )
            print(f"[Waypoints] Random waypoints saved to {path}: {generated}")

    def _apply_contact_dynamics(self, body_id, *, apply_to_base=True):
        if not self.contact_dynamics_cfg:
            return

        kwargs = {}
        for cfg_key, bullet_key in (
            ("restitution", "restitution"),
            ("lateral_friction", "lateralFriction"),
            ("rolling_friction", "rollingFriction"),
            ("spinning_friction", "spinningFriction"),
            ("contact_damping", "contactDamping"),
            ("contact_stiffness", "contactStiffness"),
        ):
            if cfg_key in self.contact_dynamics_cfg:
                kwargs[bullet_key] = float(self.contact_dynamics_cfg[cfg_key])

        if not kwargs:
            return

        link_indices = list(
            range(p.getNumJoints(body_id, physicsClientId=self.physics_client_id))
        )
        if apply_to_base:
            link_indices.insert(0, -1)

        for link_id in link_indices:
            p.changeDynamics(
                body_id,
                link_id,
                physicsClientId=self.physics_client_id,
                **kwargs,
            )

    # ------------------------------------------------------------------
    def load_scenario(self):
        print("Chargement du scénario...")
        
        self.obstacles_config = self.config.get("world", {})
        print(self.obstacles_config)
        # Obstacles
        res=self.obstacles_config.get("res",0.25)
        world_type = self.obstacles_config.get("type","city")
        obstacles = []
        if world_type == "generated":
            """Charge le sol + règle la physique."""
            obstacles=self.world.generate_city_urdf(self.obstacles_config.get("city",{}))

            world_body_id = p.loadURDF(
            "assets/city.urdf",  # <--- Votre nouveau fichier
            basePosition=[0, 0, 0],
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
            )
            self._apply_contact_dynamics(world_body_id)
            self.planner = HeightmapAStar(
            self.obstacles_config.get("Astar",{}),
            resolution=res, 
            )
            self.planner.build_from_buildings(obstacles)
            self._apply_random_waypoints(obstacles)
        
        if world_type == "custom":
            world_file = self.obstacles_config.get("filename")
            world_body_id = p.loadURDF(
            world_file,  # <--- Votre nouveau fichier
            basePosition=[0, 0, 0],
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
            )
            self._apply_contact_dynamics(world_body_id)
            self.planner = HeightmapAStar(
            self.obstacles_config.get("Astar",{}),
            resolution=res 
            )
            self.planner.custom_heightmap()
        # Drones
        astar_cfg = self.obstacles_config.get("Astar", {})
        world_bounds = astar_cfg.get("world_bounds", None)
        default_bounds_margin = float(astar_cfg.get("world_bounds_margin", 0.0))
        for agent_cfg in self.config.get("agents", []):
            if agent_cfg.get("type") == "uav" and world_bounds is not None:
                agent_cfg["world_bounds"] = world_bounds
                agent_cfg.setdefault("world_bounds_margin", default_bounds_margin)
            
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
                self._apply_contact_dynamics(uav.bodyId)
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
    def _resolve_swarm_leader(self, members, leader_name):
        if leader_name is None:
            return members[0]

        for member in members:
            if getattr(member, "name", None) == leader_name:
                return member

        raise ValueError(f"Aucun UAV avec name='{leader_name}' trouvé pour le leader.")

    def _normalize_formation_config(self, swarm_cfg):
        formation_cfg = swarm_cfg.get("formation", {})
        if formation_cfg is None:
            formation_cfg = {}
        elif isinstance(formation_cfg, str):
            formation_cfg = {"type": formation_cfg}
        elif not isinstance(formation_cfg, dict):
            raise ValueError("swarm.formation doit être une chaîne ou un dictionnaire YAML.")

        cfg = dict(formation_cfg)

        legacy_keys = {
            "formation_type": "type",
            "formation_mode": "mode",
            "formation_candidates": "candidates",
            "formation_spacing": "spacing",
            "formation_spacing_x": "spacing_x",
            "formation_spacing_y": "spacing_y",
            "formation_spacing_z": "spacing_z",
            "formation_seed": "seed",
            "formation_body_offsets": "offsets",
        }
        for old_key, new_key in legacy_keys.items():
            if old_key in swarm_cfg and new_key not in cfg:
                cfg[new_key] = swarm_cfg[old_key]

        return cfg

    def _formation_spacing(self, cfg):
        spacing = cfg.get("spacing", 1.0)
        if isinstance(spacing, (list, tuple)):
            if len(spacing) == 0:
                spacing_x = spacing_y = 1.0
                spacing_z = 0.0
            elif len(spacing) == 1:
                spacing_x = spacing_y = float(spacing[0])
                spacing_z = 0.0
            elif len(spacing) == 2:
                spacing_x, spacing_y = map(float, spacing)
                spacing_z = 0.0
            else:
                spacing_x, spacing_y, spacing_z = map(float, spacing[:3])
        else:
            spacing_x = spacing_y = float(spacing)
            spacing_z = 0.0

        spacing_x = float(cfg.get("spacing_x", spacing_x))
        spacing_y = float(cfg.get("spacing_y", spacing_y))
        spacing_z = float(cfg.get("spacing_z", spacing_z))
        return spacing_x, spacing_y, spacing_z

    def _parse_custom_formation_offsets(self, offsets_cfg, followers):
        if offsets_cfg is None:
            return None

        if isinstance(offsets_cfg, dict):
            offsets = {}
            for follower in followers:
                if follower.name not in offsets_cfg:
                    raise ValueError(
                        f"formation.offsets ne contient pas d'offset pour '{follower.name}'."
                    )
                offsets[follower.name] = np.array(offsets_cfg[follower.name], dtype=float)
        elif isinstance(offsets_cfg, list):
            if len(offsets_cfg) != len(followers):
                raise ValueError(
                    "formation.offsets en liste doit contenir exactement un offset par follower."
                )
            offsets = {
                follower.name: np.array(offset, dtype=float)
                for follower, offset in zip(followers, offsets_cfg)
            }
        else:
            raise ValueError("formation.offsets doit être un dictionnaire ou une liste.")

        for offset in offsets.values():
            if offset.shape != (3,):
                raise ValueError("Chaque offset de formation doit être un vecteur 3D [x, y, z].")

        return offsets

    def _generate_formation_offsets(self, formation_type, followers, spacing_x, spacing_y, spacing_z):
        formation_type = str(formation_type).strip().lower()
        formation_type = {
            "triangular": "triangle",
            "single_file": "trail",
            "follow": "trail",
            "follow_the_leader": "trail",
            "v_shape": "v",
            "vee": "v",
        }.get(formation_type, formation_type)

        offsets = {}
        n = len(followers)
        if n == 0:
            return offsets

        if formation_type == "triangle":
            idx = 0
            row = 1
            while idx < n:
                num_in_row = row
                center = 0.5 * (num_in_row - 1)
                for j in range(num_in_row):
                    if idx >= n:
                        break
                    offsets[followers[idx].name] = np.array(
                        [-row * spacing_x, (j - center) * spacing_y, spacing_z],
                        dtype=float,
                    )
                    idx += 1
                row += 1
        elif formation_type in {"trail", "column"}:
            for idx, follower in enumerate(followers):
                offsets[follower.name] = np.array(
                    [-(idx + 1) * spacing_x, 0.0, (idx + 1) * spacing_z],
                    dtype=float,
                )
        elif formation_type == "line":
            for idx, follower in enumerate(followers):
                offsets[follower.name] = np.array(
                    [-(idx + 1) * spacing_x, 0.0, (idx + 1) * spacing_z],
                    dtype=float,
                )
        elif formation_type == "v":
            for idx, follower in enumerate(followers):
                rank = (idx // 2) + 1
                side = -1.0 if idx % 2 == 0 else 1.0
                offsets[follower.name] = np.array(
                    [-rank * spacing_x, side * rank * spacing_y, rank * spacing_z],
                    dtype=float,
                )
        elif formation_type == "echelon_left":
            for idx, follower in enumerate(followers):
                rank = idx + 1
                offsets[follower.name] = np.array(
                    [-rank * spacing_x, -rank * spacing_y, rank * spacing_z],
                    dtype=float,
                )
        elif formation_type == "echelon_right":
            for idx, follower in enumerate(followers):
                rank = idx + 1
                offsets[follower.name] = np.array(
                    [-rank * spacing_x, rank * spacing_y, rank * spacing_z],
                    dtype=float,
                )
        elif formation_type == "diamond":
            base = [
                [-spacing_x, 0.0, spacing_z],
                [-2.0 * spacing_x, -spacing_y, 2.0 * spacing_z],
                [-2.0 * spacing_x, spacing_y, 2.0 * spacing_z],
                [-3.0 * spacing_x, 0.0, 3.0 * spacing_z],
            ]
            for idx, follower in enumerate(followers):
                if idx < len(base):
                    offset = base[idx]
                else:
                    row = idx - len(base) + 4
                    side = -1.0 if idx % 2 == 0 else 1.0
                    offset = [-row * spacing_x, side * spacing_y, row * spacing_z]
                offsets[follower.name] = np.array(offset, dtype=float)
        else:
            allowed = "triangle, trail, column, line, v, echelon_left, echelon_right, diamond"
            raise ValueError(f"Formation inconnue '{formation_type}'. Formations disponibles: {allowed}.")

        return offsets

    def _build_formation_body_offsets(self, swarm_id, members, swarm_cfg, leader_name):
        leader = self._resolve_swarm_leader(members, leader_name)
        followers = [member for member in members if member is not leader]
        formation_cfg = self._normalize_formation_config(swarm_cfg)
        rng_seed = None
        configured_seed = formation_cfg.get("seed", None)

        custom_offsets = self._parse_custom_formation_offsets(
            formation_cfg.get("offsets"), followers
        )
        if custom_offsets is not None:
            selected_type = "custom"
            offsets = custom_offsets
        else:
            random_enabled = bool(formation_cfg.get("random", False))
            mode = str(formation_cfg.get("mode", "fixed")).strip().lower()
            requested_type = str(formation_cfg.get("type", "triangle")).strip().lower()
            random_enabled = random_enabled or mode == "random" or requested_type == "random"

            if random_enabled:
                candidates = formation_cfg.get(
                    "candidates",
                    ["triangle", "trail", "line", "v", "echelon_left", "echelon_right"],
                )
                if not candidates:
                    raise ValueError("formation.candidates ne peut pas être vide en mode random.")

                run_id = int(self.config["simulation"].get("run_config", 0))
                base_seed = formation_cfg.get("seed", None)
                rng_seed = run_id if base_seed is None else int(base_seed) + run_id
                rng = random.Random(rng_seed)
                selected = rng.choice(candidates)
                selected_type = selected.get("type", "triangle") if isinstance(selected, dict) else selected
                if isinstance(selected, dict):
                    selected_cfg = dict(formation_cfg)
                    selected_cfg.update(selected)
                    formation_cfg = selected_cfg
            else:
                selected_type = requested_type

            spacing_x, spacing_y, spacing_z = self._formation_spacing(formation_cfg)
            offsets = self._generate_formation_offsets(
                selected_type, followers, spacing_x, spacing_y, spacing_z
            )

        run_id = int(self.config["simulation"].get("run_config", 0))
        formation_info = {
            "run": run_id,
            "swarm_id": swarm_id,
            "leader": leader.name,
            "formation_type": str(selected_type),
            "offsets": {name: offset.tolist() for name, offset in offsets.items()},
            "control": {
                "mode": str(
                    formation_cfg.get("control_mode", "relational")
                ),
                "offset_semantics": "directed_relative_equilibrium",
                "attraction_gain": float(
                    formation_cfg.get("attraction_gain", 0.8)
                ),
                "velocity_alignment_gain": float(
                    formation_cfg.get("velocity_alignment_gain", 0.35)
                ),
                "relational_lookahead_s": float(
                    formation_cfg.get("relational_lookahead_s", 0.25)
                ),
                "max_relational_correction_speed": float(
                    formation_cfg.get(
                        "max_relational_correction_speed", 1.5
                    )
                ),
                "ready_tolerance": float(
                    formation_cfg.get("ready_tolerance", 0.25)
                ),
            },
        }
        if str(selected_type).strip().lower() == "line":
            parent_by_follower = {}
            previous_name = leader.name
            for follower in followers:
                parent_by_follower[follower.name] = previous_name
                previous_name = follower.name
            formation_info.update(
                {
                    "topology": "directed_chain",
                    "parent_by_follower": parent_by_follower,
                }
            )
        else:
            formation_info.update(
                {
                    "topology": "leader_star",
                    "parent_by_follower": {
                        follower.name: leader.name for follower in followers
                    },
                }
            )
        if configured_seed is not None:
            formation_info["configured_seed"] = int(configured_seed)
        if rng_seed is not None:
            formation_info["effective_seed"] = rng_seed

        self.formation_run_metadata[swarm_id] = formation_info

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"run_{run_id}_formation_{swarm_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(formation_info, handle, indent=2)

        print(
            f"[Swarm] Formation groupe '{swarm_id}': {selected_type} "
            f"-> {formation_info['offsets']} (saved to {path})"
        )
        return offsets

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
            formation_body_offsets = self._build_formation_body_offsets(
                s_id, members, specific_cfg, leader_name
            )
            formation_info = self.formation_run_metadata.get(s_id, {})
            reference_by_follower = formation_info.get("parent_by_follower", {})
            relational_cfg = formation_info.get("control", {})
            
            # Paramètres Réseau
            ip = specific_cfg.get("ip", "localhost")

            # Création de l'instance
            network_in = (
                self.network.port_in
                if self.network is not None
                else int(self.config["simulation"].get("port_in", 5556))
            )
            network_out = (
                self.network.port_out
                if self.network is not None
                else int(self.config["simulation"].get("port_out", 5557))
            )
            communication_seed = int(
                self.effective_seed
                if self.effective_seed is not None
                else self.config["simulation"].get("run_config", 0)
            ) + sum((idx + 1) * ord(char) for idx, char in enumerate(s_id))
            new_swarm = Swarm(
                agents=members,
                leader_name=leader_name,
                formation_body_offsets=formation_body_offsets,
                reference_by_follower=reference_by_follower,
                min_sep=min_sep,
                avoid_gain=avoid_gain,
                formation_control_mode=relational_cfg.get(
                    "mode", "relational"
                ),
                attraction_gain=float(
                    relational_cfg.get("attraction_gain", 0.8)
                ),
                velocity_alignment_gain=float(
                    relational_cfg.get("velocity_alignment_gain", 0.35)
                ),
                relational_lookahead_s=float(
                    relational_cfg.get("relational_lookahead_s", 0.25)
                ),
                max_relational_correction_speed=float(
                    relational_cfg.get(
                        "max_relational_correction_speed", 1.5
                    )
                ),
                formation_ready_tolerance=float(
                    relational_cfg.get("ready_tolerance", 0.25)
                ),
                port_in=network_in,
                port_out=network_out,
                ip=ip,
                deterministic_communication=self.deterministic_execution,
                communication_seed=communication_seed,
            )
            formation_type = formation_info.get("formation_type", "none")
            for member in members:
                desired_offset = (
                    np.zeros(3)
                    if member is new_swarm.leader
                    else new_swarm.reference_offsets.get(member.name, np.zeros(3))
                )
                member.set_formation_metadata(
                    formation_type=formation_type,
                    desired_offset=desired_offset,
                    leader_ref=new_swarm.reference_agents.get(member.name),
                    swarm_id=s_id,
                    control_mode=new_swarm.formation_control_mode,
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
        """Run control, physics and synchronized logging until ``max_sim_time``."""
        if self.counterfactual_forks_enabled:
            self._run_with_counterfactual_forks()
            return

        max_time = float(self.config["simulation"]["max_sim_time"])

        while self.sim_time < max_time and p.isConnected(self.physics_client_id):
            self._advance_simulation_step(
                apply_interventions=True,
                log_main_artifacts=True,
            )

            if self.is_gui_mode:
                time.sleep(self.dt)

        self._finalize_run_artifacts()

    # ------------------------------------------------------------------
    def _write_run_summary(self):
        run_id = int(self.config["simulation"].get("run_config", 0))
        uavs = [agent for agent in self.agents if isinstance(agent, UAV)]
        drone_summaries = [agent.get_run_summary() for agent in uavs]
        crashed_drones = [item for item in drone_summaries if item["crashed"]]
        first_crash = None
        timed_crashes = [
            item for item in crashed_drones if item.get("first_crash_time") is not None
        ]
        if timed_crashes:
            first_crash = min(timed_crashes, key=lambda item: item["first_crash_time"])

        waypoint_summary = {
            agent.name: [np.array(wp, dtype=float).tolist() for wp in agent.waypoints]
            for agent in uavs
        }
        seed_summary = {
            "deterministic_execution": self.deterministic_execution,
            "simulation_seed": self.config.get("simulation", {}).get("seed", None),
            "effective_seed": self.config.get("simulation", {}).get("effective_seed", None),
            "random_waypoints_seed": self.config.get("random_waypoints", {}).get("seed", None),
            "random_waypoints_effective_seed": self.config.get("random_waypoints", {}).get("effective_seed", None),
            "formation_seeds": {
                swarm_id: {
                    "configured_seed": metadata.get("configured_seed", None),
                    "effective_seed": metadata.get("effective_seed", None),
                }
                for swarm_id, metadata in self.formation_run_metadata.items()
            },
        }

        summary = {
            "run": run_id,
            "max_sim_time": self.config.get("simulation", {}).get("max_sim_time", None),
            "formation": self.formation_run_metadata,
            "experiment": self.config.get("experiment", {}),
            "interventions": [event.as_dict() for event in self.intervention_events],
            "seeds": seed_summary,
            "waypoints": waypoint_summary,
            "motion_profiles": {
                agent.name: agent.config.get("motion_profile", {})
                for agent in uavs
                if agent.config.get("motion_profile", {}).get("enabled", False)
            },
            "drones": drone_summaries,
            "crashed_drones": crashed_drones,
            "first_crash": first_crash,
            "relational_ground_truth": {
                "path": (
                    str(self.relational_logger.path)
                    if self.relational_logger is not None
                    else None
                ),
                "rows": (
                    int(self.relational_logger.rows_written)
                    if self.relational_logger is not None
                    else 0
                ),
                "matrix_path": (
                    str(self.relational_logger.matrix_path)
                    if self.relational_logger is not None
                    else None
                ),
                "matrix_convention": "row=receiver,column=sender,self_pairs_excluded",
                "counterfactual_deltas": (
                    "PID command ablation from an identical controller-state snapshot; "
                    "target/RPM channels are separate from physical contact force"
                ),
            },
            "learning_trace": {
                "schema_version": (
                    self.learning_trace_logger.SCHEMA_VERSION
                    if self.learning_trace_logger is not None
                    else None
                ),
                "path": (
                    str(self.learning_trace_logger.path)
                    if self.learning_trace_logger is not None
                    else None
                ),
                "event_catalog_path": (
                    str(self.learning_trace_logger.event_catalog_path)
                    if self.learning_trace_logger is not None
                    else None
                ),
                "rows": (
                    int(self.learning_trace_logger.rows_written)
                    if self.learning_trace_logger is not None
                    else 0
                ),
                "sampling": "control_update",
                "intervention_routing": "force is non-zero only on targeted nodes",
            },
            "counterfactual_forks": {
                "enabled": self.counterfactual_forks_enabled,
                "manifest_path": (
                    self.counterfactual_fork_manifest_path
                    if self.counterfactual_forks_enabled
                    else None
                ),
                "count": len(self.counterfactual_fork_records),
                "rollout_horizon_s": self.counterfactual_fork_horizon,
                "parent_branch": "baseline_without_interventions",
            },
            "system_tracking_error_mean": (
                float(np.mean([item["tracking_error_mean"] for item in drone_summaries]))
                if drone_summaries
                else 0.0
            ),
            "system_tracking_error_max": (
                float(np.max([item["tracking_error_max"] for item in drone_summaries]))
                if drone_summaries
                else 0.0
            ),
            "system_min_nearest_neighbor_dist": (
                float(np.min([item["min_nearest_neighbor_dist"] for item in drone_summaries]))
                if drone_summaries
                else 100.0
            ),
        }

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"run_{run_id}_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[Summary] Run summary saved to {path}")

    def _finalize_run_artifacts(self):
        if self._run_artifacts_finalized:
            return
        for agent in self.agents:
            if agent.type == "uav" and not os.path.exists(agent.log_file):
                agent.write_csv()
        if self.relational_logger is not None:
            self.relational_logger.close()
        if self.learning_trace_logger is not None:
            self.learning_trace_logger.close()
        self._write_run_summary()
        self._run_artifacts_finalized = True

    # ------------------------------------------------------------------
    def stop(self):
        self._finalize_run_artifacts()

        for swarm in self.swarms:
            swarm.cleanup()

        for agent in self.agents:
            cleanup_network = getattr(agent, "cleanup_network", None)
            if callable(cleanup_network):
                cleanup_network()

        if self.network is not None:
            self.network.stop_proxy()

        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(self.physics_client_id)



        
    # ------------------------------------------------------------------
    def it_stop(self, ind: int):
        self._finalize_run_artifacts()

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
        self.formation_run_metadata = {}
        self.relational_logger = None
        self._last_relational_control_indices = None
        self._run_artifacts_finalized = False

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
        self._initialize_relational_logger()


    def reset(self):
        p.resetSimulation(self.physics_client_id)
        self.initialisation()
        
        
        
    
    def disconnect(self):
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)
            print("Déconnection de PyBullet")
        if self.network is not None:
            self.network.stop_proxy()
