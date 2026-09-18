import os
import csv
import threading
import pybullet as p
import numpy as np
import zmq 
import json
import random

# Utility Imports & Control
from utilities.utilities import point_in_cube, point_in_cylinder, discretize_obstacles
from entities.agent import Agent
from entities.sensor import GNSSensor, IMUSensor, LidarSensor
from Control.EKF import EKF
from environment.wind import DrydenGustModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class UAV(Agent):
    PID_COUNTERFACTUAL_STATE_FIELDS = (
        "control_counter",
        "last_rpy",
        "last_pos_e",
        "integral_pos_e",
        "last_rpy_e",
        "integral_rpy_e",
    )
    RELATIONAL_STATE_LOG_COLUMNS = [
        "roll", "pitch", "yaw",
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
        "target_vx", "target_vy", "target_vz",
        "rpm_0", "rpm_1", "rpm_2", "rpm_3",
        "waypoint_x", "waypoint_y", "waypoint_z",
        "effective_safety_radius",
    ]
    REPULSION_COMPONENT_LOG_COLUMNS = [
        "obstacle_rep_force_mag",
        "obstacle_rep_force_x", "obstacle_rep_force_y", "obstacle_rep_force_z",
        "inter_uav_rep_force_mag",
        "inter_uav_rep_force_x", "inter_uav_rep_force_y", "inter_uav_rep_force_z",
        "repulsion_normalization_scale",
    ]
    
    def __init__(self, config: dict, physics_client_id: int, dt: float, known_obstacles_config: dict,planner,world_type: str):
        """
        Initialize a UAV entity with physics simulation, control systems, and autonomous capabilities.
        
        Args:
            config (dict): Configuration dictionary containing:
            - name (str): UAV identifier. Defaults to "UAV".
            - body_id (int): Physics body ID. Defaults to 0000.
            - mass (float): UAV mass in kg. Defaults to 1.5.
            - urdf_path (str): Path to URDF model file. Defaults to "assets/quadrotor.urdf".
            - start_pos (list): Initial position [x, y, z]. Defaults to [0, 0, 1.0].
            - start_orn_euler (list): Initial orientation in Euler angles [roll, pitch, yaw]. Defaults to [0, 0, 0].
            - waypoints (list[list]): List of waypoint coordinates [[x, y, z], ...]. Defaults to [[0, 0, 1]].
            - ctrl_freq (int): Control loop frequency in Hz. Defaults to 80.
            - wind_mean (list): Mean wind vector [x, y, z] in m/s. Defaults to [0, 0, 0].
            - turbulence (float): Dryden gust model turbulence intensity (0-20). Defaults to 15.
            - communication (dict): Network settings with keys:
                - com_period (float): Broadcast interval in seconds. Defaults to 0.1.
                - com_delay_mean (float): Mean communication latency. Defaults to 0.1.
                - com_delay_std (float): Std dev of communication latency. Defaults to 0.02.
            - sensors (dict): Sensor configurations with keys:
                - gnss (dict): GNSS sensor settings (frequency, delay_mean, delay_std).
                - imu (dict): IMU sensor settings (noise parameters).
                - lidar (dict): Lidar sensor settings (frequency, range, noise).
            - physics (dict): Physics parameters:
                - thrust_coeff (float): Thrust coefficient KF. Defaults to 6.11e-8.
                - torque_coeff (float): Torque coefficient KM. Defaults to 1.5e-9.
                - max_rpm (float): Maximum rotor RPM. Defaults to 22000.
                - max_speed (float): Maximum velocity in m/s. Defaults to 5.
            physics_client_id (int): PyBullet physics client identifier.
            dt (float): Physics simulation timestep in seconds (typically 1/240).
            known_obstacles_config (list[dict]): Configuration list for static obstacles with keys:
            - center (list): [x, y, z] center position.
            - height, width, length (float): Obstacle dimensions.
            planner: A* path planner instance with planning interface.
        
        Initializes:
            - Physics engine: Body properties, dynamics, external force/torque application.
            - Control system: DSL PID controller running at configurable frequency (default 80Hz).
            - Navigation: Waypoint management and A* path planning with async thread support.
            - Sensor fusion: EKF (Extended Kalman Filter) with GNSS, IMU, and Lidar.
            - Obstacle avoidance: Repulsive force computation and collision detection.
            - Swarm coordination: Leader-follower formation control and neighbor tracking.
            - Communication: ZMQ-based state broadcasting and swarm message handling.
            - Wind simulation: Dryden Gust Model for realistic turbulence.
            - Logging: CSV trajectory tracking (simulation time, position).
        
        Note:
            Physics loop runs at 240Hz; logic/control loop runs at configurable frequency (80Hz default).
        """
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "UAV")
        self.bodyId = config.get("body_id", 0000)
        self.type = "uav"
        self.mass = config.get("mass", 1.5)  # kg
        
        urdf_path = config.get("urdf_path", "assets/quadrotor.urdf")
        self.start_pos = config.get("start_pos", [0, 0, 1.0])
        self.start_orn = p.getQuaternionFromEuler(config.get("start_orn_euler", [0,0,0]))
        super().__init__(urdf_path, self.start_pos, self.start_orn, physics_client_id, self.dt)
        self._sim_time = 0.0
        
        # --- INTEGER-SYNCHRONIZED CONTROL SCHEDULER ---
        self.CTRL_FREQ = float(self.config.get("ctrl_freq", 80))
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.PHYSICS_STEPS_PER_CONTROL = self._control_stride(
            self.dt,
            self.CTRL_FREQ,
        )
        self.physics_step_index = 0
        self.last_ctrl_time = -self.CTRL_DT  # First physics step performs control.

        # --- PHYSICS ---
        self.KF = self.config.get("physics", {}).get("thrust_coeff", 6.11e-8)
        self.KM = self.config.get("physics", {}).get("torque_coeff", 1.5e-9)
        self.G = 9.81
        self.MAX_RPM = config.get("physics", {}).get("max_rpm", 22000.0)
        self.max_speed = config.get("physics", {}).get("max_speed", 5)
        self.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])
        self.world_bounds = config.get("world_bounds", {})
        self.world_bounds_margin = float(config.get("world_bounds_margin", 0.0))
        self.min_target_altitude = float(config.get("min_target_altitude", 0.8))

        nav_cfg = config.get("navigation", {})
        self.waypoint_reached_radius = float(nav_cfg.get("waypoint_reached_radius", 0.5))
        self.path_point_reached_radius = float(nav_cfg.get("path_point_reached_radius", 0.7))
        self.approach_slowdown_radius = float(nav_cfg.get("approach_slowdown_radius", 5.0))
        self.approach_min_speed_ratio = float(nav_cfg.get("approach_min_speed_ratio", 0.25))
        self.approach_brake_gain = float(nav_cfg.get("approach_brake_gain", 1.2))
        self.guidance_lookahead = float(nav_cfg.get("guidance_lookahead", 1.0))
        self.guidance_min_lookahead = float(nav_cfg.get("guidance_min_lookahead", 0.25))
        self.turn_smoothing_angle = np.deg2rad(float(nav_cfg.get("turn_smoothing_angle_deg", 70.0)))
        self.max_guidance_turn_rate = np.deg2rad(float(nav_cfg.get("max_guidance_turn_rate_deg_s", 180.0)))
        self.synchronous_planning = bool(nav_cfg.get("synchronous_planning", False))
        self.last_guidance_dir_xy = None

        crash_cfg = config.get("crash", {})
        self.crashed = False
        self.crash_reason = None
        self.pending_crash_reason = None
        self.zero_velocity_on_unphysical_crash = bool(crash_cfg.get("zero_velocity_on_unphysical_crash", True))
        self.crash_contact_steps = 0
        self.crash_contact_threshold = max(
            1,
            int(round(float(crash_cfg.get("contact_time_threshold", 0.08)) / self.CTRL_DT)),
        )
        self.ground_impact_speed_threshold = float(crash_cfg.get("ground_impact_speed_threshold", 1.5))
        self.crash_tilt_threshold = float(crash_cfg.get("tilt_threshold_rad", 0.8))
        self.max_safe_speed = float(crash_cfg.get("max_safe_speed", 20.0))
        self.bounds_violation_margin = float(crash_cfg.get("bounds_violation_margin", 2.0))
        self.low_altitude_crash_z = float(crash_cfg.get("low_altitude_crash_z", 0.2))
        self.low_altitude_target_z = float(crash_cfg.get("low_altitude_target_z", 0.5))
        self.ground_penetration_z = float(crash_cfg.get("ground_penetration_z", -0.05))
        self.takeoff_complete_z = float(crash_cfg.get("takeoff_complete_z", 0.8))
        self.has_taken_off = self.start_pos[2] >= self.takeoff_complete_z

        formation_cfg = config.get("formation_control", {})
        self.formation_safe_altitude = float(
            formation_cfg.get(
                "safe_altitude",
                crash_cfg.get(
                    "formation_safe_altitude",
                    max(self.min_target_altitude, self.takeoff_complete_z + 0.7),
                ),
            )
        )
        self.formation_takeoff_altitude_tolerance = max(
            0.0,
            min(
                self.formation_safe_altitude,
                float(
                    formation_cfg.get(
                        "takeoff_altitude_tolerance",
                        crash_cfg.get("formation_takeoff_altitude_tolerance", 0.05),
                    )
                ),
            ),
        )
        self.formation_takeoff_complete = self.start_pos[2] >= self.formation_safe_altitude
        self.formation_takeoff_timer = 0.0
        self.formation_takeoff_unstable_timer = 0.0
        self.formation_takeoff_min_duration = max(
            0.0,
            float(
                formation_cfg.get(
                    "takeoff_min_duration",
                    crash_cfg.get("formation_takeoff_min_duration", 1.0),
                )
            ),
        )
        self.formation_takeoff_max_abs_vz = max(
            0.0,
            float(
                formation_cfg.get(
                    "takeoff_max_abs_vz",
                    crash_cfg.get("formation_takeoff_max_abs_vz", 0.6),
                )
            ),
        )
        self.formation_takeoff_unstable_grace_duration = max(
            0.0,
            float(
                formation_cfg.get(
                    "takeoff_unstable_grace_duration",
                    crash_cfg.get("formation_takeoff_unstable_grace_duration", 0.2),
                )
            ),
        )
        self.formation_transition_duration = max(
            0.0,
            float(
                formation_cfg.get(
                    "transition_duration",
                    crash_cfg.get("formation_transition_duration", 3.0),
                )
            ),
        )
        self.formation_transition_timer = 0.0
        self.formation_transition_start_pos = None
        self.formation_hold_leader_until_ready = bool(
            formation_cfg.get("hold_leader_until_ready", True)
        )
        self.formation_hold_active = False
        self.takeoff_xy_speed_limit = float(crash_cfg.get("takeoff_xy_speed_limit", 0.35))
        self.formation_transition_speed_limit = float(
            formation_cfg.get(
                "transition_speed_limit",
                crash_cfg.get("formation_transition_speed_limit", 1.5),
            )
        )
        self.low_altitude_xy_speed_limit = float(crash_cfg.get("low_altitude_xy_speed_limit", 1.5))
        
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.last_rpms = np.zeros(4)
        
        # --- NAVIGATION ---
        wp_list = config.get("waypoints", [])
        if not wp_list: wp_list = [[0,0,1]]
        self.waypoints = [np.array(w) for w in wp_list]
        self.wp_idx = 0

        motion_cfg = config.get("motion_profile", {})
        self.motion_profile_enabled = bool(motion_cfg.get("enabled", False))
        self.motion_profile_type = str(motion_cfg.get("type", "none")).strip().lower()
        self.motion_profile_center = np.asarray(
            motion_cfg.get("center", self.start_pos),
            dtype=float,
        )
        self.motion_profile_amplitude = np.asarray(
            motion_cfg.get("amplitude", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.motion_profile_period = np.asarray(
            motion_cfg.get("period", [40.0, 25.0, 60.0]),
            dtype=float,
        )
        self.motion_profile_phase = np.asarray(
            motion_cfg.get("phase", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.motion_profile_warmup = max(0.0, float(motion_cfg.get("warmup_s", 0.0)))
        for name, values in (
            ("center", self.motion_profile_center),
            ("amplitude", self.motion_profile_amplitude),
            ("period", self.motion_profile_period),
            ("phase", self.motion_profile_phase),
        ):
            if values.shape != (3,):
                raise ValueError(f"motion_profile.{name} doit contenir exactement trois valeurs.")
        if self.motion_profile_enabled:
            if self.motion_profile_type != "sinusoidal":
                raise ValueError("Seul motion_profile.type='sinusoidal' est actuellement supporté.")
            if np.any(self.motion_profile_period <= 0.0):
                raise ValueError("motion_profile.period doit contenir des périodes strictement positives.")
        
        # --- OBSTACLES & PLANNING ---
        self.obs_dic = known_obstacles_config # Combined list for avoidance
        print(len(self.obs_dic), "known obstacle points loaded.")
        self.environment = world_type
        
        self.planner = planner
        
        self.target_yaw_cache = 0.0
        
        self.max_repulsive_force = self.config.get("physics",{}).get("max_repulsive_force",2.0)
        self.safety_radius = self.config.get("physics",{}).get("safety_radius",1.0)

        # Planning States
        self.active_path = []
        self.is_planning = False      
        self.planning_thread = None   
        self.replan_timer = 0         
        self.Calculation_fail_count = 0

        self.deterministic_communication = bool(
            config.get("deterministic_communication", False)
        )
        deterministic_seed = config.get("deterministic_seed", None)
        if deterministic_seed is None:
            self.random_rng = random
            sensor_rngs = (None, None, None)
        else:
            deterministic_seed = int(deterministic_seed)
            self.random_rng = random.Random(deterministic_seed + 101)
            child_seeds = np.random.SeedSequence(deterministic_seed).spawn(3)
            sensor_rngs = tuple(np.random.default_rng(seed) for seed in child_seeds)
        self.zmq_ctx = None if self.deterministic_communication else zmq.Context()
        self.sub_socket = None
        self.pub_socket = None
        self.radar_sub_socket = None 

        # --- SWARM CONTROL ---
        self.swarm_active = False
        self.swarm_target_pos = None
        self.swarm_target_vel = None
        self.swarm_target_yaw = None 
        self.leader = False
        self.swarm_id = self.config.get("swarm_id", "")
        self.swarm_leader_ref = None
        self.formation_type = "none"
        self.desired_formation_offset = np.zeros(3)
        self.other_agent_pos = {}
        self.neighbors_data = {} 
        self.swarm_name=[]
        # --- COMMUNICATION ---
        com=self.config.get("communication")
        self.com_period = com.get("com_period",0.1)
        self.last_com_time = -self.com_period
        self.message_buffer = []
        self.last_message_receive_time_by_sender = {}
        self.last_message_source_time_by_sender = {}
        self.perception_delay_mean = com.get("com_delay_mean",0.1)  # 100ms de retard
        self.perception_delay_std = com.get("com_delay_std",0.02)  # +/- 20ms
        # --- SENSORS ---
        sens = self.config.get("sensors", {})
        self.ekf = EKF(self.CTRL_DT); self.ekf.x[:3] = self.start_pos
        self.gnss = GNSSensor(sens.get("gnss", {}), rng=sensor_rngs[0])
        self.imu = IMUSensor(sens.get("imu", {}), rng=sensor_rngs[1])
        self.lidar = LidarSensor(sens.get("lidar", {}))
        
        lidar_freq = sens.get("lidar", {}).get("frequency", 10.0)
        self.lidar_period = 1.0 / lidar_freq
        self.last_lidar_time = -self.lidar_period
        
        self.gnss_FREQ = sens.get("gnss", {}).get("frequency", 10.0)    # 10 Hz (Realistic Standard)
        self.gnss_DT = 1.0 / self.gnss_FREQ
        self.last_gnss_update_time = -self.gnss_DT
        self.gnss_delay_mean= sens.get("gnss", {}).get("delay_mean", 0.1)
        self.gnss_delay_std= sens.get("gnss", {}).get("delay_std", 0.01)
        self.next_gnss_trigger = 0.0
        #wind 
        self.current_wind = np.zeros(3)
        self.mean_wind = self.config.get("wind_mean",[0,0,0])
        self.turbulence = self.config.get("turbulence",15)
        self.wind_module = DrydenGustModel(
            self.dt, self.turbulence, self.mean_wind, rng=sensor_rngs[2]
        )
        
        # --- CAUSAL ANALYSIS & LOGGING SETUP ---
        self.logging_enabled = True
        self.runId = self.config.get("run_config", 0)
        self.log_dir = str(self.config.get("log_dir", "logs"))
        self.log_file = os.path.join(self.log_dir, f"run_{self.runId}_{self.name}.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        if os.path.exists(self.log_file): os.remove(self.log_file)

        # Test pour optimiser l'écriture du CSV
        self.log_data_buffer = []
        # Header complet pour l'analyse causale
        self.log_header = [
            "time", 
            "gt_x", "gt_y", "gt_z",         # Ground Truth
            "gt_vx", "gt_vy", "gt_vz",      
            "meas_x", "meas_y", "meas_z",   # Sensors
            "gnss_error_mag",                
            "wind_x", "wind_y", "wind_z",   # Environment
            "wind_mag",
            "rep_force_mag",                # Interaction
            "rep_force_x", "rep_force_y", "rep_force_z",
            "nearest_obstacle_dist",
            "nearest_obstacle_dir_x", "nearest_obstacle_dir_y", "nearest_obstacle_dir_z",
            "nearest_neighbor_dist",
            "target_x", "target_y", "target_z", # Intent                
            "tracking_error_mag",
            "collision_flag",               # Flags
            "collision_with_obstacle_flag",
            "run_id", "drone_name", "swarm_id", "is_leader",
            "waypoint_idx", "distance_to_waypoint",
            "formation_type",
            "desired_offset_x", "desired_offset_y", "desired_offset_z",
            "formation_error_mag",
            "crash_flag", "crash_reason", "pending_crash_reason",
            "ground_contact_flag", "non_ground_contact_flag",
            "rpm_mean", "rpm_max", "motor_saturation_ratio",
            "intervention_active", "intervention_type",
            "external_force_x", "external_force_y", "external_force_z",
        ]
        # Existing columns above are intentionally untouched and in their
        # historical order. Relational state fields are append-only.
        self.log_header.extend(self.RELATIONAL_STATE_LOG_COLUMNS)
        self.log_header.extend(self.REPULSION_COMPONENT_LOG_COLUMNS)

        # Variables internes pour le logging
        self.last_repulsive_force_mag = 0.0
        self.last_repulsive_force = np.zeros(3)
        self.nearest_obstacle_dist = 100.0
        self.nearest_obstacle_dir = np.zeros(3)
        self.dist_to_nearest_neighbor = 100.0
        self.current_target_pos = self.start_pos
        self.current_target_vel = np.zeros(3)
        self.control_update_index = 0
        self.last_inter_uav_repulsion_raw = {}
        self.last_inter_uav_repulsion_applied = {}
        self.last_inter_uav_repulsion_source_age_s = {}
        self.last_building_repulsive_force_raw = np.zeros(3)
        self.last_building_repulsive_force_applied = np.zeros(3)
        self.last_repulsion_normalization_scale = 1.0
        self.last_central_avoidance_raw_by_sender = {}
        self.last_central_avoidance_applied_by_sender = {}
        self.last_formation_relative_error_by_sender = {}
        self.last_formation_relative_velocity_error_by_sender = {}
        self.last_formation_attraction_raw_by_sender = {}
        self.last_formation_attraction_applied_by_sender = {}
        self.last_formation_damping_raw_by_sender = {}
        self.last_formation_damping_applied_by_sender = {}
        self.last_formation_transport_velocity_by_sender = {}
        self.last_formation_correction_scale = 1.0
        self.last_formation_application_scale = 0.0
        self.formation_control_mode = "none"
        self.last_leader_reference_active = False
        self.last_leader_message_received = False
        self.last_leader_message_age_s = np.nan
        self.last_counterfactual_control_by_sender = {}
        self.last_counterfactual_control_valid = False
        self.last_counterfactual_control_time = np.nan
        self.intervention_active = 0
        self.intervention_type = "none"
        self.external_force = np.zeros(3)
        self.first_crash_time = None
        self.tracking_error_sum = 0.0
        self.tracking_error_count = 0
        self.tracking_error_max = 0.0
        self.min_nearest_neighbor_dist_seen = 100.0

        self.broadcast_state(pos=self.start_pos, vel=[0, 0, 0])
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    # ----------------------------------------------------------------------
    # SWARM API
    # ----------------------------------------------------------------------
    def set_swarm_activate(self):
        self.swarm_active = True
        self.future_state = {"pos": np.array(self.start_pos), "vel": np.zeros(3), "yaw": 0.0}

    def set_formation_metadata(
        self,
        formation_type="none",
        desired_offset=None,
        leader_ref=None,
        swarm_id=None,
        control_mode="none",
    ):
        self.formation_type = str(formation_type or "none")
        self.formation_control_mode = str(control_mode or "none")
        self.desired_formation_offset = (
            np.zeros(3) if desired_offset is None else np.array(desired_offset, dtype=float)
        )
        if self.desired_formation_offset.shape != (3,):
            self.desired_formation_offset = np.zeros(3)
        self.swarm_leader_ref = leader_ref
        if swarm_id is not None:
            self.swarm_id = str(swarm_id)

    def set_applied_central_avoidance_metadata(self, future_state, application_scale):
        """Record central terms from the exact FUTURE_POS command in use."""
        self.last_central_avoidance_raw_by_sender = {}
        self.last_central_avoidance_applied_by_sender = {}
        self.last_formation_relative_error_by_sender = {}
        self.last_formation_relative_velocity_error_by_sender = {}
        self.last_formation_attraction_raw_by_sender = {}
        self.last_formation_attraction_applied_by_sender = {}
        self.last_formation_damping_raw_by_sender = {}
        self.last_formation_damping_applied_by_sender = {}
        self.last_formation_transport_velocity_by_sender = {}
        self.last_formation_correction_scale = 1.0
        self.last_formation_application_scale = 0.0
        self.last_leader_reference_active = False
        self.last_leader_message_received = False
        self.last_leader_message_age_s = np.nan
        if not isinstance(future_state, dict) or application_scale <= 0.0:
            return

        scale = float(application_scale)
        self.last_formation_application_scale = scale
        for sender, value in future_state.get("central_avoidance_raw_by_sender", {}).items():
            vector = np.asarray(value, dtype=float)
            if vector.shape == (3,):
                self.last_central_avoidance_raw_by_sender[str(sender)] = scale * vector
        for sender, value in future_state.get("central_avoidance_applied_by_sender", {}).items():
            vector = np.asarray(value, dtype=float)
            if vector.shape == (3,):
                self.last_central_avoidance_applied_by_sender[str(sender)] = scale * vector

        def vectors(name):
            result = {}
            values = future_state.get(name, {})
            if not isinstance(values, dict):
                return result
            for sender, value in values.items():
                vector = np.asarray(value, dtype=float)
                if vector.shape == (3,):
                    result[str(sender)] = vector.copy()
            return result

        self.last_formation_relative_error_by_sender = vectors(
            "relative_position_error_by_sender"
        )
        self.last_formation_relative_velocity_error_by_sender = vectors(
            "relative_velocity_error_by_sender"
        )
        self.last_formation_attraction_raw_by_sender = vectors(
            "formation_attraction_raw_by_sender"
        )
        self.last_formation_attraction_applied_by_sender = {
            sender: scale * value
            for sender, value in vectors(
                "formation_attraction_applied_by_sender"
            ).items()
        }
        self.last_formation_damping_raw_by_sender = vectors(
            "formation_damping_raw_by_sender"
        )
        self.last_formation_damping_applied_by_sender = {
            sender: scale * value
            for sender, value in vectors(
                "formation_damping_applied_by_sender"
            ).items()
        }
        self.last_formation_transport_velocity_by_sender = {
            sender: scale * value
            for sender, value in vectors(
                "formation_transport_velocity_by_sender"
            ).items()
        }
        try:
            self.last_formation_correction_scale = float(
                future_state.get("formation_correction_scale", 1.0)
            )
        except (TypeError, ValueError):
            self.last_formation_correction_scale = 1.0

        leader_sender = future_state.get("leader_reference_sender")
        expected_leader = getattr(self.swarm_leader_ref, "name", None)
        self.last_leader_reference_active = bool(
            leader_sender is not None and leader_sender == expected_leader
        )
        self.last_leader_message_received = bool(
            future_state.get("leader_message_received_flag", 0)
        )
        source_time = future_state.get("leader_source_time", None)
        try:
            if source_time is not None:
                self.last_leader_message_age_s = max(
                    0.0, float(self._sim_time) - float(source_time)
                )
        except (TypeError, ValueError):
            self.last_leader_message_age_s = np.nan

    def set_intervention_state(self, active=False, intervention_type="none", external_force=None):
        self.intervention_active = int(bool(active))
        self.intervention_type = str(intervention_type or "none")
        self.external_force = (
            np.zeros(3) if external_force is None else np.array(external_force, dtype=float)
        )
        if self.external_force.shape != (3,):
            self.external_force = np.zeros(3)

    def _snapshot_pid_state(self):
        """Copy every mutable DSLPID state needed for a side-effect-free replay."""
        snapshot = {}
        for field in self.PID_COUNTERFACTUAL_STATE_FIELDS:
            value = getattr(self.ctrl, field, None)
            snapshot[field] = value.copy() if isinstance(value, np.ndarray) else value
        return snapshot

    def _restore_pid_state(self, snapshot):
        for field, value in snapshot.items():
            setattr(
                self.ctrl,
                field,
                value.copy() if isinstance(value, np.ndarray) else value,
            )

    def _finalize_pid_targets(
        self,
        pos,
        vel,
        target_pos,
        target_vel,
        goal_pos_for_approach,
        repulsive_force,
        holding_formation_takeoff,
        in_formation_transition,
        guidance_seed=None,
        preserve_guidance=False,
    ):
        """Apply the same post-processing to real and counterfactual targets."""
        saved_guidance = (
            None
            if self.last_guidance_dir_xy is None
            else np.asarray(self.last_guidance_dir_xy, dtype=float).copy()
        )
        if guidance_seed is not None:
            self.last_guidance_dir_xy = np.asarray(guidance_seed, dtype=float).copy()
        elif preserve_guidance:
            self.last_guidance_dir_xy = None

        final_target_vel, local_max_speed = self.apply_waypoint_approach(
            pos, vel, target_vel, goal_pos_for_approach
        )
        if holding_formation_takeoff:
            local_max_speed = min(local_max_speed, self.takeoff_xy_speed_limit)
        elif in_formation_transition:
            local_max_speed = min(local_max_speed, self.formation_transition_speed_limit)
        elif pos[2] < self.formation_safe_altitude:
            local_max_speed = min(local_max_speed, self.low_altitude_xy_speed_limit)

        final_target_vel = final_target_vel + (
            np.asarray(repulsive_force, dtype=float) / self.mass * 3 * self.CTRL_DT
        )
        speed_xy = np.linalg.norm(final_target_vel[:2])
        if speed_xy > local_max_speed:
            final_target_vel[:2] *= local_max_speed / speed_xy

        final_target_pos = np.asarray(target_pos, dtype=float) + (
            final_target_vel * self.CTRL_DT
        )
        final_target_pos = self.clamp_to_world_bounds(final_target_pos)
        final_target_pos = self.limit_guidance_turn(pos, final_target_pos)
        vector_to_target = final_target_pos - pos
        dist_to_target = np.linalg.norm(vector_to_target)
        if dist_to_target > 3:
            virtual_target_pos = (
                pos + (vector_to_target / dist_to_target) * self.guidance_lookahead
            )
        else:
            virtual_target_pos = final_target_pos

        if preserve_guidance:
            self.last_guidance_dir_xy = saved_guidance
        return virtual_target_pos, final_target_vel, final_target_pos

    def _compute_pair_counterfactuals(
        self,
        state_vec,
        pos,
        vel,
        target_pos,
        target_vel,
        goal_pos_for_approach,
        repulsive_force,
        holding_formation_takeoff,
        in_formation_transition,
        guidance_before,
        actual_virtual_target_pos,
        actual_target_vel,
        actual_rpms,
        pid_state_before,
        pid_state_after,
    ):
        """Ablate each sender's active contribution and replay the PID once."""
        position_contributions = {
            str(name): np.asarray(value, dtype=float).copy()
            for name, value in self.last_central_avoidance_applied_by_sender.items()
            if np.linalg.norm(np.asarray(value, dtype=float)) > 1e-12
        }
        velocity_contributions = {}

        if self.last_leader_reference_active and self.swarm_leader_ref is not None:
            leader_name = str(self.swarm_leader_ref.name)
            central_total = sum(position_contributions.values(), np.zeros(3))
            leader_position = np.asarray(target_pos, dtype=float) - pos - central_total
            position_contributions[leader_name] = (
                position_contributions.get(leader_name, np.zeros(3)) + leader_position
            )
            velocity_contributions[leader_name] = np.asarray(target_vel, dtype=float).copy()

        local_force_contributions = {
            str(name): np.asarray(value, dtype=float).copy()
            for name, value in self.last_inter_uav_repulsion_applied.items()
            if np.linalg.norm(np.asarray(value, dtype=float)) > 1e-12
        }
        senders = sorted(
            set(position_contributions)
            | set(velocity_contributions)
            | set(local_force_contributions)
        )
        results = {}
        try:
            for sender in senders:
                counterfactual_pos = (
                    np.asarray(target_pos, dtype=float)
                    - position_contributions.get(sender, np.zeros(3))
                )
                counterfactual_vel = (
                    np.asarray(target_vel, dtype=float)
                    - velocity_contributions.get(sender, np.zeros(3))
                )
                counterfactual_force = (
                    np.asarray(repulsive_force, dtype=float)
                    - local_force_contributions.get(sender, np.zeros(3))
                )
                cf_virtual_pos, cf_target_vel, _ = self._finalize_pid_targets(
                    pos=pos,
                    vel=vel,
                    target_pos=counterfactual_pos,
                    target_vel=counterfactual_vel,
                    goal_pos_for_approach=goal_pos_for_approach,
                    repulsive_force=counterfactual_force,
                    holding_formation_takeoff=holding_formation_takeoff,
                    in_formation_transition=in_formation_transition,
                    guidance_seed=guidance_before,
                    preserve_guidance=True,
                )
                self._restore_pid_state(pid_state_before)
                cf_rpms, _, _ = self.ctrl.computeControlFromState(
                    control_timestep=self.CTRL_DT,
                    state=state_vec,
                    target_pos=cf_virtual_pos,
                    target_vel=cf_target_vel,
                    target_rpy=np.array([0, 0, self.target_yaw_cache]),
                )
                results[sender] = {
                    "delta_target": np.asarray(actual_virtual_target_pos) - cf_virtual_pos,
                    "delta_target_vel": np.asarray(actual_target_vel) - cf_target_vel,
                    "delta_rpm": np.asarray(actual_rpms) - np.asarray(cf_rpms),
                    "status": "valid_pid_command_ablation",
                }
        finally:
            self._restore_pid_state(pid_state_after)
        self.last_counterfactual_control_by_sender = results
        self.last_counterfactual_control_valid = True
        self.last_counterfactual_control_time = float(self._sim_time)

    def apply_logged_external_force(self, force, intervention_type="external_force", link_id=-1, frame=p.WORLD_FRAME):
        force = np.array(force, dtype=float)
        if force.shape != (3,):
            raise ValueError("force doit être un vecteur 3D [fx, fy, fz].")
        application_point = [0.0, 0.0, 0.0]
        if frame == p.WORLD_FRAME:
            # In WORLD_FRAME, posObj is an absolute world position. Applying a
            # push at [0, 0, 0] while the UAV is tens of metres away creates a
            # huge artificial lever arm. Use the selected link's centre so the
            # intervention is a pure translational force, not an unintended torque.
            if link_id == -1:
                application_point = list(
                    p.getBasePositionAndOrientation(
                        self.bodyId, physicsClientId=self.physics_client_id
                    )[0]
                )
            else:
                application_point = list(
                    p.getLinkState(
                        self.bodyId,
                        link_id,
                        physicsClientId=self.physics_client_id,
                    )[0]
                )
        p.applyExternalForce(
            self.bodyId,
            link_id,
            forceObj=force.tolist(),
            posObj=application_point,
            flags=frame,
            physicsClientId=self.physics_client_id,
        )
        self.set_intervention_state(
            active=True,
            intervention_type=intervention_type,
            external_force=force,
        )

    def clear_intervention_state(self):
        self.set_intervention_state(active=False, intervention_type="none", external_force=np.zeros(3))

    # ----------------------------------------------------------------------
    # Obstacle management and path planning
    # ----------------------------------------------------------------------
    def trigger_planning(self, start_pos, target_pos):
        if not self.is_planning:
            self.is_planning = True
            start_pos = np.array(start_pos, dtype=float).copy()
            target_pos = np.array(target_pos, dtype=float).copy()
            if self.synchronous_planning:
                self.run_async_plan(start_pos, target_pos)
                return
            print(f"[{self.name}] ⏳ Starting A* Thread...")
            self.planning_thread = threading.Thread(
            target=self.run_async_plan, 
            args=(start_pos, target_pos)
            )
            self.planning_thread.daemon = True 
            self.planning_thread.start()

    def run_async_plan(self, start_pos, target_pos):
        try:
            path = self.planner.plan(start_pos, target_pos)
            if path and len(path) > 0:
                self.active_path = [np.array(point, dtype=float).copy() for point in path]
                self.Calculation_fail_count = 0
            else:
                self.Calculation_fail_count += 1
        except Exception as e:
            print(f"[{self.name}] 💥 Error in A* thread: {e}")
        finally:
            self.is_planning = False 


    def clamp_to_world_bounds(self, point):
        point = np.array(point, dtype=float).copy()
        if not self.world_bounds:
            return point

        if "x" in self.world_bounds:
            lo, hi = self.world_bounds["x"]
            point[0] = np.clip(point[0], float(lo) + self.world_bounds_margin, float(hi) - self.world_bounds_margin)
        if "y" in self.world_bounds:
            lo, hi = self.world_bounds["y"]
            point[1] = np.clip(point[1], float(lo) + self.world_bounds_margin, float(hi) - self.world_bounds_margin)
        if "z" in self.world_bounds:
            lo, hi = self.world_bounds["z"]
            min_z = float(lo)
            if self._sim_time > 1.0:
                min_z = max(min_z, self.min_target_altitude)
            point[2] = np.clip(point[2], min_z, float(hi))

        return point

    def sinusoidal_motion_target(self):
        """Return the deterministic sinusoidal target position and velocity."""

        if self._sim_time < self.motion_profile_warmup:
            start = np.asarray(self.start_pos, dtype=float)
            orbit_start = (
                self.motion_profile_center
                + self.motion_profile_amplitude * np.sin(self.motion_profile_phase)
            )
            progress = np.clip(self._sim_time / max(self.motion_profile_warmup, 1e-9), 0.0, 1.0)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            smooth_rate = (
                (6.0 * progress - 6.0 * progress * progress)
                / max(self.motion_profile_warmup, 1e-9)
            )
            target_pos = start + smooth_progress * (orbit_start - start)
            target_vel = smooth_rate * (orbit_start - start)
            return self.clamp_to_world_bounds(target_pos), target_vel

        profile_time = self._sim_time - self.motion_profile_warmup
        angular_frequency = 2.0 * np.pi / self.motion_profile_period
        angle = angular_frequency * profile_time + self.motion_profile_phase
        target_pos = self.motion_profile_center + self.motion_profile_amplitude * np.sin(angle)
        target_vel = self.motion_profile_amplitude * angular_frequency * np.cos(angle)
        return self.clamp_to_world_bounds(target_pos), target_vel

    def limit_guidance_turn(self, current_pos, desired_target_pos):
        desired_target_pos = np.array(desired_target_pos, dtype=float).copy()
        current_pos = np.array(current_pos, dtype=float)
        vec_xy = desired_target_pos[:2] - current_pos[:2]
        dist_xy = float(np.linalg.norm(vec_xy))

        if dist_xy < 1e-6:
            self.last_guidance_dir_xy = None
            return desired_target_pos

        desired_dir = vec_xy / dist_xy
        if self.last_guidance_dir_xy is None:
            self.last_guidance_dir_xy = desired_dir
            return desired_target_pos

        prev_dir = self.last_guidance_dir_xy
        cross = prev_dir[0] * desired_dir[1] - prev_dir[1] * desired_dir[0]
        dot = float(np.clip(np.dot(prev_dir, desired_dir), -1.0, 1.0))
        angle = float(np.arctan2(cross, dot))

        max_turn = self.max_guidance_turn_rate * self.CTRL_DT
        if abs(angle) > self.turn_smoothing_angle:
            limited_angle = float(np.clip(angle, -max_turn, max_turn))
            c, s = np.cos(limited_angle), np.sin(limited_angle)
            new_dir = np.array([c * prev_dir[0] - s * prev_dir[1], s * prev_dir[0] + c * prev_dir[1]])
            desired_target_pos[:2] = current_pos[:2] + new_dir * dist_xy
            self.last_guidance_dir_xy = new_dir
        else:
            self.last_guidance_dir_xy = desired_dir

        return desired_target_pos

    def apply_waypoint_approach(self, current_pos, current_vel, target_vel, goal_pos):
        target_vel = np.array(target_vel, dtype=float).copy()
        if goal_pos is None or self.approach_slowdown_radius <= 0:
            return target_vel, self.max_speed

        current_pos = np.array(current_pos, dtype=float)
        current_vel = np.array(current_vel, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        dist_to_goal = float(np.linalg.norm(goal_pos - current_pos))

        if dist_to_goal >= self.approach_slowdown_radius:
            return target_vel, self.max_speed

        speed_ratio = np.clip(
            dist_to_goal / self.approach_slowdown_radius,
            self.approach_min_speed_ratio,
            1.0,
        )
        brake_ratio = 1.0 - speed_ratio
        target_vel -= current_vel * self.approach_brake_gain * brake_ratio
        return target_vel, self.max_speed * speed_ratio
    
    # Dans causal_inference_sim/entities/uav.py

    def _building_surface_features(self, current_pos, obs):
        center = np.array(obs.get("center", [0.0, 0.0, 0.0]), dtype=float)
        height = float(obs.get("height", center[2] if center.size >= 3 else 0.0))
        width = float(obs.get("width", 0.0))
        length = float(obs.get("length", 0.0))

        min_x = center[0] - length / 2.0
        max_x = center[0] + length / 2.0
        min_y = center[1] - width / 2.0
        max_y = center[1] + width / 2.0

        closest = np.array(
            [
                np.clip(current_pos[0], min_x, max_x),
                np.clip(current_pos[1], min_y, max_y),
                np.clip(current_pos[2], 0.0, height),
            ],
            dtype=float,
        )
        vec = np.array(current_pos, dtype=float) - closest
        dist = float(np.linalg.norm(vec))
        if dist > 1e-9:
            return dist, vec / dist

        # Inside the obstacle volume: distance is zero, direction points to the
        # closest exit face so the logged direction is still meaningful.
        face_distances = [
            (abs(current_pos[0] - min_x), np.array([-1.0, 0.0, 0.0])),
            (abs(max_x - current_pos[0]), np.array([1.0, 0.0, 0.0])),
            (abs(current_pos[1] - min_y), np.array([0.0, -1.0, 0.0])),
            (abs(max_y - current_pos[1]), np.array([0.0, 1.0, 0.0])),
            (abs(height - current_pos[2]), np.array([0.0, 0.0, 1.0])),
        ]
        _, direction = min(face_distances, key=lambda item: item[0])
        return 0.0, direction

    def _nearest_generated_obstacle_features(self, current_pos):
        if not self.obs_dic:
            return 100.0, np.zeros(3)

        candidate_indices = range(len(self.obs_dic))
        if getattr(self.planner, "building_tree", None) is not None:
            nearby = self.planner.building_tree.query_ball_point(current_pos[:2], r=30.0)
            if nearby:
                candidate_indices = nearby

        best_dist = np.inf
        best_dir = np.zeros(3)
        for idx in candidate_indices:
            try:
                dist, direction = self._building_surface_features(current_pos, self.obs_dic[idx])
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best_dir = direction

        if np.isfinite(best_dist):
            return float(best_dist), best_dir
        return 100.0, np.zeros(3)

    def _nearest_heightmap_obstacle_features(self, current_pos):
        h_map = getattr(self.planner, "h_map", None)
        if h_map is None or h_map.size == 0:
            return 100.0, np.zeros(3)

        res = float(getattr(self.planner, "res", 1.0))
        min_x = float(getattr(self.planner, "min_x", -h_map.shape[0] * res / 2.0))
        min_y = float(getattr(self.planner, "min_y", -h_map.shape[1] * res / 2.0))
        ix = int(round((current_pos[0] - min_x) / res))
        iy = int(round((current_pos[1] - min_y) / res))
        radius = max(self.safety_radius * 3.0, 5.0)
        window = max(1, int(np.ceil(radius / res)))

        x0, x1 = max(0, ix - window), min(h_map.shape[0], ix + window + 1)
        y0, y1 = max(0, iy - window), min(h_map.shape[1], iy + window + 1)
        if x0 >= x1 or y0 >= y1:
            return 100.0, np.zeros(3)

        local_h = h_map[x0:x1, y0:y1]
        obstacle_mask = local_h > 1.0
        if not np.any(obstacle_mask):
            return 100.0, np.zeros(3)

        xs = min_x + np.arange(x0, x1) * res
        ys = min_y + np.arange(y0, y1) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        closest_z = np.clip(current_pos[2], 0.0, local_h)
        dx = current_pos[0] - X
        dy = current_pos[1] - Y
        dz = current_pos[2] - closest_z
        dists = np.sqrt(dx * dx + dy * dy + dz * dz)
        dists = np.where(obstacle_mask, dists, np.inf)
        flat_idx = int(np.argmin(dists))
        best_dist = float(np.ravel(dists)[flat_idx])
        if not np.isfinite(best_dist):
            return 100.0, np.zeros(3)

        vec = np.array(
            [
                np.ravel(dx)[flat_idx],
                np.ravel(dy)[flat_idx],
                np.ravel(dz)[flat_idx],
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(vec))
        direction = vec / norm if norm > 1e-9 else np.zeros(3)
        return best_dist, direction

    def update_obstacle_features(self, current_pos):
        if self.environment == "generated":
            dist, direction = self._nearest_generated_obstacle_features(current_pos)
        elif self.environment == "custom":
            dist, direction = self._nearest_heightmap_obstacle_features(current_pos)
        else:
            dist, direction = 100.0, np.zeros(3)

        self.nearest_obstacle_dist = float(dist)
        self.nearest_obstacle_dir = np.array(direction, dtype=float)
        if self.nearest_obstacle_dir.shape != (3,):
            self.nearest_obstacle_dir = np.zeros(3)

    def _inter_uav_repulsion_contributions(self, current_pos):
        """Return the unnormalized force contributed by each received UAV state."""
        contributions = {}
        source_ages = {}
        current_pos = np.asarray(current_pos, dtype=float)
        for sender_name, other_pos in self.other_agent_pos.items():
            diff = current_pos - np.asarray(other_pos, dtype=float)
            dist_uav = float(np.linalg.norm(diff))
            contribution = np.zeros(3)
            if 1e-6 < dist_uav < self.safety_radius:
                magnitude = 1.0 - (dist_uav / self.safety_radius)
                contribution = (
                    (diff / dist_uav) * magnitude * self.max_repulsive_force
                ) / 2.0
            contributions[str(sender_name)] = contribution
            source_time = self.last_message_source_time_by_sender.get(sender_name, np.nan)
            source_ages[str(sender_name)] = (
                max(0.0, float(self._sim_time) - float(source_time))
                if np.isfinite(source_time)
                else np.nan
            )
        return contributions, source_ages

    def _combine_repulsive_components(self, obstacle_force_raw, pairwise_raw, source_ages):
        obstacle_force_raw = np.asarray(obstacle_force_raw, dtype=float)
        inter_uav_total = sum(pairwise_raw.values(), np.zeros(3))
        raw_total = obstacle_force_raw + inter_uav_total
        raw_norm = float(np.linalg.norm(raw_total))
        scale = (
            float(self.max_repulsive_force) / raw_norm
            if raw_norm > float(self.max_repulsive_force) and raw_norm > 0.0
            else 1.0
        )
        self.last_inter_uav_repulsion_raw = {
            name: np.asarray(value, dtype=float).copy() for name, value in pairwise_raw.items()
        }
        self.last_inter_uav_repulsion_applied = {
            name: scale * value for name, value in self.last_inter_uav_repulsion_raw.items()
        }
        self.last_inter_uav_repulsion_source_age_s = dict(source_ages)
        self.last_building_repulsive_force_raw = obstacle_force_raw.copy()
        self.last_building_repulsive_force_applied = scale * obstacle_force_raw
        self.last_repulsion_normalization_scale = float(scale)
        return scale * raw_total

    def clear_repulsion_decomposition(self):
        self.last_inter_uav_repulsion_raw = {}
        self.last_inter_uav_repulsion_applied = {}
        self.last_inter_uav_repulsion_source_age_s = {}
        self.last_building_repulsive_force_raw = np.zeros(3)
        self.last_building_repulsive_force_applied = np.zeros(3)
        self.last_repulsion_normalization_scale = 1.0

    def applied_repulsion_components(self):
        """Return obstacle and inter-UAV terms after their shared saturation."""
        obstacle = np.asarray(
            self.last_building_repulsive_force_applied, dtype=float
        ).reshape(-1)
        if obstacle.size != 3:
            obstacle = np.zeros(3)
        inter_uav = sum(
            (
                np.asarray(value, dtype=float).reshape(-1)
                for value in self.last_inter_uav_repulsion_applied.values()
                if np.asarray(value).size == 3
            ),
            np.zeros(3),
        )
        return obstacle, inter_uav

    def compute_repulsive_force(self, current_pos):
        """Generated-world repulsion with exact UAV/building attribution."""
        obstacle_force = np.zeros(3)
        pairwise_raw, source_ages = self._inter_uav_repulsion_contributions(current_pos)

        if self.planner.building_tree is not None:
            indices = self.planner.building_tree.query_ball_point(current_pos[:2], r=15.0)
            for idx in indices:
                obs = self.obs_dic[idx]
                center = np.array(obs["center"])
                h, w, length = obs["height"], obs["width"], obs["length"]
                if current_pos[2] > h + 1.0:
                    continue

                min_x, max_x = center[0] - length / 2, center[0] + length / 2
                min_y, max_y = center[1] - w / 2, center[1] + w / 2
                closest_pt = np.array(
                    [
                        max(min_x, min(current_pos[0], max_x)),
                        max(min_y, min(current_pos[1], max_y)),
                    ]
                )
                diff = current_pos[:2] - closest_pt
                dist = float(np.linalg.norm(diff))
                if dist < 0.01:
                    continue
                if dist < self.safety_radius:
                    magnitude = 1.0 - (dist / self.safety_radius)
                    obstacle_force[:2] += (
                        (diff / dist) * magnitude * self.max_repulsive_force
                    )

        return self._combine_repulsive_components(
            obstacle_force, pairwise_raw, source_ages
        )

    def compute_custom_repulsive_force(self, current_pos):
        """Custom-world repulsion, preserving the planner's raw obstacle term."""
        obstacle_force = self.planner.compute_repulsive_force(
            current_pos,
            self.safety_radius,
            self.max_repulsive_force,
            False,
            True,
            {},
            normalize=False,
        )
        pairwise_raw, source_ages = self._inter_uav_repulsion_contributions(current_pos)
        return self._combine_repulsive_components(
            obstacle_force, pairwise_raw, source_ages
        )

    def update_failure_features(self, current_pos, repulsive_force):
        self.last_repulsive_force = np.array(repulsive_force, dtype=float)
        if self.last_repulsive_force.shape != (3,):
            self.last_repulsive_force = np.zeros(3)
        self.last_repulsive_force_mag = float(np.linalg.norm(repulsive_force))
        self.update_obstacle_features(np.array(current_pos, dtype=float))

        nearest = np.inf
        for source in (self.neighbors_data, self.other_agent_pos):
            for d_name, d_info in source.items():
                if d_name == self.name:
                    continue
                if self.swarm_name and d_name not in self.swarm_name:
                    continue
                other_pos = d_info.get("pos", d_info) if isinstance(d_info, dict) else d_info
                try:
                    dist = float(np.linalg.norm(np.array(current_pos) - np.array(other_pos, dtype=float)))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(dist):
                    nearest = min(nearest, dist)

        self.dist_to_nearest_neighbor = nearest if np.isfinite(nearest) else 100.0

    def contact_link_name(self, body_id, link_id):
        try:
            if link_id == -1:
                raw = p.getBodyInfo(body_id, physicsClientId=self.physics_client_id)[0]
            else:
                raw = p.getJointInfo(body_id, link_id, physicsClientId=self.physics_client_id)[12]
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        except Exception:
            return ""

    def is_ground_contact(self, body_id, link_id):
        name = self.contact_link_name(body_id, link_id).lower()
        return "ground" in name or "plane" in name or name == "world_link"

    def is_uav_contact(self, body_id):
        try:
            raw = p.getBodyInfo(body_id, physicsClientId=self.physics_client_id)[1]
            body_name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        except Exception:
            return False
        return body_name.strip().lower() in {"cf2", "quadrotor", "uav", "drone"}

    def compute_obstacle_collision_flag(self):
        try:
            contacts = p.getContactPoints(
                bodyA=self.bodyId,
                physicsClientId=self.physics_client_id,
            ) or ()
        except Exception:
            return 0

        for contact in contacts:
            other_body = contact[2]
            other_link = contact[4]
            if self.is_ground_contact(other_body, other_link):
                continue
            if self.is_uav_contact(other_body):
                continue
            return 1
        return 0

    def compute_collision_flag(self):
        try:
            contacts = p.getContactPoints(
                bodyA=self.bodyId,
                physicsClientId=self.physics_client_id,
            ) or ()
        except Exception:
            return 0

        for contact in contacts:
            other_body = contact[2]
            other_link = contact[4]
            if not self.is_ground_contact(other_body, other_link):
                return 1
        return 0

    def contact_flags(self):
        ground_contact = False
        non_ground_contact = False
        try:
            contacts = p.getContactPoints(
                bodyA=self.bodyId,
                physicsClientId=self.physics_client_id,
            ) or ()
        except Exception:
            return ground_contact, non_ground_contact

        for contact in contacts:
            other_body = contact[2]
            other_link = contact[4]
            if self.is_ground_contact(other_body, other_link):
                ground_contact = True
            else:
                non_ground_contact = True
        return ground_contact, non_ground_contact

    def outside_world_bounds(self, pos):
        if not self.world_bounds:
            return False

        pos = np.array(pos, dtype=float)
        margin = self.bounds_violation_margin
        if "x" in self.world_bounds:
            lo, hi = self.world_bounds["x"]
            if pos[0] < float(lo) - margin or pos[0] > float(hi) + margin:
                return True
        if "y" in self.world_bounds:
            lo, hi = self.world_bounds["y"]
            if pos[1] < float(lo) - margin or pos[1] > float(hi) + margin:
                return True
        if "z" in self.world_bounds:
            lo, hi = self.world_bounds["z"]
            if pos[2] < float(lo) - margin or pos[2] > float(hi) + margin:
                return True
        return False

    def is_abnormal_crash_state(self, gt, rpy):
        pos = np.array(gt["pos"], dtype=float)
        vel = np.array(gt["vel"], dtype=float)
        speed = float(np.linalg.norm(vel))
        ground_contact, non_ground_contact = self.contact_flags()
        if pos[2] >= self.takeoff_complete_z:
            self.has_taken_off = True

        target_z = float(np.array(self.current_target_pos, dtype=float)[2])
        tipped = abs(float(rpy[0])) > self.crash_tilt_threshold or abs(float(rpy[1])) > self.crash_tilt_threshold
        hard_ground_impact = (
            self.has_taken_off
            and ground_contact
            and float(vel[2]) < -self.ground_impact_speed_threshold
        )
        unexpected_low_altitude = (
            self.has_taken_off
            and self._sim_time > 1.0
            and pos[2] < self.low_altitude_crash_z
            and target_z > self.low_altitude_target_z
        )
        unphysical_speed = speed > self.max_safe_speed
        out_of_bounds = self.outside_world_bounds(pos)
        ground_penetration = pos[2] < self.ground_penetration_z

        if unphysical_speed:
            self.pending_crash_reason = "unphysical_speed"
        elif out_of_bounds:
            self.pending_crash_reason = "out_of_bounds"
        elif ground_penetration:
            self.pending_crash_reason = "ground_penetration"
        elif non_ground_contact:
            self.pending_crash_reason = "non_ground_contact"
        elif hard_ground_impact:
            self.pending_crash_reason = "hard_ground_impact"
        elif self.has_taken_off and ground_contact and tipped:
            self.pending_crash_reason = "ground_contact_tipped"
        elif unexpected_low_altitude:
            self.pending_crash_reason = "unexpected_low_altitude"
        else:
            self.pending_crash_reason = None

        return self.pending_crash_reason is not None

    def update_formation_takeoff_state(self, gt):
        if self.formation_takeoff_complete or not self.swarm_active or self.leader:
            return

        pos = np.array(gt["pos"], dtype=float)
        vel = np.array(gt["vel"], dtype=float)
        stable_altitude = pos[2] >= (
            self.formation_safe_altitude
            - self.formation_takeoff_altitude_tolerance
        )
        stable_vertical_speed = abs(float(vel[2])) <= self.formation_takeoff_max_abs_vz

        if stable_altitude and stable_vertical_speed:
            self.formation_takeoff_unstable_timer = 0.0
            self.formation_takeoff_timer += self.CTRL_DT
            if self.formation_takeoff_timer >= self.formation_takeoff_min_duration:
                self.formation_takeoff_complete = True
                self.formation_transition_timer = 0.0
                self.formation_transition_start_pos = pos.copy()
        else:
            self.formation_takeoff_unstable_timer += self.CTRL_DT
            if (
                self.formation_takeoff_unstable_timer
                > self.formation_takeoff_unstable_grace_duration
            ):
                self.formation_takeoff_timer = 0.0

    def zero_velocity_if_unphysical_crash(self):
        if not self.zero_velocity_on_unphysical_crash:
            return
        if self.crash_reason not in {"unphysical_speed", "out_of_bounds", "ground_penetration"}:
            return
        try:
            p.resetBaseVelocity(
                self.bodyId,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self.physics_client_id,
            )
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # COMMUNICATION
    # ----------------------------------------------------------------------
    def setup_network_swarm(self, ip, port_pub_swarm, port_sub_swarm):
        if self.deterministic_communication:
            self.sub_socket = None
            self.pub_socket = None
            self.radar_sub_socket = None
            return

        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.connect(f"tcp://{ip}:{port_sub_swarm}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1) 
        try: self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error: pass
        
        self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.connect(f"tcp://{ip}:{port_pub_swarm}")
        self.pub_socket.setsockopt_string(zmq.IDENTITY, self.name)


    def radar_com_setup(self, radars_list):

        self.radar_sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.radar_sub_socket.setsockopt(zmq.LINGER, 0)
        self.radar_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.radar_sub_socket.setsockopt(zmq.RCVTIMEO, 1)
        
        try:
            self.radar_sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error:
            pass

        for radar_info in radars_list:
            # On récupère les infos depuis le dictionnaire du config.yaml
            ip = radar_info.get('ip', 'localhost')
            port = radar_info.get('port')
            
            if port:
                address = f"tcp://{ip}:{port}"
                print(f"[{self.name}] Connexion au radar défini dans config : {address}")
                self.radar_sub_socket.connect(address)
            else:
                print(f"[{self.name}] ⚠️ Erreur : Port du radar non spécifié dans la config.")

    def cleanup_network(self):
        """Close all UAV sockets before terminating their private ZMQ context."""

        for attr in ("sub_socket", "pub_socket", "radar_sub_socket"):
            socket = getattr(self, attr, None)
            if socket is None:
                continue
            try:
                socket.close(linger=0)
            except zmq.ZMQError:
                pass
            setattr(self, attr, None)

        context = getattr(self, "zmq_ctx", None)
        if context is not None:
            try:
                context.term()
            except zmq.ZMQError:
                pass
            self.zmq_ctx = None
        
    def broadcast_state(self, pos, vel):
        pos = [round(p, 3) for p in pos]
        vel = [round(v, 3) for v in vel]
        msg = {
            "name": self.name,
            "pos": pos,
            "vel": vel,
            "yaw": round(self.target_yaw_cache, 3),
            "sim_time": round(self._sim_time, 3)
        }
        # The deterministic in-process bus reads the exact same packet that the
        # ZeroMQ transport would serialize. Keeping this snapshot here also
        # decouples simulated communication from wall-clock delivery timing.
        self.last_broadcast_packet = msg
        if self.pub_socket is None:
            return
        self.pub_socket.send_string("State " + json.dumps(msg))

    def listen_radar(self):
        while True:
            try:
                # Lecture non-bloquante
                msg = self.radar_sub_socket.recv_string()
                delay = max(0,self.random_rng.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self._sim_time + delay
                self.message_buffer.append((visible_time,msg))
            except zmq.Again:
                # Plus de messages
                break
            except Exception as e:
                print(f"Erreur réseau sur {self.name}: {e}")
                break
            
        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self._sim_time >= target_time:
                # --- LE MESSAGE EST PRÊT : ON LE TRAITE ---
                if " " in msg:
                    _, json_str = msg.split(" ", 1)
                    try:
                        data = json.loads(json_str)
                        radar_data = data.get("data",{})
                        for d_name, d_info in radar_data.items():
                            self.neighbors_data[d_name] = d_info
                            if d_name != self.name and not self.leader:
                                if d_name in self.swarm_name:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                            if d_name == self.name: 
                                self.radar_reports = d_info
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
                # --- PAS ENCORE PRÊT : ON LE GARDE ---
        # On remplace l'ancien buffer par ceux qui restent
        self.message_buffer = buffer_remaining

    def listen_swarm(self):
        """
        Vérifie la boite aux lettres et met à jour la liste des voisins.
        À appeler à chaque step.
        """
        while True:
            try:
                # Lecture non-bloquante
                msg = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
                delay = max(0,self.random_rng.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self._sim_time + delay
                self.message_buffer.append((visible_time,msg))
            except zmq.Again:
                # Plus de messages
                break
            except Exception as e:
                print(f"Erreur réseau sur {self.name}: {e}")
                break
            
        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self._sim_time >= target_time:
                # --- LE MESSAGE EST PRÊT : ON LE TRAITE ---
                if " " in msg:
                    topic, json_str = msg.split(" ", 1)
                    try:
                        if topic == "SWARM":
                            data = json.loads(json_str)
                            for d_name, d_info in data.items():
                                if self.swarm_name and d_name not in self.swarm_name:
                                    continue
                                self.neighbors_data[d_name] = d_info
                                self.last_message_receive_time_by_sender[d_name] = float(self._sim_time)
                                source_time = d_info.get("sim_time", np.nan)
                                try:
                                    self.last_message_source_time_by_sender[d_name] = float(source_time)
                                except (TypeError, ValueError):
                                    self.last_message_source_time_by_sender[d_name] = np.nan
                                if d_name != self.name and not self.leader:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                        elif topic == "FUTURE_POS" and self.swarm_active and not self.leader:
                            state = json.loads(json_str)
                            new_future_state = state.get(self.name, None)
                            if new_future_state is not None:
                                self.future_state = new_future_state
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
                # --- PAS ENCORE PRÊT : ON LE GARDE ---
        # On remplace l'ancien buffer par ceux qui restent
        self.message_buffer = buffer_remaining

    # ----------------------------------------------------------------------
    # MAIN LOOP & LOGIC
    # ----------------------------------------------------------------------
    @staticmethod
    def _control_stride(physics_dt: float, control_frequency: float) -> int:
        """Return the exact integer number of physics steps per control update."""
        physics_dt = float(physics_dt)
        control_frequency = float(control_frequency)
        if physics_dt <= 0.0:
            raise ValueError("The physics timestep must be strictly positive.")
        if control_frequency <= 0.0:
            raise ValueError("ctrl_freq must be strictly positive.")

        requested_stride = 1.0 / (physics_dt * control_frequency)
        integer_stride = int(round(requested_stride))
        if integer_stride < 1 or not np.isclose(
            requested_stride,
            integer_stride,
            rtol=0.0,
            atol=1e-9,
        ):
            physics_frequency = 1.0 / physics_dt
            raise ValueError(
                "ctrl_freq must divide the physics frequency exactly: "
                f"physics={physics_frequency:.12g} Hz, "
                f"control={control_frequency:.12g} Hz, "
                f"requested stride={requested_stride:.12g}."
            )
        return integer_stride

    def think_and_act(self):
        """
        Run one physics tick and update control on an integer step schedule.
        """
        if not p.isConnected(self.physics_client_id): return
        
        # 1. Update Sim Time
        self._sim_time += self.dt

        #wind
        gt = self.get_ground_truth_state()
        h = gt["pos"][2]
        V_airspeed = np.linalg.norm(gt["vel"] - self.current_wind)

        self.current_wind = self.wind_module.step(h, V_airspeed)
        # 2. Logic schedule. For the standard 240/80 Hz configuration this
        # executes exactly once every three physics ticks.
        current_physics_step = self.physics_step_index
        self.physics_step_index += 1
        if current_physics_step % self.PHYSICS_STEPS_PER_CONTROL == 0:
            self._update_control_loop(gt)

            if self.logging_enabled:
                self._log_full_state(gt)
            self.control_update_index += 1

            self.last_ctrl_time = self._sim_time
        # 3. Physics application (240 Hz in the standard configuration).
        # Use the last calculated RPMs to maintain stability
        if self.crashed:
            self.last_rpms = np.zeros(4)
            self.zero_velocity_if_unphysical_crash()
            self.apply_lib_physics(self.last_rpms, gt)
            return
        self.apply_lib_physics(self.last_rpms, gt)
        


    def _update_control_loop(self,gt):
        """
        High-level logic loop (80 Hz in the standard configuration).
        Handles: Sensors, Communication, Planning, and PID Calculation.
        """
        
        # Never let a skipped/crashed control update reuse previous causal data.
        self.last_counterfactual_control_by_sender = {}
        self.last_counterfactual_control_valid = False
        self.last_counterfactual_control_time = np.nan

        orn_q = np.array(gt["orn_q"])
        ang_vel = np.array(gt["ang_vel"])
        rpy = np.array(p.getEulerFromQuaternion(orn_q))

        # ==================== ADVANCED SENSOR FUSION ====================
        
        # 1. Read Sensors (Noisy)

        
        if self._sim_time >= self.next_gnss_trigger:
            # 1. Mesure et Mise à jour EKF (inchangé)
            meas_pos, meas_vel = self.gnss.measure(gt["pos"], gt["vel"])
            self.ekf.update(meas_pos, meas_vel)
    
            # 2. Calcul du prochain temps de mise à jour
            # On garde la base théorique stable (self.next_gnss_update_time + self.gnss_DT)
            # Et on ajoute le bruit (jitter) juste pour le déclenchement
    
            # Exemple : +/- 10% de variation sur la période
            jitter = max(0,self.random_rng.gauss(self.gnss_delay_mean, self.gnss_delay_std))
    
            # IMPORTANT : On incrémente la cible théorique pour ne pas dériver
            # Si on faisait juste self._sim_time + dt, on accumulerait le retard du bruit.
            # Ici, on repart de l'heure prévue théorique précédente.
    
            # Si c'est la toute première fois ou pour réinitialiser la base théorique 
            # MÉTHODE RECOMMANDÉE ET PLUS SIMPLE (Sans dérive long terme) :
            # On met à jour last_gnss_update_time théorique
            self.last_gnss_update_time += self.gnss_DT
    
            # Mais le prochain "check" se fera avec un décalage
            self.next_gnss_trigger = self.last_gnss_update_time + jitter


        # Read the IMU (Acceleration + Orientation)
        # Note: In simple PyBullet, you can cheat and take gt['orn_q']
        # or use self.imu.measure(...) if your IMU sensor is complete.
        # Here I assume self.imu returns the raw accel.
        imu_acc, _ = self.imu.measure(gt["vel"], gt["orn_q"]) 
        # Note: Make sure your IMUSensor returns a np.array for imu_acc
        
        # 2. EKF Prediction (Based on IMU)
        self.ekf.predict(imu_acc_body=imu_acc, orientation_quat=gt["orn_q"])
        
        # 3. EKF Correction (GPS)
        # GPS corrects for IMU drift
        pos = self.ekf.x[:3]
        vel = self.ekf.x[3:6]
        # Reset known obstacles periodically

        # --- SENSORS (Lidar) ---
        if (self._sim_time - self.last_lidar_time) >= self.lidar_period:
            self.last_lidar_time = self._sim_time
            # new_pts = self.lidar.measure(pos, rpy[0], rpy[2], rpy[1])
            # if len(new_pts) > 0: ...

        # --- COMMUNICATION ---
        if self.swarm_active or self.leader:
            if (self._sim_time - self.last_com_time) >= self.com_period:
                self.last_com_time = self._sim_time
                self.broadcast_state(pos, vel)

        # Receive Messages
        if self.sub_socket is not None : 
            self.listen_swarm()
        if self.radar_sub_socket is not None: 
            self.listen_radar()


        if self.is_abnormal_crash_state(gt, rpy):
            self.crash_contact_steps += 1
        else:
            self.crash_contact_steps = 0

        if self.crash_contact_steps >= self.crash_contact_threshold:
            self.crashed = True
            self.crash_reason = self.pending_crash_reason
            self.last_rpms = np.zeros(4)
            self.zero_velocity_if_unphysical_crash()
            return

        self.update_formation_takeoff_state(gt)

        # --- TARGET LOGIC ---
        target_pos = pos 
        target_vel = np.zeros(3)
        goal_pos_for_approach = None
        holding_formation_takeoff = False
        in_formation_transition = False
        central_application_scale = 0.0
        central_future_state = None
        self.set_applied_central_avoidance_metadata(None, 0.0)
        
        # 1. Deterministic motion profile for an independent negative-control UAV.
        if self.motion_profile_enabled:
            target_pos, target_vel = self.sinusoidal_motion_target()

        # 2. The leader takes off vertically and waits for formation assembly.
        elif self.swarm_active and self.leader and self.formation_hold_active:
            holding_formation_takeoff = True
            target_pos = np.array(
                [
                    self.start_pos[0],
                    self.start_pos[1],
                    self.formation_safe_altitude,
                ],
                dtype=float,
            )
            target_vel = np.zeros(3)

        # 3. Swarm Follower
        elif self.swarm_active and not self.leader:
            future_state = self.future_state or {"pos": pos, "vel": np.zeros(3), "yaw": self.target_yaw_cache}
            central_future_state = future_state
            desired_swarm_pos = np.array(future_state["pos"], dtype=float)
            desired_swarm_vel = np.array(future_state.get("vel", [0, 0, 0]), dtype=float)
            desired_swarm_pos[2] = max(desired_swarm_pos[2], self.formation_safe_altitude)

            if not self.formation_takeoff_complete:
                holding_formation_takeoff = True
                target_pos = np.array(
                    [
                        self.start_pos[0],
                        self.start_pos[1],
                        self.formation_safe_altitude,
                    ],
                    dtype=float,
                )
                target_vel = np.zeros(3)
            elif self.formation_transition_timer < self.formation_transition_duration:
                in_formation_transition = True
                if self.formation_transition_start_pos is None:
                    self.formation_transition_start_pos = np.array(pos, dtype=float)

                self.formation_transition_timer += self.CTRL_DT
                alpha = np.clip(self.formation_transition_timer / self.formation_transition_duration, 0.0, 1.0)
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                central_application_scale = float(smooth_alpha)
                target_pos = (
                    (1.0 - smooth_alpha) * self.formation_transition_start_pos
                    + smooth_alpha * desired_swarm_pos
                )
                target_vel = np.zeros(3)
            else:
                central_application_scale = 1.0
                target_pos = desired_swarm_pos
                if self.swarm_target_vel is not None:
                    target_vel = desired_swarm_vel
                if target_pos[2] <= self.formation_safe_altitude:
                    target_vel[2] = max(target_vel[2], 0.0)

            self.set_applied_central_avoidance_metadata(
                central_future_state, central_application_scale
            )

        # 4. Planning (Wait)
        elif self.is_planning:
            target_pos = pos 
            target_vel = -1 * vel # Brake
            
        # 5. Autonomous Navigation
        else:
            # --- FAILSAFE CHECK ---
            if self.Calculation_fail_count > 5:
                print(f"[{self.name}] ⚠️ Too many A* failures ({self.Calculation_fail_count}). Skipping WP.")
                self.wp_idx += 1
                self.Calculation_fail_count = 0
                self.replan_timer = 0
                return # Skip this cycle to reset logic
            # ----------------------

            else:
            # Detect arrival at Waypoint
                if self.wp_idx < len(self.waypoints):
                    dist_wp = np.linalg.norm(self.waypoints[self.wp_idx] - pos)

                    if dist_wp < self.waypoint_reached_radius and not self.is_planning:
                        print(f"\n[{self.name}] WAYPOINT VALIDÉ !")
                        print(f" -> Cible théorique (Target) : {self.waypoints[self.wp_idx]}")
                        print(f" -> Position réelle (PyBullet): {pos}")
                        print(f" -> Distance calculée : {dist_wp} mètres")
                        print(f"[{self.name}] Waypoint {self.wp_idx} reached.")
                        self.wp_idx += 1
                        self.active_path = [] # Force a new calculation
                        self.replan_timer = 0

                while self.wp_idx < len(self.waypoints):
                    global_target = self.waypoints[self.wp_idx]
                
                    if len(self.active_path) == 0:
                        # Force A* planning to trigger for each new WP
                        if self.replan_timer <= 0:
                            self.trigger_planning(pos, global_target)
                            self.replan_timer = 100
                        break 
                    break

            # Follow Path
            if len(self.active_path) > 0:
                local_target = self.active_path[0]
                if np.linalg.norm(local_target - pos) < self.path_point_reached_radius:
                    self.active_path.pop(0)
                    if len(self.active_path) > 0:
                        local_target = self.active_path[0]
                    elif self.wp_idx < len(self.waypoints):
                        local_target = self.waypoints[self.wp_idx]
                target_pos = local_target
                if self.wp_idx < len(self.waypoints):
                    goal_pos_for_approach = self.waypoints[self.wp_idx]
                
            elif self.wp_idx < len(self.waypoints):
                target_pos = self.waypoints[self.wp_idx]
                if np.linalg.norm(target_pos - pos) < 0.3:
                    print(f"[{self.name}] WP {self.wp_idx} Reached.")
                    self.wp_idx += 1
                else:
                    goal_pos_for_approach = target_pos

        if self.replan_timer > 0: self.replan_timer -= 1
        target_pos = self.clamp_to_world_bounds(target_pos)

        # STOCKAGE DE LA CIBLE POUR LE LOGGING
        self.current_target_pos = target_pos

        # --- CONTROL COMMANDS ---
        # Repulsive Force
        F_rep = np.zeros(3)
        if self.environment == "generated":
            F_rep = self.compute_repulsive_force(pos)    
        if self.environment == "custom":
            F_rep = self.compute_custom_repulsive_force(pos)
        if holding_formation_takeoff:
            F_rep = np.zeros(3)
            self.clear_repulsion_decomposition()
        self.update_failure_features(pos, F_rep)
        
        guidance_before = (
            None
            if self.last_guidance_dir_xy is None
            else np.asarray(self.last_guidance_dir_xy, dtype=float).copy()
        )
        virtual_target_pos, final_target_vel, final_target_pos = (
            self._finalize_pid_targets(
                pos=pos,
                vel=vel,
                target_pos=target_pos,
                target_vel=target_vel,
                goal_pos_for_approach=goal_pos_for_approach,
                repulsive_force=F_rep,
                holding_formation_takeoff=holding_formation_takeoff,
                in_formation_transition=in_formation_transition,
            )
        )
        self.current_target_vel = np.asarray(final_target_vel, dtype=float).copy()
        
        # Yaw
        direction_vec = final_target_pos - pos
        if self.swarm_active and self.swarm_target_yaw is not None:
            future_state = self.future_state or {"yaw": self.target_yaw_cache}
            self.target_yaw_cache = future_state.get("yaw", self.target_yaw_cache)
        elif np.linalg.norm(direction_vec[:2]) > 0.5:
            self.target_yaw_cache = np.arctan2(direction_vec[1], direction_vec[0])

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        # Compute the real PID command once, then replay counterfactual commands
        # from the exact same internal PID state without affecting the simulation.
        pid_state_before = self._snapshot_pid_state()
        rpms, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.CTRL_DT,
            state=state_vec, 
            target_pos=virtual_target_pos, 
            target_vel=final_target_vel, 
            target_rpy=np.array([0, 0, self.target_yaw_cache]) 
        )
        pid_state_after = self._snapshot_pid_state()
        self.last_rpms = rpms
        self._compute_pair_counterfactuals(
            state_vec=state_vec,
            pos=pos,
            vel=vel,
            target_pos=target_pos,
            target_vel=target_vel,
            goal_pos_for_approach=goal_pos_for_approach,
            repulsive_force=F_rep,
            holding_formation_takeoff=holding_formation_takeoff,
            in_formation_transition=in_formation_transition,
            guidance_before=guidance_before,
            actual_virtual_target_pos=virtual_target_pos,
            actual_target_vel=final_target_vel,
            actual_rpms=rpms,
            pid_state_before=pid_state_before,
            pid_state_after=pid_state_after,
        )
        
    def apply_lib_physics(self, rpms, gt):
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        forces = np.array(rpms**2) * self.KF
        torques = np.array(rpms**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(self.bodyId, i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        
        try:
            p.applyExternalTorque(self.bodyId, 4, [0, 0, z_torque], p.LINK_FRAME)
            rot = np.array(p.getMatrixFromQuaternion(gt["orn_q"])).reshape(3,3)
            v_air_world = gt["vel"] - self.current_wind
            v_air_body = rot.T @ v_air_world
            prop_wash_factor = np.sum(2 * np.pi * rpms / 60)
            drag_force_body = -1 * self.DRAG_COEFF * prop_wash_factor * v_air_body
            f_drag_for_bullet = rot @ drag_force_body 
            p.applyExternalForce(self.bodyId, -1, forceObj=f_drag_for_bullet, posObj=[0,0,0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        except:
            pass 

    def _log_full_state(self, gt):
        """
        Logging avancé pour l'analyse causale sans perturber l'affichage console.
        """
        collision_flag = self.compute_collision_flag()
        collision_with_obstacle_flag = self.compute_obstacle_collision_flag()
        ground_contact, non_ground_contact = self.contact_flags()
        
        meas_pos = self.ekf.x[:3]
        rpy = np.asarray(p.getEulerFromQuaternion(gt["orn_q"]), dtype=float)
        angular_velocity = np.asarray(gt["ang_vel"], dtype=float)
        target_velocity = np.asarray(self.current_target_vel, dtype=float)
        rpms = np.full(4, np.nan)
        available_rpms = np.asarray(self.last_rpms, dtype=float).reshape(-1)
        rpms[: min(4, available_rpms.size)] = available_rpms[:4]
        waypoint = self.current_waypoint_for_log()
        obstacle_repulsion, inter_uav_repulsion = self.applied_repulsion_components()
        
        # Derived metrics
        gnss_error = np.linalg.norm(np.array(meas_pos) - np.array(gt["pos"]))
        wind_mag = np.linalg.norm(self.current_wind)
        tracking_error = np.linalg.norm(np.array(gt["pos"]) - np.array(self.current_target_pos))
        distance_to_waypoint = self.compute_distance_to_current_waypoint(gt["pos"])
        formation_error = self.compute_formation_error(gt)
        rpm_mean = float(np.mean(self.last_rpms)) if len(self.last_rpms) else 0.0
        rpm_max = float(np.max(self.last_rpms)) if len(self.last_rpms) else 0.0
        motor_saturation_ratio = (
            float(np.mean(np.array(self.last_rpms) >= 0.98 * self.MAX_RPM))
            if self.MAX_RPM > 0 and len(self.last_rpms)
            else 0.0
        )
        crash_flag = int(bool(self.crashed))
        if crash_flag and self.first_crash_time is None:
            self.first_crash_time = float(self._sim_time)

        self.tracking_error_sum += float(tracking_error)
        self.tracking_error_count += 1
        self.tracking_error_max = max(self.tracking_error_max, float(tracking_error))
        if np.isfinite(self.dist_to_nearest_neighbor):
            self.min_nearest_neighbor_dist_seen = min(
                self.min_nearest_neighbor_dist_seen,
                float(self.dist_to_nearest_neighbor),
            )

        # with open(self.log_file, "a", newline="") as f:
        row = [
            round(self._sim_time, 3),
            # Ground Truth
            *gt["pos"], *gt["vel"],
            # Sensors
            *meas_pos,
            gnss_error,
            # Env
            *self.current_wind,
            wind_mag,
            # Interaction
            round(self.last_repulsive_force_mag, 3),
            *self.last_repulsive_force,
            round(self.nearest_obstacle_dist, 3),
            *self.nearest_obstacle_dir,
            round(self.dist_to_nearest_neighbor, 3),
            # Intent
            *self.current_target_pos,
            tracking_error,
            collision_flag,
            collision_with_obstacle_flag,
            self.runId,
            self.name,
            self.swarm_id,
            int(bool(self.leader)),
            int(self.wp_idx),
            distance_to_waypoint,
            self.formation_type,
            *self.desired_formation_offset,
            formation_error,
            crash_flag,
            self.crash_reason or "none",
            self.pending_crash_reason or "none",
            int(bool(ground_contact)),
            int(bool(non_ground_contact)),
            rpm_mean,
            rpm_max,
            motor_saturation_ratio,
            int(self.intervention_active),
            self.intervention_type,
            *self.external_force,
            # Append-only relational state instrumentation.
            *rpy,
            *angular_velocity,
            *target_velocity,
            *rpms,
            *waypoint,
            float(self.safety_radius),
            # Applied components sum to the historical rep_force_* vector.
            float(np.linalg.norm(obstacle_repulsion)),
            *obstacle_repulsion,
            float(np.linalg.norm(inter_uav_repulsion)),
            *inter_uav_repulsion,
            float(self.last_repulsion_normalization_scale),
        ]
        if len(row) != len(self.log_header):
            raise RuntimeError(
                f"UAV log row/header mismatch: {len(row)} values for "
                f"{len(self.log_header)} columns."
            )
        # Clean float formatting
        self.log_data_buffer.append(row)

    def compute_distance_to_current_waypoint(self, pos):
        if self.wp_idx >= len(self.waypoints):
            return 0.0
        try:
            return float(np.linalg.norm(np.array(self.waypoints[self.wp_idx]) - np.array(pos)))
        except Exception:
            return 0.0

    def current_waypoint_for_log(self):
        if 0 <= self.wp_idx < len(self.waypoints):
            waypoint = np.asarray(self.waypoints[self.wp_idx], dtype=float)
            if waypoint.shape == (3,):
                return waypoint
        return np.full(3, np.nan)

    def compute_formation_error(self, gt):
        if self.leader or self.swarm_leader_ref is None:
            return 0.0

        try:
            leader_state = self.swarm_leader_ref.get_ground_truth_state()
            leader_pos = np.array(leader_state["pos"], dtype=float)
            leader_yaw = p.getEulerFromQuaternion(leader_state["orn_q"])[2]
            if hasattr(self, "future_state") and isinstance(self.future_state, dict):
                leader_yaw = float(self.future_state.get("yaw", leader_yaw))
            pos = np.array(gt["pos"], dtype=float)
        except Exception:
            return 0.0

        cy = np.cos(leader_yaw)
        sy = np.sin(leader_yaw)
        r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        desired_world_offset = r_yaw @ self.desired_formation_offset
        current_world_offset = pos - leader_pos
        return float(np.linalg.norm(current_world_offset - desired_world_offset))

    def get_run_summary(self):
        tracking_mean = (
            self.tracking_error_sum / self.tracking_error_count
            if self.tracking_error_count > 0
            else 0.0
        )
        return {
            "name": self.name,
            "swarm_id": self.swarm_id,
            "is_leader": bool(self.leader),
            "crashed": bool(self.crashed),
            "first_crash_time": self.first_crash_time,
            "crash_reason": self.crash_reason or "none",
            "pending_crash_reason": self.pending_crash_reason or "none",
            "formation_type": self.formation_type,
            "desired_offset": self.desired_formation_offset.tolist(),
            "waypoint_idx": int(self.wp_idx),
            "waypoints_total": len(self.waypoints),
            "tracking_error_mean": float(tracking_mean),
            "tracking_error_max": float(self.tracking_error_max),
            "min_nearest_neighbor_dist": float(self.min_nearest_neighbor_dist_seen),
        }
        
    # ----------------------------------------------------------------------
    # SAVES FUNCTIONS
    # ----------------------------------------------------------------------

    # Ferme le fichier csv à la fin de la simulation
    def write_csv(self):

        print(f"[{self.name}] Sauvegarde de {len(self.log_data_buffer)} lignes dans le CSV...")

        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.log_header)

            for row in self.log_data_buffer:
                formatted_row = [x if isinstance(x, (int, str)) else round(float(x), 4) for x in row]
                writer.writerow(formatted_row)

        self.log_data_buffer.clear()
        print(f"[{self.name}] Sauvegarde CSV terminée.")

    
    # ----------------------------------------------------------------------
    def get_state(self):

        return
