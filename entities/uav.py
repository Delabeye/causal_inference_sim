import os
import csv
import threading
import pybullet as p
import numpy as np
import zmq 
import json

# Imports Utilitaires & Contrôle
from utilities.utilities import point_in_cube, point_in_cylinder, discretize_obstacles
from entities.agent import Agent
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF
from Control.Path_planning import AStarPlanner

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class UAV(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float, known_obstacles_config: list[dict]):
        """
        Initialize a UAV entity with physics simulation, control systems, and autonomous capabilities.
        Args:
            config (dict): Configuration dictionary containing:
                - name (str, optional): UAV identifier. Defaults to "UAV".
                - body_id (int, optional): Physics body ID. Defaults to 0000.
                - mass (float, optional): UAV mass in kg. Defaults to 1.5.
                - urdf_path (str, optional): Path to URDF model file. Defaults to "assets/quadrotor.urdf".
                - start_pos (list, optional): Initial position [x, y, z]. Defaults to [0, 0, 1.0].
                - start_orn_euler (list, optional): Initial orientation in Euler angles. Defaults to [0, 0, 0].
                - waypoints (list, optional): List of waypoint coordinates. Defaults to [[0, 0, 1]].
                - astar (dict, optional): A* planner configuration with world bounds.
                - sensors (dict, optional): Sensor configurations for GPS, IMU, and Lidar.
                - physics (dict, optional): Physics parameters including thrust_coeff, torque_coeff, max_rpm.
            physics_client_id (int): PyBullet physics client identifier.
            dt (float): Physics simulation timestep in seconds.
            known_obstacles_config (list[dict]): Configuration list for known static obstacles.
        Initializes:
            - Physics engine connection and body properties
            - Control system (DSL PID controller at 100Hz)
            - Navigation system with waypoint planning
            - A* path planner with world bounds
            - Sensor suite (EKF, GPS, IMU, Lidar)
            - Swarm coordination parameters
            - Communication timing
            - Logging system
        """
        """"""
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
        
        # --- CONTROL FREQUENCY OPTIMIZATION (100Hz) ---
        self.CTRL_FREQ = 100.0  
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.last_ctrl_time = -self.CTRL_DT # Force update at t=0

        # --- PHYSIQUE ---
        self.KF = self.config.get("physics", {}).get("thrust_coeff", 6.11e-8)
        self.KM = self.config.get("physics", {}).get("torque_coeff", 1.5e-9)
        self.G = 9.81
        self.MAX_RPM = config.get("physics", {}).get("max_rpm", 22000.0)
        self.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])
        
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.last_rpms = np.zeros(4)
        
        # --- NAVIGATION ---
        wp_list = config.get("waypoints", [])
        if not wp_list: wp_list = [[0,0,1]]
        first_wp = np.array(self.start_pos)+np.array([0,0,1])
        self.waypoints = [first_wp] + [np.array(w) for w in wp_list]
        self.wp_idx = 0
        
        # --- OBSTACLES & PLANNING ---
        self.obs_dic = known_obstacles_config
        self.known_obstacles = discretize_obstacles(known_obstacles_config)
        self.obstacles = self.known_obstacles.copy()
        self.total_obstacles = [] # Combined list for avoidance
        print(len(self.known_obstacles), "points d'obstacles connus chargés.")
        
        a_config = {"world_bounds": {"x": [-10, 10], "y": [-10, 10], "z": [0.1, 5.0]}}
        self.planner = AStarPlanner(self.config.get("astar", a_config))
        
        self.target_yaw_cache = 0.0
        
        # États du Planning
        self.active_path = []
        self.is_planning = False      
        self.planning_thread = None   
        self.replan_timer = 0         
        self.Calculation_fail_count = 0

        # --- SWARM CONTROL ---
        self.swarm_active = False
        self.swarm_target_pos = None
        self.swarm_target_vel = None
        self.swarm_target_yaw = None 
        self.leader = False
        self.swarm_pos = [] 

        # --- COMMUNICATION ---
        self.com_period = 0.1 
        self.last_com_time = -self.com_period
        
        # --- CAPTEURS ---
        sens = config.get("sensors", {})
        self.ekf = GPSEKF(dt); self.ekf.x[:3] = self.start_pos
        self.gps = GPSSensor(sens.get("gps", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
        self.lidar = LidarSensor(sens.get("lidar", {}))
        
        lidar_freq = sens.get("lidar", {}).get("frequency", 10.0)
        self.lidar_period = 1.0 / lidar_freq
        self.last_lidar_time = -self.lidar_period
        
        # Logs
        self.logging_enabled = True
        self.log_file = os.path.join("logs", f"{self.name}.csv")
        os.makedirs("logs", exist_ok=True)
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        self.broadcast_state(pos=self.start_pos, vel=[0,0,0])
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    # ----------------------------------------------------------------------
    # API SWARM 
    # ----------------------------------------------------------------------
    def set_swarm_activate(self):
        self.swarm_active = True
        self.future_state = {"pos": np.array(self.start_pos), "vel": np.zeros(3), "yaw": 0.0}
        self.swarm_pos = []

    # ----------------------------------------------------------------------
    # Obstacle management and path planning
    # ----------------------------------------------------------------------
    def _trigger_planning(self, start_pos, target_pos, obstacles):
        if self.is_planning: return 
        self.is_planning = True
        print(f"[{self.name}] ⏳ Starting A* Thread...")
        obs_copy = list(obstacles) 
        self.planning_thread = threading.Thread(
            target=self._run_async_plan, 
            args=(start_pos, target_pos, obs_copy)
        )
        self.planning_thread.daemon = True 
        self.planning_thread.start()

    def _run_async_plan(self, start_pos, target_pos, obstacles):
        try:
            path = self.planner.plan(start_pos, target_pos, obstacles, smooth=True)
            if path and len(path) > 0:
                self.active_path = path 
                self.Calculation_fail_count = 0
            else:
                self.Calculation_fail_count += 1
        except Exception as e:
            print(f"[{self.name}] 💥 Error thread A*: {e}")
        finally:
            self.is_planning = False 

    def _analyze_target_accessibility(self, start_pos, target_pos, obs):
        for obstacle in self.obs_dic:
            if obstacle["type"] == "cube":
                if point_in_cube(target_pos, obstacle): return "INVALID"
            elif obstacle["type"] == "sphere":
                if np.linalg.norm(np.array(target_pos) - np.array(obstacle["center"])) <= obstacle["radius"]: return "INVALID"
            elif obstacle["type"] == "cylinder":
                if point_in_cylinder(target_pos, obstacle): return "INVALID"
        
        check_obs = obs if len(obs) < 2000 else obs[::2]
        if len(check_obs) > 0:
            vec_dir = target_pos - start_pos
            dist_target = np.linalg.norm(vec_dir)
            if dist_target > 0.1:
                vec_dir /= dist_target
                for pt in check_obs:
                    pt_vec = np.array(pt) - start_pos
                    proj = np.dot(pt_vec, vec_dir)
                    if 0 < proj < dist_target:
                        dist_ortho = np.linalg.norm(pt_vec - proj * vec_dir)
                        if dist_ortho < 0.6: return "BLOCKED"
        return "CLEAR"
    
    def _compute_repulsive_force(self, current_pos, obstacles, safety_radius=1.0, max_force=2.0):
        force_vec = np.array([0.0, 0.0, 0.0])
        if not obstacles: return force_vec

        obs_to_check = obstacles[::5] if len(obstacles) > 500 else obstacles
        for obs in obs_to_check:
            diff_vec = current_pos - np.array(obs)
            diff_vec[2] = 0.0 # Ignore Z
            dist = np.linalg.norm(diff_vec)
            if 0.05 < dist < safety_radius:
                coef = (1.0 - (dist / safety_radius))
                repulsion = diff_vec / dist * coef * max_force
                force_vec += repulsion

        total_norm = np.linalg.norm(force_vec)
        if total_norm > max_force:
            force_vec = (force_vec / total_norm) * max_force
        return force_vec
    
    # ----------------------------------------------------------------------
    # COMMUNICATION 
    # ----------------------------------------------------------------------
    def setup_network(self, ip, port_pub, port_sub):
        self.zmq_ctx = zmq.Context()
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{ip}:{port_sub}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1) 
        try: self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error: pass
        
        self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{ip}:{port_pub}")
        self.pub_socket.setsockopt_string(zmq.IDENTITY, self.name)
        self.neighbors_data = {} 
        
    def broadcast_state(self, pos, vel):
        if not hasattr(self, "pub_socket"): return 
        pos = [round(p, 3) for p in pos]
        vel = [round(v, 3) for v in vel]
        msg = {
            "name": self.name,
            "pos": pos,
            "vel": vel,
            "yaw": round(self.target_yaw_cache, 3),
            "sim_time": round(self._sim_time, 3)
        }
        self.pub_socket.send_string("State " + json.dumps(msg))

    # ----------------------------------------------------------------------
    # MAIN LOOP & LOGIC
    # ----------------------------------------------------------------------
    def think_and_act(self):
        """
        Main simulation loop (Physics Frequency = 240 Hz).
        Handles physical application and schedules the logic loop.
        """
        if not p.isConnected(self.physics_client_id): return
        
        # 1. Update Sim Time
        self._sim_time += self.dt
        
        # 2. Logic Schedule (100 Hz)
        if (self._sim_time - self.last_ctrl_time) >= self.CTRL_DT:
            self._update_control_loop()
            self.last_ctrl_time = self._sim_time
            
        # 3. Physics Application (Always 240Hz)
        # Using the last calculated RPMs to maintain stability
        gt = self.get_ground_truth_state()
        self._apply_lib_physics(self.last_rpms, gt)
        
        # Logging (Optional: Log at physics freq or control freq)
        if int(self._sim_time/self.dt) % 10 == 0:
             self._log(gt["pos"])

    def _update_control_loop(self):
        """
        High-Level Logic Loop (100 Hz).
        Handles: Sensors, Communication, Planning, and PID Calculation.
        """
        # --- STATE UPDATE ---
        gt = self.get_ground_truth_state()
        pos, vel = gt["pos"], gt["vel"]
        orn_q, ang_vel = gt["orn_q"], gt["ang_vel"]
        rpy = p.getEulerFromQuaternion(orn_q)

        # Reset known obstacles periodically
        if self._sim_time > 0 and (self._sim_time % 10) < self.dt:
            self.obstacles = self.known_obstacles.copy()

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
        try: 
            while True:
                msg = self.sub_socket.recv_string()
                if " " not in msg: continue
                title, data_str = msg.split(" ", 1)

                if title == "SWARM":
                    incoming_data = json.loads(data_str)
                    for d_name, d_info in incoming_data.items():
                        self.neighbors_data[d_name] = d_info
                        if d_name != self.name and not self.leader:
                            self.swarm_pos.append(d_info["pos"])

                elif title == "FUTURE_POS" and self.swarm_active and not self.leader:
                    state = json.loads(data_str)
                    self.future_state = state.get(self.name, None)
        except zmq.Again:
            pass

        # Compile Obstacles
        self.total_obstacles = self.obstacles.copy()
        if self.swarm_active and not self.leader:
            for other_pos in self.swarm_pos:
                self.total_obstacles.append(np.round(other_pos, 2))
            self.swarm_pos = []

        # --- TARGET LOGIC ---
        target_pos = pos 
        target_vel = np.zeros(3)

        # 1. Swarm Follower
        if self.swarm_active and not self.leader:          
            target_pos = self.future_state["pos"]
            if self.swarm_target_vel is not None:
                target_vel = self.future_state["vel"]

        # 2. Planning (Wait)
        elif self.is_planning:
            target_pos = pos 
            target_vel = -0.5 * vel # Brake
            
        # 3. Autonomous Navigation
        else:
            # --- FAILSAFE CHECK ---
            if self.Calculation_fail_count > 5:
                print(f"[{self.name}] ⚠️ Too many A* failures ({self.Calculation_fail_count}). Skipping WP.")
                self.wp_idx += 1
                self.Calculation_fail_count = 0
                self.replan_timer = 0
                return # Skip this cycle to reset logic
            # ----------------------

            while self.wp_idx < len(self.waypoints):
                global_target = self.waypoints[self.wp_idx]
                
                if len(self.active_path) == 0:
                    status = self._analyze_target_accessibility(pos, global_target, self.obstacles)
                    
                    if status == "INVALID":
                        print(f"[{self.name}] Target {self.wp_idx} Wall. Skip.")
                        self.wp_idx += 1
                        continue 
                    elif status == "BLOCKED":
                        if self.replan_timer <= 0:
                            self._trigger_planning(pos, global_target, self.obstacles)
                            self.replan_timer = 50 
                        break 
                break

            # Follow Path
            if len(self.active_path) > 0:
                local_target = self.active_path[0]
                if np.linalg.norm(local_target - pos) < 0.4:
                    self.active_path.pop(0)
                    if len(self.active_path) > 0:
                        local_target = self.active_path[0]
                    elif self.wp_idx < len(self.waypoints):
                        local_target = self.waypoints[self.wp_idx]
                target_pos = local_target
                
            elif self.wp_idx < len(self.waypoints):
                target_pos = self.waypoints[self.wp_idx]
                if np.linalg.norm(target_pos - pos) < 0.2:
                    print(f"[{self.name}] WP {self.wp_idx} Reached.")
                    self.wp_idx += 1

        if self.replan_timer > 0: self.replan_timer -= 1

        # --- CONTROL COMMANDS ---
        # Repulsive Force
        F_rep = self._compute_repulsive_force(pos, self.total_obstacles, safety_radius=0.45, max_force=1.0)    
        drone_mass = self.config.get("mass", 0.03) 
        acc_rep = F_rep / drone_mass
        
        final_target_vel = target_vel + (acc_rep * 5 * self.CTRL_DT) # Use CTRL_DT here

        # Clamp Speed
        max_speed_xy = 5.0 
        speed_xy = np.linalg.norm(final_target_vel[:2])
        if speed_xy > max_speed_xy:
            ratio = max_speed_xy / speed_xy
            final_target_vel[:2] *= ratio
        
        # PID Target Helper
        final_target_pos = target_pos + (final_target_vel * self.CTRL_DT)
        vector_to_target = final_target_pos - pos
        dist_to_target = np.linalg.norm(vector_to_target)
        if dist_to_target > 1.0:
            virtual_target_pos = pos + (vector_to_target / dist_to_target) * 1.0
        else:
            virtual_target_pos = final_target_pos
        
        # Yaw
        direction_vec = final_target_pos - pos
        if self.swarm_active and self.swarm_target_yaw is not None:
            self.target_yaw_cache = self.future_state["yaw"]
        elif np.linalg.norm(direction_vec[:2]) > 0.5:
            self.target_yaw_cache = np.arctan2(direction_vec[1], direction_vec[0])

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        # Compute RPMs (PID)
        rpms, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.CTRL_DT, # Important: 0.01s
            state=state_vec, 
            target_pos=virtual_target_pos, 
            target_vel=final_target_vel, 
            target_rpy=np.array([0, 0, self.target_yaw_cache]) 
        )
        
        self.last_rpms = rpms

    def _apply_lib_physics(self, rpms, gt):
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        forces = np.array(rpms**2) * self.KF
        torques = np.array(rpms**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(self.bodyId, i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        
        try:
            p.applyExternalTorque(self.bodyId, 4, torqueObj=[0, 0, z_torque], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
            rot = np.array(p.getMatrixFromQuaternion(gt["orn_q"])).reshape(3,3)
            drag = -1 * self.DRAG_COEFF * np.sum(2 * np.pi * rpms / 60)
            f_drag = rot @ (drag * (rot.T @ gt["vel"]))
            p.applyExternalForce(self.bodyId, 4, forceObj=f_drag, posObj=[0,0,0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        except:
            pass 

    def _log(self, pos):
        if int(self._sim_time/self.dt)%10==0:
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow([round(self._sim_time,3), *pos])