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
            - ctrl_freq (int): Control loop frequency in Hz. Defaults to 100.
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
            - Control system: DSL PID controller running at configurable frequency (default 100Hz).
            - Navigation: Waypoint management and A* path planning with async thread support.
            - Sensor fusion: EKF (Extended Kalman Filter) with GNSS, IMU, and Lidar.
            - Obstacle avoidance: Repulsive force computation and collision detection.
            - Swarm coordination: Leader-follower formation control and neighbor tracking.
            - Communication: ZMQ-based state broadcasting and swarm message handling.
            - Wind simulation: Dryden Gust Model for realistic turbulence.
            - Logging: CSV trajectory tracking (simulation time, position).
        
        Note:
            Physics loop runs at 240Hz; logic/control loop runs at configurable frequency (100Hz default).
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
        
        # --- CONTROL FREQUENCY OPTIMIZATION (100Hz) ---
        self.CTRL_FREQ = self.config.get("ctrl_freq",100)
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.last_ctrl_time = -self.CTRL_DT # Force update at t=0

        # --- PHYSICS ---
        self.KF = self.config.get("physics", {}).get("thrust_coeff", 6.11e-8)
        self.KM = self.config.get("physics", {}).get("torque_coeff", 1.5e-9)
        self.G = 9.81
        self.MAX_RPM = config.get("physics", {}).get("max_rpm", 22000.0)
        self.max_speed = config.get("physics", {}).get("max_spedd", 5)
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
        self.obs_dic = known_obstacles_config # Combined list for avoidance
        print(len(self.obs_dic), "known obstacle points loaded.")
        self.environment = world_type
        
        self.planner = planner
        
        self.target_yaw_cache = 0.0
        
        self.max_repulsive_force = self.config.get("physics",{}).get("max_repulsive_force",2.0)
        self.safety_radius = self.config.get("physics",{}).get("safety_radius",2.0)

        # Planning States
        self.active_path = []
        self.is_planning = False      
        self.planning_thread = None   
        self.replan_timer = 0         
        self.Calculation_fail_count = 0

        self.zmq_ctx = zmq.Context()
        self.sub_socket = None
        self.pub_socket = None
        self.radar_sub_socket = None 

        # --- SWARM CONTROL ---
        self.swarm_active = False
        self.swarm_target_pos = None
        self.swarm_target_vel = None
        self.swarm_target_yaw = None 
        self.leader = False
        self.other_agent_pos = {}
        self.neighbors_data = {} 
        self.swarm_name=[]
        # --- COMMUNICATION ---
        com=self.config.get("communication")
        self.com_period = com.get("com_period",0.1)
        self.last_com_time = -self.com_period
        self.message_buffer = []
        self.perception_delay_mean = com.get("com_delay_mean",0.1)  # 100ms de retard
        self.perception_delay_std = com.get("com_delay_std",0.02)  # +/- 20ms
        # --- SENSORS ---
        sens = self.config.get("sensors", {})
        self.ekf = EKF(self.CTRL_DT); self.ekf.x[:3] = self.start_pos
        self.gnss = GNSSensor(sens.get("gnss", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
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
        self.wind_module = DrydenGustModel(self.dt,self.turbulence,self.mean_wind)
        
        # --- CAUSAL ANALYSIS & LOGGING SETUP ---
        self.logging_enabled = True
        self.runId = self.config.get("run_config", 0)
        self.log_file = os.path.join("logs", f"run_{self.runId}_{self.name}.csv")
        os.makedirs("logs", exist_ok=True)
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
            "nearest_neighbor_dist",
            "target_x", "target_y", "target_z", # Intent                
            "tracking_error_mag",
            "collision_flag"                # Flags
        ]

        # Variables internes pour le logging
        self.last_repulsive_force_mag = 0.0
        self.dist_to_nearest_neighbor = 100.0
        self.current_target_pos = self.start_pos

        if self.pub_socket is not None:
            self.broadcast_state(pos=self.start_pos, vel=[0,0,0])
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    # ----------------------------------------------------------------------
    # SWARM API
    # ----------------------------------------------------------------------
    def set_swarm_activate(self):
        self.swarm_active = True
        self.future_state = {"pos": np.array(self.start_pos), "vel": np.zeros(3), "yaw": 0.0}

    # ----------------------------------------------------------------------
    # Obstacle management and path planning
    # ----------------------------------------------------------------------
    def _trigger_planning(self, start_pos, target_pos):
        if not self.is_planning:
            self.is_planning = True
            print(f"[{self.name}] ⏳ Starting A* Thread...")
            self.planning_thread = threading.Thread(
            target=self._run_async_plan, 
            args=(start_pos, target_pos)
            )
            self.planning_thread.daemon = True 
            self.planning_thread.start()

    def _run_async_plan(self, start_pos, target_pos):
        try:
            path = self.planner.plan(start_pos, target_pos)
            if path and len(path) > 0:
                self.active_path = path 
                self.Calculation_fail_count = 0
            else:
                self.Calculation_fail_count += 1
        except Exception as e:
            print(f"[{self.name}] 💥 Error in A* thread: {e}")
        finally:
            self.is_planning = False 

    def _analyze_target_accessibility(self, start_pos, target_pos):
        for obstacle in self.obs_dic:
                if point_in_cube(target_pos, obstacle): return "INVALID"
        if p.raycast(start_pos,target_pos)[1][0]>= 0:
            return "BLOCKED"
        return "CLEAR"
    
    # Dans causal_inference_sim/entities/uav.py

    def _compute_repulsive_force(self, current_pos):
        force_vec = np.array([0.0, 0.0, 0.0])

        if self.other_agent_pos != {}:
            for _,other_pos in self.other_agent_pos.items():
                diff = current_pos - other_pos
                dist_uav = np.linalg.norm(diff)
                if dist_uav < self.safety_radius:
                    mag = (1.0 - (dist_uav / self.safety_radius))
                    force_vec += ((diff / dist_uav) * mag * self.max_repulsive_force)/2

        # On vérifie que le planner et son index sont prêts
        if self.planner.building_tree is None:
            total_norm = np.linalg.norm(force_vec)
            if total_norm > self.max_repulsive_force:
                force_vec = (force_vec / total_norm) * self.max_repulsive_force
            return force_vec

    # 1. Trouver les indices des bâtiments proches (ex: rayon 15m)
        indices = self.planner.building_tree.query_ball_point(current_pos[:2], r=15.0)
    
        for idx in indices:
            obs = self.obs_dic[idx] 
            center = np.array(obs["center"])
            h, w, l = obs["height"], obs["width"], obs["length"]
        
            if current_pos[2] > h + 1.0: 
                continue

            # 1. Définir les limites du rectangle (AABB)
            min_x = center[0] - l / 2
            max_x = center[0] + l / 2
            min_y = center[1] - w / 2
            max_y = center[1] + w / 2

            # 2. Trouver le point le plus proche sur ou dans le rectangle
            # On "clappe" la position du drone entre les bornes du bâtiment
            closest_x = max(min_x, min(current_pos[0], max_x))
            closest_y = max(min_y, min(current_pos[1], max_y))
            closest_pt = np.array([closest_x, closest_y])

            # 3. Calculer le vecteur de distance
            diff = current_pos[:2] - closest_pt
            dist = np.linalg.norm(diff)

            # Cas particulier : si le drone est pile à l'intérieur (dist ~ 0)
            # on crée une force pour le repousser vers l'extérieur
            if dist < 0.01:
                # On peut ignorer ou repousser vers le bord le plus proche
                continue

            # 4. Appliquer la force repulsive
            if dist < self.safety_radius:
                mag = (1.0 - (dist / self.safety_radius))
                # Le vecteur (diff / dist) est maintenant toujours perpendiculaire au mur
                force_vec[:2] += (diff / dist) * mag * self.max_repulsive_force

        # Normalisation finale
        total_norm = np.linalg.norm(force_vec)
        if total_norm > self.max_repulsive_force:
            force_vec = (force_vec / total_norm) * self.max_repulsive_force

        return force_vec
    
    # ----------------------------------------------------------------------
    # COMMUNICATION
    # ----------------------------------------------------------------------
    def setup_network_swarm(self, ip, port_pub_swarm, port_sub_swarm):

        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{ip}:{port_sub_swarm}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1) 
        try: self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error: pass
        
        self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{ip}:{port_pub_swarm}")
        self.pub_socket.setsockopt_string(zmq.IDENTITY, self.name)


    def radar_com_setup(self, radars_list):

        self.radar_sub_socket = self.zmq_ctx.socket(zmq.SUB)
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

    def listen_radar(self):
        while True:
            try:
                # Lecture non-bloquante
                msg = self.radar_sub_socket.recv_string()
                delay = max(0,random.gauss(self.perception_delay_mean, self.perception_delay_std))
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
                msg = self.sub_socket.recv_string()
                delay = max(0,random.gauss(self.perception_delay_mean, self.perception_delay_std))
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
                                self.neighbors_data[d_name] = d_info
                                if d_name != self.name and not self.leader:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                        elif topic == "FUTURE_POS" and self.swarm_active and not self.leader:
                            state = json.loads(json_str)
                            self.future_state=state.get(self.name,None)
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
    def think_and_act(self):
        """
        Main simulation loop (Physics Frequency = 240 Hz).
        Handles physical application and schedules the logic loop.
        """
        if not p.isConnected(self.physics_client_id): return
        
        # 1. Update Sim Time
        self._sim_time += self.dt

        #wind
        gt = self.get_ground_truth_state()
        h = gt["pos"][2]
        V_airspeed = np.linalg.norm(gt["vel"] - self.current_wind)

        self.current_wind = self.wind_module.step(h, V_airspeed)
        # 2. Logic Schedule (100 Hz)
        if (self._sim_time - self.last_ctrl_time) >= self.CTRL_DT:
            self._update_control_loop(gt)

            # Add the current row in the drone logs (CHANGEMENT)
            self._log_full_state(gt)

            self.last_ctrl_time = self._sim_time
        # 3. Physics Application (Always 240Hz)
        # Use the last calculated RPMs to maintain stability
        self._apply_lib_physics(self.last_rpms, gt)
        


    def _update_control_loop(self,gt):
        """
        High-Level Logic Loop (100 Hz).
        Handles: Sensors, Communication, Planning, and PID Calculation.
        """
        
        orn_q = np.array(gt["orn_q"])
        ang_vel = np.array(gt["ang_vel"])
        rpy = np.array(p.getEulerFromQuaternion(orn_q))

        # ==================== ADVANCED SENSOR FUSION ====================
        
        # 1. Read Sensors (Noisy)
        if (self._sim_time - self.last_gnss_update_time) >= self.gnss_DT:
            meas_pos, meas_vel = self.gnss.measure(gt["pos"], gt["vel"])
            self.ekf.update(meas_pos, meas_vel)
            self.last_gnss_update_time = self._sim_time
        
        if self._sim_time >= self.next_gnss_trigger:
            # 1. Mesure et Mise à jour EKF (inchangé)
            meas_pos, meas_vel = self.gnss.measure(gt["pos"], gt["vel"])
            self.ekf.update(meas_pos, meas_vel)
    
            # 2. Calcul du prochain temps de mise à jour
            # On garde la base théorique stable (self.next_gnss_update_time + self.gnss_DT)
            # Et on ajoute le bruit (jitter) juste pour le déclenchement
    
            # Exemple : +/- 10% de variation sur la période
            jitter = max(0,random.gauss(self.gnss_delay_mean, self.gnss_delay_std))
    
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
            target_vel = -1 * vel # Brake
            
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

            else:
            # Detect arrival at Waypoint
                if self.wp_idx < len(self.waypoints):
                    dist_wp = np.linalg.norm(self.waypoints[self.wp_idx] - pos)
                    if dist_wp < 0.5 and not self.is_planning:
                        print(f"[{self.name}] Waypoint {self.wp_idx} reached.")
                        self.wp_idx += 1
                        self.active_path = [] # Force a new calculation
                        self.replan_timer = 0

                while self.wp_idx < len(self.waypoints):
                    global_target = self.waypoints[self.wp_idx]
                
                    if len(self.active_path) == 0:
                        # Force A* planning to trigger for each new WP
                        if self.replan_timer <= 0:
                            self._trigger_planning(pos, global_target)
                            self.replan_timer = 100
                        break 
                    break

            # Follow Path
            if len(self.active_path) > 0:
                local_target = self.active_path[0]
                if np.linalg.norm(local_target - pos) < 0.7:
                    self.active_path.pop(0)
                    if len(self.active_path) > 0:
                        local_target = self.active_path[0]
                    elif self.wp_idx < len(self.waypoints):
                        local_target = self.waypoints[self.wp_idx]
                target_pos = local_target
                
            elif self.wp_idx < len(self.waypoints):
                target_pos = self.waypoints[self.wp_idx]
                if np.linalg.norm(target_pos - pos) < 0.3:
                    print(f"[{self.name}] WP {self.wp_idx} Reached.")
                    self.wp_idx += 1

        if self.replan_timer > 0: self.replan_timer -= 1
        
        # STOCKAGE DE LA CIBLE POUR LE LOGGING
        self.current_target_pos = target_pos

        # --- CONTROL COMMANDS ---
        # Repulsive Force
        if self.environment == "generated":
            F_rep = self._compute_repulsive_force(pos)    
        if self.environment == "custom":
            F_rep= self.planner.compute_repulsive_force(pos,self.safety_radius,self.max_repulsive_force,self.swarm_active,self.leader,self.other_agent_pos)
        
        acc_rep = F_rep / self.mass

        final_target_vel = target_vel + (acc_rep * 3 * self.CTRL_DT) # Use CTRL_DT here

        # Clamp Speed
        speed_xy = np.linalg.norm(final_target_vel[:2])
        if speed_xy > self.max_speed:
            ratio = self.max_speed / speed_xy
            final_target_vel[:2] *= ratio
        
        # PID Target Helper
        final_target_pos = target_pos + (final_target_vel * self.CTRL_DT)
        vector_to_target = final_target_pos - pos
        dist_to_target = np.linalg.norm(vector_to_target)
        if dist_to_target > 3:
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
        # Detect collision (simple proximity check for log flag)
        collision_flag = 0
        if len(p.getContactPoints(self.bodyId)) > 0:
            collision_flag = 1
        
        meas_pos = self.ekf.x[:3]
        
        # Derived metrics
        gnss_error = np.linalg.norm(np.array(meas_pos) - np.array(gt["pos"]))
        wind_mag = np.linalg.norm(self.current_wind)
        tracking_error = np.linalg.norm(np.array(gt["pos"]) - np.array(self.current_target_pos))

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
            round(self.dist_to_nearest_neighbor, 3),
            # Intent
            *self.current_target_pos,
            tracking_error,
            collision_flag
        ]
        # Clean float formatting
        self.log_data_buffer.append(row)
        


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