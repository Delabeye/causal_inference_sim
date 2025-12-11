import os
import csv
import pybullet as p
import numpy as np

from entities.agent import Agent
from Control.PID import PIDController
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF


class UAV(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = self.config.get("name", "unnamed_uav")

        # ----------- EKF / Lidar / Logging -----------
        self.ekf: GPSEKF | None = None
        self.lidar_sensor: LidarSensor | None = None
        self.last_lidar_points = None

        log_cfg = self.config.get("logging", {})
        self.logging_enabled = bool(log_cfg.get("enabled", False))
        self.log_file_path = None
        self._sim_time = 0.0

        if self.logging_enabled:
            log_dir = log_cfg.get("dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            base_name = log_cfg.get("file", f"{self.name}_log.csv")
            self.log_file_path = os.path.join(log_dir, base_name)
            if os.path.isfile(self.log_file_path):
                try: os.remove(self.log_file_path)
                except: pass
            print(f"[{self.name}] Logging activé -> {self.log_file_path}")

        # ----------- Pose initiale -----------
        urdf_path = self.config["urdf_path"]
        start_pos = self.config.get("start_pos", [0.0, 0.0, 0.2])
        start_pos = [float(v) for v in start_pos]
        start_orn_euler = self.config.get("start_orn_euler", [0.0, 0.0, 0.0])
        start_orn_q = p.getQuaternionFromEuler(start_orn_euler)

        self.current_yaw = float(start_orn_euler[2])
        self.current_roll = 0.0
        self.current_pitch = 0.0

        super().__init__(
            urdf_path=urdf_path,
            start_pos=start_pos,
            start_orn_q=start_orn_q,
            physics_client_id=self.physics_client_id,
            dt=self.dt,
        )
        self._create_body_frame_axes(axis_length=0.5)

        # ----------- Paramètres physiques (AMÉLIORATION STABILITÉ) -----------
        self.g = 9.81
        num_joints = p.getNumJoints(self.bodyId, physicsClientId=self.physics_client_id)
        total_mass = p.getDynamicsInfo(self.bodyId, -1, physicsClientId=self.physics_client_id)[0] or 0.0
        for j in range(num_joints):
            mj = p.getDynamicsInfo(self.bodyId, j, physicsClientId=self.physics_client_id)[0]
            if mj is not None:
                total_mass += mj
        self.mass = float(total_mass) if total_mass > 0 else 0.03

        # --- AJOUT CRUCIAL : Damping (Frottement de l'air) ---
        # Cela empêche le drone de glisser indéfiniment et tue les micro-oscillations
        p.changeDynamics(self.bodyId, -1, linearDamping=0.5, angularDamping=1.0, physicsClientId=self.physics_client_id)

        # ----------- PID -----------
        components_cfg = self.config.get("components", {})
        def get_pid_cfg(name: str, fallback: dict = None) -> dict:
            if name in components_cfg: return components_cfg[name]
            return fallback if fallback else {"gains": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0}, "windup": 0.0}

        cfg_z = get_pid_cfg("controller_z")
        cfg_x = get_pid_cfg("controller_x", fallback=cfg_z)
        cfg_y = get_pid_cfg("controller_y", fallback=cfg_z)

        self.acc_limit_xy = float(self.config.get("acc_limit_xy", 5.0))
        self.acc_limit_z = float(self.config.get("acc_limit_z", 5.0))

        self.pid_x = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_x)
        self.pid_y = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_y)
        self.pid_z = PIDController(self.acc_limit_z, self.acc_limit_z, cfg_z)

        F_max_factor = float(self.config.get("F_max_factor", 2.5))
        self.F_max = F_max_factor * self.mass * self.g

        # Paramètres de vol
        self.pos_tolerance = 0.20
        self.vel_tolerance = 0.20
        
        self.yaw_align_gain = 3.0 # Réduit
        self.yaw_rate_max = np.radians(90.0) # Réduit
        self.yaw_align_threshold = np.radians(10.0)
        self.tilt_gain = 0.3 # Réduit
        self.pitch_max = np.radians(30.0)
        self.roll_max = np.radians(30.0)
        self.tilt_smoothing = 0.2 # Réduit (lissage plus fort sur l'attitude)
        self.process_noise_std = max(0.0, float(self.config.get("process_noise_std", 0.0)))

        # ----------- Waypoints -----------
        wp_list = self.config.get("waypoints", None)
        if wp_list is not None and len(wp_list) > 0:
            self.waypoints = [np.array(w, dtype=float) for w in wp_list]
        else:
            self.waypoints = [np.array(start_pos, dtype=float)]
        self.current_wp_idx = 0

        self.last_gps_meas = None
        self.last_imu_meas = None
        self.last_control_accel = np.zeros(3, dtype=float)
        self.last_attitude_cmd = np.array([self.current_roll, self.current_pitch, self.current_yaw], dtype=float)
        self.last_wp_index = 0
        self.last_wp_distance = np.nan

        print(f"UAV '{self.name}' chargé.")

    def _initialize_components(self):
        self.components = {}
        sensors_cfg = self.config.get("sensors", {})

        if sensors_cfg.get("gps", {}).get("enabled", False):
            self.gps = GPSSensor(sensors_cfg["gps"])
        else: self.gps = None

        if sensors_cfg.get("imu", {}).get("enabled", False):
            self.imu = IMUSensor(sensors_cfg["imu"])
        else: self.imu = None

        ekf_cfg = sensors_cfg.get("ekf", {})
        if self.gps is not None:
            r_pos = float(ekf_cfg.get("r_pos", 0.3))
            r_vel = float(ekf_cfg.get("r_vel", 0.1))
            accel_std = float(ekf_cfg.get("accel_noise_std", 0.2))

            self.ekf = GPSEKF(dt=self.dt, r_pos=r_pos, r_vel=r_vel, accel_noise_std=accel_std)
            print(f"[{self.name}] EKF ACTIVÉ.")
        else:
            self.ekf = None

        lidar_cfg = sensors_cfg.get("lidar", {})
        if lidar_cfg.get("enabled", False):
            self.lidar_sensor = LidarSensor(
                float(lidar_cfg.get("max_distance", 10.0)),
                float(lidar_cfg.get("angle_resolution", 1.0))
            )
        else:
            self.lidar_sensor = None

    def _create_body_frame_axes(self, axis_length=0.3):
        p.addUserDebugLine([0,0,0], [axis_length,0,0], [1,0,0], parentObjectUniqueId=self.bodyId, parentLinkIndex=-1)
        p.addUserDebugLine([0,0,0], [0,axis_length,0], [0,1,0], parentObjectUniqueId=self.bodyId, parentLinkIndex=-1)
        p.addUserDebugLine([0,0,0], [0,0,axis_length], [0,0,1], parentObjectUniqueId=self.bodyId, parentLinkIndex=-1)

    def set_target_pos(self, target):
        self.waypoints = [np.array(target, dtype=float)]
        self.current_wp_idx = 0

    def set_waypoints(self, waypoints):
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.current_wp_idx = 0

    def _get_active_target(self) -> np.ndarray:
        if self.current_wp_idx >= len(self.waypoints): return self.waypoints[-1]
        return self.waypoints[self.current_wp_idx]

    def _update_waypoint_if_reached(self, pos, vel):
        target = self._get_active_target()
        dist = np.linalg.norm(target - pos)
        velocity = np.linalg.norm(vel)
        if dist < self.pos_tolerance and velocity < self.vel_tolerance:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1

    def get_lidar_data(self, sensor_position, roll, yaw, pitch):
        return self.lidar_sensor.measure(sensor_position, roll, yaw, pitch) if self.lidar_sensor else []

    def _log_state(self, pos_true, vel_true):
        if not self.logging_enabled or not self.log_file_path: return
        
        xg, yg, zg, vxg, vyg, vzg = [np.nan]*6
        xe, ye, ze, vxe, vye, vze = [np.nan]*6
        p_xx, p_yy, p_xy = np.nan, np.nan, np.nan

        if self.last_gps_meas and self.last_gps_meas[0] is not None:
            xg, yg, zg = self.last_gps_meas[0]
            vxg, vyg, vzg = self.last_gps_meas[1]
        
        if self.ekf is not None:
            try:
                state = self.ekf.x
                if not np.isnan(state).all():
                    xe, ye, ze = state[0], state[1], state[2]
                    vxe, vye, vze = state[3], state[4], state[5]
                    P = self.ekf.P
                    p_xx, p_yy, p_xy = P[0, 0], P[1, 1], P[0, 1]
            except: pass

        row = {
            "t": self._sim_time,
            "x_true": float(pos_true[0]), "y_true": float(pos_true[1]), "z_true": float(pos_true[2]),
            "vx_true": float(vel_true[0]), "vy_true": float(vel_true[1]), "vz_true": float(vel_true[2]),
            "x_gps": xg, "y_gps": yg, "z_gps": zg,
            "vx_gps": vxg, "vy_gps": vyg, "vz_gps": vzg,
            "x_ekf": xe, "y_ekf": ye, "z_ekf": ze,
            "vx_ekf": vxe, "vy_ekf": vye, "vz_ekf": vze,
            "P_xx": p_xx, "P_yy": p_yy, "P_xy": p_xy,
            "ax_cmd": self.last_control_accel[0], "ay_cmd": self.last_control_accel[1], "az_cmd": self.last_control_accel[2],
            "roll_cmd": self.last_attitude_cmd[0], "pitch_cmd": self.last_attitude_cmd[1], "yaw_cmd": self.last_attitude_cmd[2],
            "wp_index": self.last_wp_index, "dist_to_wp": self.last_wp_distance
        }
        
        file_exists = os.path.isfile(self.log_file_path)
        with open(self.log_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists: writer.writeheader()
            writer.writerow(row)

    def think_and_act(self, setpoint_pos: np.ndarray | None = None, setpoint_vel: np.ndarray | None = None):
        if not p.isConnected(self.physics_client_id): return
        
        # --- 1. Consigne ---
        use_feedforward = False
        target_vel = np.zeros(3)

        if setpoint_pos is not None:
            # SUIVEUR
            target_pos = np.array(setpoint_pos)
            if setpoint_vel is not None:
                target_vel = np.array(setpoint_vel)
                use_feedforward = True 
        else:
            # LEADER
            try:
                r_pos, _ = p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
                r_vel, _ = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
                self._update_waypoint_if_reached(np.array(r_pos), np.array(r_vel))
            except: pass
            target_pos = self._get_active_target()

        # --- 2. Vérité Terrain ---
        try:
            pos, orn_q = p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
            vel, _ = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
            pos = np.array(pos); vel = np.array(vel)
            self.current_roll, self.current_pitch, self.current_yaw = p.getEulerFromQuaternion(orn_q)
        except p.error: return

        # --- 3. EKF ---
        imu_acc = None
        if self.imu:
            _, ang_v = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
            imu_acc, _ = self.imu.measure(pos, orn_q, vel, ang_v, self.dt)

        m_pos = None
        if self.gps:
            m_pos, m_vel = self.gps.measure(pos, vel)
            self.last_gps_meas = (m_pos, m_vel)

        if self.ekf:
            if m_pos is not None:
                self.ekf.step(m_pos, m_vel, imu_accel=imu_acc, orientation_q=orn_q)
            elif self.ekf._initialized:
                self.ekf.predict(imu_accel=imu_acc, orientation_q=orn_q)
            
            if self.lidar_sensor:
                z_meas = self.lidar_sensor.measure_altitude(pos, orn_q)
                self.ekf.update_height(z_meas)

        # --- 4. Contrôle ---
        # Utilisation EKF pour le contrôle
        ctrl_pos = self.ekf.x[0:3] if self.ekf else pos
        ctrl_vel = self.ekf.x[3:6] if self.ekf else vel

        e_pos = target_pos - ctrl_pos
        self.last_wp_distance = float(np.linalg.norm(e_pos))

        # Yaw Soft
        dir_w = e_pos.copy(); dir_w[2] = 0.
        yaw_des = np.arctan2(dir_w[1], dir_w[0]) if np.linalg.norm(dir_w) > 0.5 else self.current_yaw
        yaw_err = np.arctan2(np.sin(yaw_des - self.current_yaw), np.cos(yaw_des - self.current_yaw))
        yaw_rate = np.clip(1.5 * yaw_err, -self.yaw_rate_max, self.yaw_rate_max) # Gain yaw très doux
        self.current_yaw += yaw_rate * self.dt

        # PID
        ax = self.pid_x.compute(e_pos[0], self.dt)
        ay = self.pid_y.compute(e_pos[1], self.dt)
        az = self.pid_z.compute(e_pos[2], self.dt)

        # Feedforward Soft
        if use_feedforward:
            kv_ff = 0.5 # Gain très faible pour ne pas brusquer
            ax += kv_ff * (target_vel[0] - ctrl_vel[0])
            ay += kv_ff * (target_vel[1] - ctrl_vel[1])
            az += kv_ff * (target_vel[2] - ctrl_vel[2])

        # Saturation
        ax = np.clip(ax, -self.acc_limit_xy, self.acc_limit_xy)
        ay = np.clip(ay, -self.acc_limit_xy, self.acc_limit_xy)
        az = np.clip(az, -self.acc_limit_z, self.acc_limit_z)

        # --- LISSAGE PUISSANT (Low Pass Filter) ---
        # C'est ça qui arrête les tremblements : on lisse la commande physique
        alpha = 0.1 # 10% new, 90% old -> Très inerte
        self.last_control_accel[0] = (1-alpha)*self.last_control_accel[0] + alpha*ax
        self.last_control_accel[1] = (1-alpha)*self.last_control_accel[1] + alpha*ay
        self.last_control_accel[2] = (1-alpha)*self.last_control_accel[2] + alpha*az
        
        s_ax, s_ay, s_az = self.last_control_accel

        cy, sy = np.cos(self.current_yaw), np.sin(self.current_yaw)
        acc_fwd = s_ax * cy + s_ay * sy
        acc_right = -s_ax * sy + s_ay * cy
        
        p_des = np.clip(self.tilt_gain * acc_fwd, -self.pitch_max, self.pitch_max)
        r_des = np.clip(-self.tilt_gain * acc_right, -self.roll_max, self.roll_max)

        self.last_attitude_cmd = [r_des, p_des, yaw_des]

        # --- 5. Physique ---
        try:
            target_orn = p.getQuaternionFromEuler([r_des, p_des, self.current_yaw])
            curr_lin_vel, _ = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
            
            p.resetBasePositionAndOrientation(self.bodyId, pos, target_orn, physicsClientId=self.physics_client_id)
            p.resetBaseVelocity(self.bodyId, linearVelocity=curr_lin_vel, angularVelocity=[0,0,0], physicsClientId=self.physics_client_id)

            F_world = self.mass * np.array([s_ax, s_ay, s_az + self.g])
            # Suppression du bruit de process (inutile et déstabilisant pour des PID simples)
            # if self.process_noise_std > 0: ... (Commenté volontairement)

            p.applyExternalForce(self.bodyId, -1, F_world.tolist(), pos.tolist(), p.WORLD_FRAME, self.physics_client_id)

            if self.lidar_sensor:
                self.last_lidar_points = self.get_lidar_data(np.array(pos), self.current_roll, self.current_yaw, self.current_pitch)
        
        except p.error: return

        self._log_state(pos, vel)
        self._sim_time += self.dt

    def apply_physics(self, *args, **kwargs): pass