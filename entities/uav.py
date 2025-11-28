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

        # ----------- Paramètres physiques -----------
        self.g = 9.81
        num_joints = p.getNumJoints(self.bodyId, physicsClientId=self.physics_client_id)
        total_mass = p.getDynamicsInfo(self.bodyId, -1, physicsClientId=self.physics_client_id)[0] or 0.0
        for j in range(num_joints):
            mj = p.getDynamicsInfo(self.bodyId, j, physicsClientId=self.physics_client_id)[0]
            if mj is not None:
                total_mass += mj
        self.mass = float(total_mass) if total_mass > 0 else 0.03

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

        self.pos_tolerance = float(self.config.get("pos_tolerance", 0.05))
        self.vel_tolerance = float(self.config.get("vel_tolerance", 0.05))

        self.yaw_align_gain = float(self.config.get("yaw_align_gain", 4.0))
        self.yaw_rate_max = float(self.config.get("yaw_rate_max_deg", 120.0)) * np.pi / 180.0
        self.yaw_align_threshold = float(self.config.get("yaw_align_threshold_deg", 10.0)) * np.pi / 180.0

        self.tilt_gain = float(self.config.get("tilt_gain", 0.4))
        self.pitch_max = float(self.config.get("pitch_max_deg", 35.0)) * np.pi / 180.0
        self.roll_max = float(self.config.get("roll_max_deg", 35.0)) * np.pi / 180.0
        self.tilt_smoothing = np.clip(float(self.config.get("tilt_smoothing", 0.5)), 0.0, 1.0)

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
        """Initialise capteurs et EKF."""
        self.components = {}
        sensors_cfg = self.config.get("sensors", {})

        # GPS
        gps_cfg = sensors_cfg.get("gps", {})
        if gps_cfg.get("enabled", False):
            self.gps = GPSSensor(gps_cfg)
            self.components["gps"] = self.gps
        else:
            self.gps = None

        # IMU
        imu_cfg = sensors_cfg.get("imu", {})
        if imu_cfg.get("enabled", False):
            self.imu = IMUSensor(imu_cfg)
            self.components["imu"] = self.imu
        else:
            self.imu = None

        # EKF
        ekf_cfg = sensors_cfg.get("ekf", {})
        if self.gps is not None:
            r_pos = float(ekf_cfg.get("r_pos", 0.3))
            r_vel = float(ekf_cfg.get("r_vel", 0.1))
            accel_std = float(ekf_cfg.get("accel_noise_std", 0.2))

            self.ekf = GPSEKF(
                dt=self.dt,
                r_pos=r_pos,
                r_vel=r_vel,
                accel_noise_std=accel_std
            )
            self.components["ekf"] = self.ekf
            print(f"[{self.name}] EKF ACTIVÉ.")
        else:
            self.ekf = None
            print(f"[{self.name}] Pas de GPS -> Pas d'EKF.")

        # Lidar
        lidar_cfg = sensors_cfg.get("lidar", {})
        if lidar_cfg.get("enabled", False):
            self.lidar_sensor = LidarSensor(
                float(lidar_cfg.get("max_distance", 10.0)),
                float(lidar_cfg.get("angle_resolution", 1.0))
            )
            self.components["lidar"] = self.lidar_sensor
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
        if np.linalg.norm(target - pos) < self.pos_tolerance and np.linalg.norm(vel) < self.vel_tolerance:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1

    def get_lidar_data(self, sensor_position, roll, yaw, pitch):
        return self.lidar_sensor.measure(sensor_position, roll, yaw, pitch) if self.lidar_sensor else []

    def _log_state(self, pos_true, vel_true):
        if not self.logging_enabled or not self.log_file_path: return
        
        xg, yg, zg, vxg, vyg, vzg = [np.nan]*6
        xe, ye, ze, vxe, vye, vze = [np.nan]*6
        p_xx, p_yy, p_xy = np.nan, np.nan, np.nan

        # GPS
        if self.last_gps_meas and self.last_gps_meas[0] is not None:
            xg, yg, zg = self.last_gps_meas[0]
            vxg, vyg, vzg = self.last_gps_meas[1]
        
        # EKF : Lecture directe de l'état
        if self.ekf is not None:
            try:
                state = self.ekf.x
                if not np.isnan(state).all():
                    xe, ye, ze = state[0], state[1], state[2]
                    vxe, vye, vze = state[3], state[4], state[5]
                    P = self.ekf.P
                    p_xx = P[0, 0]
                    p_yy = P[1, 1]
                    p_xy = P[0, 1]
            except Exception:
                pass

        row = {
            "t": self._sim_time,
            "x_true": pos_true[0], "y_true": pos_true[1], "z_true": pos_true[2],
            "vx_true": vel_true[0], "vy_true": vel_true[1], "vz_true": vel_true[2],
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

    def think_and_act(self, setpoint: np.ndarray | None = None):
        if not p.isConnected(self.physics_client_id): return
        if setpoint is not None: self.set_target_pos(setpoint)

        try:
            state = self.get_ground_truth_state()
        except p.error: return
        pos, vel = state["pos"], state["vel"]
        orn_q, ang_vel = state["orn_q"], state["ang_vel"]

        imu_acc = None
        if self.imu is not None:
            sf_body, gyro_body = self.imu.measure(pos, orn_q, vel, ang_vel, self.dt)
            self.last_imu_meas = (sf_body, gyro_body)
            imu_acc = sf_body

        # EKF : Predict toujours, Update si GPS
        m_pos, m_vel = None, None
        if self.gps is not None:
            m_pos, m_vel = self.gps.measure(pos, vel)
            self.last_gps_meas = (m_pos, m_vel)

        if self.ekf is not None:
            if m_pos is not None:
                self.ekf.step(m_pos, m_vel, imu_accel=imu_acc, orientation_q=orn_q)
            elif self.ekf._initialized:
                self.ekf.predict(imu_accel=imu_acc, orientation_q=orn_q)

        # PID
        self._update_waypoint_if_reached(pos, vel) 
        target = self._get_active_target()
        e_pos = target - pos
        dist = float(np.linalg.norm(e_pos))
        self.last_wp_distance = dist

        dir_world = e_pos.copy()
        dir_world[2] = 0.0
        yaw_des = np.arctan2(dir_world[1], dir_world[0]) if np.linalg.norm(dir_world) > 1e-6 else self.current_yaw
        yaw_err = np.arctan2(np.sin(yaw_des - self.current_yaw), np.cos(yaw_des - self.current_yaw))
        yaw_rate = np.clip(self.yaw_align_gain * yaw_err, -self.yaw_rate_max, self.yaw_rate_max)
        self.current_yaw += yaw_rate * self.dt

        cy, sy = np.cos(self.current_yaw), np.sin(self.current_yaw)
        v_fwd = cy * vel[0] + sy * vel[1]
        v_side = -sy * vel[0] + cy * vel[1]
        p_des = np.clip(self.tilt_gain * v_fwd, -self.pitch_max, self.pitch_max)
        r_des = np.clip(self.tilt_gain * v_side, -self.roll_max, self.roll_max)
        self.current_pitch = (1-self.tilt_smoothing)*self.current_pitch + self.tilt_smoothing*p_des
        self.current_roll = (1-self.tilt_smoothing)*self.current_roll + self.tilt_smoothing*r_des
        self.last_attitude_cmd = [self.current_roll, self.current_pitch, self.current_yaw]

        if dist < self.pos_tolerance and np.linalg.norm(vel) < self.vel_tolerance:
            self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()
            ax, ay, az = 0.0, 0.0, 0.0
            a_tot = [0,0,self.g]
        else:
            ax = np.clip(self.pid_x.compute(e_pos[0], self.dt), -self.acc_limit_xy, self.acc_limit_xy)
            ay = np.clip(self.pid_y.compute(e_pos[1], self.dt), -self.acc_limit_xy, self.acc_limit_xy)
            az = np.clip(self.pid_z.compute(e_pos[2], self.dt), -self.acc_limit_z, self.acc_limit_z)
            if abs(yaw_err) > self.yaw_align_threshold: ax, ay = 0.0, 0.0
            a_tot = [ax, ay, az + self.g]

        self.last_control_accel = [ax, ay, az]
        F = self.mass * np.array(a_tot)
        if np.linalg.norm(F) > self.F_max: F *= self.F_max / np.linalg.norm(F)
        if self.process_noise_std > 0: F += np.random.normal(0, self.process_noise_std, 3)

        try:
            p.applyExternalForce(self.bodyId, -1, F.tolist(), [0,0,0], p.WORLD_FRAME, self.physics_client_id)
            orn = p.getQuaternionFromEuler([self.current_roll, self.current_pitch, self.current_yaw])
            cp, _ = p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
            lv, _ = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
            p.resetBasePositionAndOrientation(self.bodyId, cp, orn, physicsClientId=self.physics_client_id)
            p.resetBaseVelocity(self.bodyId, lv, [0,0,0], physicsClientId=self.physics_client_id)
        except p.error: return

        if self.lidar_sensor:
            self.last_lidar_points = self.get_lidar_data(np.array(cp), self.current_roll, self.current_yaw, self.current_pitch)

        self._log_state(pos, vel)
        self._sim_time += self.dt

    def apply_physics(self, *args, **kwargs): pass