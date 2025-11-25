import os
import csv
import pybullet as p
import numpy as np

from entities.agent import Agent
from Control.PID import PIDController
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF


class UAV(Agent):
    """
    Drone contrôlé en position par PID (x, y, z) avec orientation "raisonnablement réaliste":
      - PID sur x, y, z -> accélérations désirées dans le repère monde
      - force = m * (a_cmd + gravité) appliquée au centre de masse
      - le drone tourne autour de z pour que son axe x pointe vers la cible
      - il se penche vers l'avant (pitch) lorsqu'il avance, et reste droit en vol stationnaire
      - plusieurs waypoints possibles (liste de positions à suivre)
      - capteurs GPS / IMU optionnels
      - EKF optionnel (analyse / log uniquement, NE MODIFIE PAS le contrôle)
      - Lidar optionnel (analyse / log uniquement)
      - Logging optionnel vers un fichier CSV
      - Process noise optionnel sur la force (modélise des perturbations type vent)
    """

    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = self.config.get("name", "unnamed_uav")

        # ----------- EKF / Lidar / Logging -----------
        self.ekf: GPSEKF | None = None
        self.last_ekf_state = None  # (pos_est, vel_est)

        self.lidar_sensor: LidarSensor | None = None
        self.last_lidar_points = None  # liste de np.ndarray

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
                os.remove(self.log_file_path)
            print(f"[{self.name}] Logging activé -> {self.log_file_path}")
        else:
            print(f"[{self.name}] Logging désactivé")

        # ----------- Pose initiale -----------
        urdf_path = self.config["urdf_path"]

        start_pos = self.config.get("start_pos", [0.0, 0.0, 0.2])
        start_pos = [float(v) for v in start_pos]

        start_orn_euler = self.config.get("start_orn_euler", [0.0, 0.0, 0.0])
        start_orn_euler = [float(a) for a in start_orn_euler]
        start_orn_q = p.getQuaternionFromEuler(start_orn_euler)

        # Yaw initial
        self.current_yaw = float(start_orn_euler[2])
        self.current_roll = 0.0
        self.current_pitch = 0.0

        # Charge l'URDF via la classe Agent
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

        # Masse totale = base + tous les liens
        num_joints = p.getNumJoints(self.bodyId, physicsClientId=self.physics_client_id)
        total_mass = p.getDynamicsInfo(
            self.bodyId, -1, physicsClientId=self.physics_client_id
        )[0] or 0.0

        for j in range(num_joints):
            mj = p.getDynamicsInfo(
                self.bodyId, j, physicsClientId=self.physics_client_id
            )[0]
            if mj is not None:
                total_mass += mj

        if total_mass <= 0.0:
            total_mass = 0.03  # fallback si URDF bizarre

        self.mass = float(total_mass)

        # ----------- PID à partir du YAML -----------
        components_cfg = self.config.get("components", {})

        def get_pid_cfg(name: str, fallback: dict = None) -> dict:
            if name in components_cfg:
                return components_cfg[name]
            if fallback is not None:
                return fallback
            # config par défaut si rien n'est défini
            return {
                "gains": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
                "windup": 0.0,
            }

        cfg_z = get_pid_cfg("controller_z")
        cfg_x = get_pid_cfg("controller_x", fallback=cfg_z)
        cfg_y = get_pid_cfg("controller_y", fallback=cfg_z)

        # Limites d'accélération (x,y,z)
        self.acc_limit_xy = float(self.config.get("acc_limit_xy", 5.0))  # m/s^2
        self.acc_limit_z = float(self.config.get("acc_limit_z", 5.0))    # m/s^2

        self.pid_x = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_x)
        self.pid_y = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_y)
        self.pid_z = PIDController(self.acc_limit_z, self.acc_limit_z, cfg_z)

        # Limite de la force totale (en fonction du poids)
        F_max_factor = float(self.config.get("F_max_factor", 2.5))
        self.F_max = F_max_factor * self.mass * self.g

        # Zone morte autour de la cible (pour éviter les tremblements)
        self.pos_tolerance = float(self.config.get("pos_tolerance", 0.05))  # 5 cm
        self.vel_tolerance = float(self.config.get("vel_tolerance", 0.05))  # 5 cm/s

        # ----------- Paramètres d'orientation "réaliste" -----------
        # Alignement de l'axe x avec la direction cible
        self.yaw_align_gain = float(self.config.get("yaw_align_gain", 4.0))
        self.yaw_rate_max = float(
            self.config.get("yaw_rate_max_deg", 120.0)
        ) * np.pi / 180.0

        # Seuil d'alignement : tant que |erreur_yaw| > seuil, on n'avance pas en x,y
        self.yaw_align_threshold = float(
            self.config.get("yaw_align_threshold_deg", 10.0)
        ) * np.pi / 180.0

        # Tilt visuel en fonction de la vitesse dans le repère drone
        self.tilt_gain = float(self.config.get("tilt_gain", 0.4))
        self.pitch_max = float(
            self.config.get("pitch_max_deg", 35.0)
        ) * np.pi / 180.0
        self.roll_max = float(
            self.config.get("roll_max_deg", 35.0)
        ) * np.pi / 180.0

        # Filtre pour lisser la rotation (0 -> très lissé, 1 -> pas lissé)
        self.tilt_smoothing = float(self.config.get("tilt_smoothing", 0.5))
        self.tilt_smoothing = np.clip(self.tilt_smoothing, 0.0, 1.0)

        # ----------- Process noise (sur la force) -----------
        # Ecart-type du bruit de processus appliqué sur la force (N).
        # clamp à >= 0.0 pour être safe.
        self.process_noise_std = max(0.0, float(self.config.get("process_noise_std", 0.0)))

        # ----------- Waypoints / cible -----------
        wp_list = self.config.get("waypoints", None)
        if wp_list is not None and len(wp_list) > 0:
            self.waypoints = [np.array(w, dtype=float) for w in wp_list]
        else:
            default_target = self.config.get("setpoint", start_pos)
            self.waypoints = [np.array(default_target, dtype=float)]

        self.current_wp_idx = 0

        # Pour log / debug capteurs
        self.last_gps_meas = None   # (pos, vel)
        self.last_imu_meas = None   # (specific_force_body, gyro_body)

        # Pour log / debug contrôle
        # accélération de commande PID (sans gravité) au dernier pas
        self.last_control_accel = np.zeros(3, dtype=float)  # [ax_cmd, ay_cmd, az_cmd]
        # attitude commandée (roll, pitch, yaw) au dernier pas
        self.last_attitude_cmd = np.array(
            [self.current_roll, self.current_pitch, self.current_yaw], dtype=float
        )
        # info waypoint
        self.last_wp_index = 0
        self.last_wp_distance = np.nan

        print(
            f"UAV '{self.name}' chargé, bodyId={self.bodyId}, "
            f"masse_totale={self.mass:.4f} kg, {len(self.waypoints)} waypoint(s)"
        )

    # ------------------------------------------------------------------
    def _initialize_components(self):
        """Initialise les capteurs + EKF + Lidar si définis dans le YAML."""
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

        # EKF (optionnel, analyse / log uniquement)
        ekf_cfg = sensors_cfg.get("ekf", {})
        if self.gps is not None and ekf_cfg.get("enabled", False):
            q_pos = float(ekf_cfg.get("q_pos", 0.05))
            q_vel = float(ekf_cfg.get("q_vel", 0.10))
            r_pos = float(ekf_cfg.get("r_pos", sensors_cfg.get("gps", {}).get("position_noise_std", 0.1)))
            r_vel = float(ekf_cfg.get("r_vel", sensors_cfg.get("gps", {}).get("velocity_noise_std", 0.05)))

            self.ekf = GPSEKF(
                dt=self.dt,
                q_pos=q_pos,
                q_vel=q_vel,
                r_pos=r_pos,
                r_vel=r_vel,
            )
            self.components["ekf"] = self.ekf
            print(
                f"[{self.name}] EKF initialisé (analyse uniquement) "
                f"(q_pos={q_pos}, q_vel={q_vel}, r_pos={r_pos}, r_vel={r_vel})"
            )
        else:
            self.ekf = None
            print(f"[{self.name}] EKF désactivé")

        # Lidar
        lidar_cfg = sensors_cfg.get("lidar", {})
        if lidar_cfg.get("enabled", False):
            max_dist = float(lidar_cfg.get("max_distance", 10.0))
            ang_res = float(lidar_cfg.get("angle_resolution", 1.0))
            self.lidar_sensor = LidarSensor(max_dist, ang_res)
            self.components["lidar"] = self.lidar_sensor
            print(
                f"[{self.name}] Lidar activé "
                f"(max_distance={max_dist}, angle_resolution={ang_res})"
            )
        else:
            self.lidar_sensor = None
            print(f"[{self.name}] Lidar désactivé")

        # Reset états capteurs
        self.last_gps_meas = None
        self.last_imu_meas = None
        self.last_ekf_state = None
        self.last_lidar_points = None

    def _create_body_frame_axes(self, axis_length: float = 0.3):
        """
        Affiche le repère local attaché au drone :
          - X : rouge (avant)
          - Y : vert (latéral)
          - Z : bleu (vertical)
        Les lignes sont attachées à la base (linkIndex = -1).
        """
        p.addUserDebugLine(
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # X rouge
            parentObjectUniqueId=self.bodyId,
            parentLinkIndex=-1,
            physicsClientId=self.physics_client_id,
        )

        p.addUserDebugLine(
            [0.0, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 1.0, 0.0],  # Y vert
            parentObjectUniqueId=self.bodyId,
            parentLinkIndex=-1,
            physicsClientId=self.physics_client_id,
        )

        p.addUserDebugLine(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, axis_length],
            [0.0, 0.0, 1.0],  # Z bleu
            parentObjectUniqueId=self.bodyId,
            parentLinkIndex=-1,
            physicsClientId=self.physics_client_id,
        )

    # ------------------------------------------------------------------
    def set_target_pos(self, target):
        """Compatibilité: remplace la liste de waypoints par une seule cible."""
        self.waypoints = [np.array(target, dtype=float)]
        self.current_wp_idx = 0

    # ------------------------------------------------------------------
    def set_waypoints(self, waypoints):
        """Définir une liste de waypoints à suivre dans l'ordre."""
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        if len(self.waypoints) == 0:
            raise ValueError("set_waypoints() requiert au moins un point")
        self.current_wp_idx = 0

    # ------------------------------------------------------------------
    def _get_active_target(self) -> np.ndarray:
        """Retourne le waypoint courant."""
        if self.current_wp_idx >= len(self.waypoints):
            return self.waypoints[-1]
        return self.waypoints[self.current_wp_idx]

    # ------------------------------------------------------------------
    def _update_waypoint_if_reached(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Passe au waypoint suivant si on est proche et presque immobile."""
        target = self._get_active_target()
        e = target - pos
        dist = float(np.linalg.norm(e))
        speed = float(np.linalg.norm(vel))

        if dist < self.pos_tolerance and speed < self.vel_tolerance:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1

    # ------------------------------------------------------------------
    def get_lidar_data(self, sensor_position: np.ndarray, roll: float, yaw: float, pitch: float):
        """
        Mesure Lidar (si activé). Ne modifie pas le contrôle.
        Retourne une liste de points (np.ndarray) ou [] si désactivé.
        """
        if self.lidar_sensor is None:
            return []
        return self.lidar_sensor.measure(sensor_position, roll, yaw, pitch)

    # ------------------------------------------------------------------
    def _log_state(self, pos_true, vel_true):
        """Enregistre dans le CSV : vérité, GPS, EKF, Lidar, commande PID, attitude, waypoint."""
        if not self.logging_enabled or self.log_file_path is None:
            return

        # Par défaut, NaN si mesure absente
        x_gps = y_gps = z_gps = np.nan
        vx_gps = vy_gps = vz_gps = np.nan
        x_ekf = y_ekf = z_ekf = np.nan
        vx_ekf = vy_ekf = vz_ekf = np.nan
        lidar_count = np.nan

        # Commande PID (accélération) + attitude + waypoint
        ax_cmd = ay_cmd = az_cmd = np.nan
        roll = pitch = yaw = np.nan
        wp_idx = self.last_wp_index
        dist_wp = self.last_wp_distance

        gps_enabled = 1 if self.gps is not None else 0
        ekf_enabled = 1 if self.ekf is not None else 0
        lidar_enabled = 1 if self.lidar_sensor is not None else 0

        if self.last_gps_meas is not None:
            gps_pos, gps_vel = self.last_gps_meas
            gps_pos = np.asarray(gps_pos, dtype=float)
            gps_vel = np.asarray(gps_vel, dtype=float)
            x_gps, y_gps, z_gps = gps_pos.tolist()
            vx_gps, vy_gps, vz_gps = gps_vel.tolist()

        if self.last_ekf_state is not None:
            ekf_pos, ekf_vel = self.last_ekf_state
            ekf_pos = np.asarray(ekf_pos, dtype=float)
            ekf_vel = np.asarray(ekf_vel, dtype=float)
            x_ekf, y_ekf, z_ekf = ekf_pos.tolist()
            vx_ekf, vy_ekf, vz_ekf = ekf_vel.tolist()

        if self.last_lidar_points is not None:
            lidar_count = float(len(self.last_lidar_points))

        if self.last_control_accel is not None:
            ax_cmd, ay_cmd, az_cmd = np.asarray(self.last_control_accel, dtype=float).tolist()

        if self.last_attitude_cmd is not None:
            roll, pitch, yaw = np.asarray(self.last_attitude_cmd, dtype=float).tolist()

        row = {
            "t": self._sim_time,
            "x_true": float(pos_true[0]),
            "y_true": float(pos_true[1]),
            "z_true": float(pos_true[2]),
            "vx_true": float(vel_true[0]),
            "vy_true": float(vel_true[1]),
            "vz_true": float(vel_true[2]),
            "x_gps": x_gps,
            "y_gps": y_gps,
            "z_gps": z_gps,
            "vx_gps": vx_gps,
            "vy_gps": vy_gps,
            "vz_gps": vz_gps,
            "x_ekf": x_ekf,
            "y_ekf": y_ekf,
            "z_ekf": z_ekf,
            "vx_ekf": vx_ekf,
            "vy_ekf": vy_ekf,
            "vz_ekf": vz_ekf,
            "gps_enabled": gps_enabled,
            "ekf_enabled": ekf_enabled,
            "lidar_enabled": lidar_enabled,
            "lidar_count": lidar_count,
            "ax_cmd": ax_cmd,
            "ay_cmd": ay_cmd,
            "az_cmd": az_cmd,
            "roll_cmd": roll,
            "pitch_cmd": pitch,
            "yaw_cmd": yaw,
            "wp_index": wp_idx,
            "dist_to_wp": dist_wp,
        }

        file_exists = os.path.isfile(self.log_file_path)
        fieldnames = list(row.keys())

        with open(self.log_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    def think_and_act(self, setpoint: np.ndarray | None = None):
        """
        Contrôle de position avec orientation réaliste (DYNAMIQUE ORIGINALE) :
          - mêmes PID, même yaw_align, même tilt
          - contrôle basé EXCLUSIVEMENT sur la vérité terrain (pos / vel GT)
          - GPS / EKF / Lidar uniquement pour mesure + log
          - Process noise optionnel ajouté sur la force
          - Logging détaillé des commandes PID, attitude et waypoint
        """
        # Si le serveur n'est plus connecté, on ne fait rien
        if not p.isConnected(self.physics_client_id):
            return

        # Compatibilité: si on passe un setpoint, remplace les waypoints
        if setpoint is not None:
            self.set_target_pos(setpoint)

        # 1. Vérité terrain
        try:
            state = self.get_ground_truth_state()
        except p.error:
            return

        pos = state["pos"]      # [x, y, z] monde
        vel = state["vel"]      # [vx, vy, vz] monde
        orn_q = state["orn_q"]
        ang_vel = state["ang_vel"]

        # 2. Mesures GPS / EKF (NE MODIFIENT PAS LA COMMANDE)
        if self.gps is not None:
            meas_pos, meas_vel = self.gps.measure(pos, vel)
            self.last_gps_meas = (meas_pos, meas_vel)

            if self.ekf is not None:
                est_pos, est_vel = self.ekf.step(
                    np.asarray(meas_pos, dtype=float),
                    np.asarray(meas_vel, dtype=float),
                )
                self.last_ekf_state = (est_pos, est_vel)
        else:
            self.last_gps_meas = None
            self.last_ekf_state = None

        # 3. Mesures IMU (pour info / log éventuel)
        if self.imu is not None:
            specific_force_body, gyro_body = self.imu.measure(
                ground_truth_position=pos,
                ground_truth_orientation=orn_q,
                ground_truth_velocity=vel,
                ground_truth_ang_vel=ang_vel,
                dt=self.dt,
            )
            self.last_imu_meas = (specific_force_body, gyro_body)
        else:
            self.last_imu_meas = None

        # 4. Met à jour le waypoint actif (sur la base de la vérité terrain)
        self._update_waypoint_if_reached(pos, vel)
        target = self._get_active_target()

        # Erreurs de position
        e_pos = target - pos
        dist = float(np.linalg.norm(e_pos))
        speed3d = float(np.linalg.norm(vel))

        # Sauvegarde pour le log
        self.last_wp_index = self.current_wp_idx
        self.last_wp_distance = dist

        # 5. Calcul du yaw désiré: axe x pointe vers le waypoint (projection au sol)
        dir_world = e_pos.copy()
        dir_world[2] = 0.0
        norm_dir = float(np.linalg.norm(dir_world))
        if norm_dir < 1e-6:
            yaw_des = self.current_yaw
        else:
            yaw_des = float(np.arctan2(dir_world[1], dir_world[0]))

        # Erreur de yaw dans [-pi, pi]
        yaw_err = np.arctan2(
            np.sin(yaw_des - self.current_yaw),
            np.cos(yaw_des - self.current_yaw),
        )

        # 6. Mise à jour du yaw (1er ordre, saturé en vitesse angulaire)
        yaw_rate_cmd = self.yaw_align_gain * yaw_err
        yaw_rate_cmd = float(
            np.clip(yaw_rate_cmd, -self.yaw_rate_max, self.yaw_rate_max)
        )
        self.current_yaw += yaw_rate_cmd * self.dt

        # 7. Vitesse dans le repère drone (pour le tilt)
        cy = np.cos(self.current_yaw)
        sy = np.sin(self.current_yaw)

        v_forward = cy * vel[0] + sy * vel[1]
        v_side = -sy * vel[0] + cy * vel[1]

        # Tilt désiré en fonction de la vitesse
        pitch_des = self.tilt_gain * v_forward
        roll_des = self.tilt_gain * v_side

        pitch_des = float(np.clip(pitch_des, -self.pitch_max, self.pitch_max))
        roll_des = float(np.clip(roll_des, -self.roll_max, self.roll_max))

        alpha = self.tilt_smoothing
        self.current_pitch = (1.0 - alpha) * self.current_pitch + alpha * pitch_des
        self.current_roll = (1.0 - alpha) * self.current_roll + alpha * roll_des

        # Sauvegarde attitude commandée pour logs
        self.last_attitude_cmd = np.array(
            [self.current_roll, self.current_pitch, self.current_yaw], dtype=float
        )

        # 8. Contrôle de position (PID x,y,z) -> accélérations désirées
        #    *** TOUJOURS basé sur la vérité terrain ***
        if dist < self.pos_tolerance and speed3d < self.vel_tolerance:
            # Arrivé et quasi immobile -> reset PID, juste compensation gravité
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            # aucune accélération de commande, juste la gravité compensée
            ax_cmd = ay_cmd = az_cmd = 0.0
            self.last_control_accel = np.array([0.0, 0.0, 0.0], dtype=float)
            a_total = np.array([0.0, 0.0, self.g], dtype=float)
        else:
            ex, ey, ez = e_pos

            ax_cmd = self.pid_x.compute(float(ex), self.dt)
            ay_cmd = self.pid_y.compute(float(ey), self.dt)
            az_cmd = self.pid_z.compute(float(ez), self.dt)

            # Saturation des accélérations
            ax_cmd = float(np.clip(ax_cmd, -self.acc_limit_xy, self.acc_limit_xy))
            ay_cmd = float(np.clip(ay_cmd, -self.acc_limit_xy, self.acc_limit_xy))
            az_cmd = float(np.clip(az_cmd, -self.acc_limit_z, self.acc_limit_z))

            # Tant que l'axe x n'est pas bien aligné, on bloque x,y (comme avant)
            if abs(yaw_err) > self.yaw_align_threshold:
                ax_cmd = 0.0
                ay_cmd = 0.0

            # Sauvegarde de la commande PID (sans gravité) pour logs
            self.last_control_accel = np.array([ax_cmd, ay_cmd, az_cmd], dtype=float)

            # Ajout de la gravité sur Z
            a_total = np.array([ax_cmd, ay_cmd, az_cmd + self.g], dtype=float)

        # 9. Force souhaitée en repère monde
        F = self.mass * a_total

        # Saturation de la force totale
        norm_F = float(np.linalg.norm(F))
        if norm_F > self.F_max:
            F *= self.F_max / (norm_F + 1e-9)

        # 9bis. Bruit de processus sur la force (si activé)
        if self.process_noise_std > 0.0:
            noise = np.random.normal(0.0, self.process_noise_std, size=3)
            F = F + noise
            # Optionnel: re-saturation
            # norm_F = float(np.linalg.norm(F))
            # if norm_F > self.F_max:
            #     F *= self.F_max / (norm_F + 1e-9)

        # 10. Application de la force + mise à jour de l'orientation
        if not p.isConnected(self.physics_client_id):
            return

        try:
            # appliquer la force au centre de masse
            p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=-1,  # base
                forceObj=F.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client_id,
            )

            # Orientation: roll/pitch/yaw calculés ci-dessus
            orn_q_cmd = p.getQuaternionFromEuler(
                [self.current_roll, self.current_pitch, self.current_yaw]
            )

            cur_pos, _ = p.getBasePositionAndOrientation(
                self.bodyId, physicsClientId=self.physics_client_id
            )
            lin_vel, _ = p.getBaseVelocity(
                self.bodyId, physicsClientId=self.physics_client_id
            )

            p.resetBasePositionAndOrientation(
                self.bodyId,
                cur_pos,
                orn_q_cmd,
                physicsClientId=self.physics_client_id,
            )

            # On annule la vitesse angulaire pour éviter les rotations parasites
            p.resetBaseVelocity(
                self.bodyId,
                linearVelocity=lin_vel,
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.physics_client_id,
            )

        except p.error:
            return

        # 11. Mesure Lidar (après mise à jour de la pose, pour le log)
        self.last_lidar_points = self.get_lidar_data(
            np.array(cur_pos, dtype=float),
            self.current_roll,
            self.current_yaw,
            self.current_pitch,
        )

        # 12. Logging de l'état courant
        self._log_state(pos_true=pos, vel_true=vel)
        self._sim_time += self.dt

    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Toute la logique de contrôle est dans think_and_act(),
        on garde cette méthode pour compatibilité.
        """
        pass
