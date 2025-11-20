# entities/uav.py
import pybullet as p
import numpy as np

from entities.agent import Agent
from Control.PID import PIDController
from entities.sensor import GPSSensor, IMUSensor


class UAV(Agent):
    """
    Drone contrôlé en position par PID (x, y, z) avec orientation "raisonnablement réaliste":
      - PID sur x, y, z -> accélérations désirées dans le repère monde
      - force = m * (a_cmd + gravité) appliquée au centre de masse
      - le drone tourne autour de z pour que son axe x pointe vers la cible
      - il n'avance que lorsque l'axe x est suffisamment aligné
      - il se penche vers l'avant (pitch) lorsqu'il avance, et reste droit en vol stationnaire
      - plusieurs waypoints possibles (liste de positions à suivre)
      - capteurs GPS / IMU optionnels (mesurent mais ne modifient pas le contrôle)
    """

    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = self.config.get("name", "unnamed_uav")

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

        # ----------- Waypoints / cible -----------
        wp_list = self.config.get("waypoints", None)
        if wp_list is not None and len(wp_list) > 0:
            self.waypoints = [np.array(w, dtype=float) for w in wp_list]
        else:
            default_target = self.config.get("setpoint", start_pos)
            self.waypoints = [np.array(default_target, dtype=float)]

        self.current_wp_idx = 0

        print(
            f"UAV '{self.name}' chargé, bodyId={self.bodyId}, "
            f"masse_totale={self.mass:.4f} kg, {len(self.waypoints)} waypoint(s)"
        )

    # ------------------------------------------------------------------
    def _initialize_components(self):
        """Initialise les capteurs si définis dans le YAML."""
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

        # Pour log / debug
        self.last_gps_meas = None   # (pos, vel)
        self.last_imu_meas = None   # (specific_force_body, gyro_body)

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
    def think_and_act(self, setpoint: np.ndarray | None = None):
        """
        Contrôle de position avec orientation réaliste:
          - plusieurs waypoints possibles (self.waypoints)
          - yaw pour aligner l'axe x vers le waypoint courant
          - tant que l'axe x n'est pas aligné, pas d'accélération en x,y
          - tilt (roll/pitch) en fonction de la vitesse dans le repère drone
          - GPS/IMU mesurés mais NON utilisés pour le contrôle
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

        # 2. Mesures capteurs (pour info / logging)
        if self.gps is not None:
            meas_pos, meas_vel = self.gps.measure(pos, vel)
            self.last_gps_meas = (meas_pos, meas_vel)
        else:
            self.last_gps_meas = None

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

        # 3. Met à jour le waypoint actif si le courant est atteint
        self._update_waypoint_if_reached(pos, vel)
        target = self._get_active_target()

        # Erreurs de position
        e_pos = target - pos
        dist = float(np.linalg.norm(e_pos))
        speed3d = float(np.linalg.norm(vel))

        # 4. Calcul du yaw désiré: axe x pointe vers le waypoint (projection au sol)
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

        # 5. Mise à jour du yaw (1er ordre, saturé en vitesse angulaire)
        yaw_rate_cmd = self.yaw_align_gain * yaw_err
        yaw_rate_cmd = float(
            np.clip(yaw_rate_cmd, -self.yaw_rate_max, self.yaw_rate_max)
        )
        self.current_yaw += yaw_rate_cmd * self.dt

        # 6. Vitesse dans le repère drone (pour le tilt)
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

        # 7. Contrôle de position (PID x,y,z) -> accélérations désirées
        if dist < self.pos_tolerance and speed3d < self.vel_tolerance:
            # Arrivé et quasi immobile -> reset PID, juste compensation gravité
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
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

            # Tant que l'axe x n'est pas bien aligné, on bloque x,y
            if abs(yaw_err) > self.yaw_align_threshold:
                ax_cmd = 0.0
                ay_cmd = 0.0

            # Ajout de la gravité sur Z
            a_total = np.array([ax_cmd, ay_cmd, az_cmd + self.g], dtype=float)

        # 8. Force souhaitée en repère monde
        F = self.mass * a_total

        # Saturation de la force totale
        norm_F = float(np.linalg.norm(F))
        if norm_F > self.F_max:
            F *= self.F_max / (norm_F + 1e-9)

        # 9. Application de la force + mise à jour de l'orientation
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
            orn_q = p.getQuaternionFromEuler(
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
                orn_q,
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

    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Toute la logique de contrôle est dans think_and_act(),
        on garde cette méthode pour compatibilité.
        """
        pass
