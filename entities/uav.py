import os
import csv
import pybullet as p
import numpy as np

from entities.agent import Agent
from Control.PID import PIDController
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF
from Control.Path_planning import RRTStarPlanner 


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

        # ----------- Waypoints / cible -----------
        wp_list = self.config.get("waypoints", None)
        if wp_list is not None and len(wp_list) > 0:
            self.waypoints = [np.array(w, dtype=float) for w in wp_list]
        else:
            default_target = self.config.get("setpoint", start_pos)
            self.waypoints = [np.array(default_target, dtype=float)]

        self.current_wp_idx = 0
        # ... (Config du planner RRTStarPlanner comme avant) ...
        self.planner = RRTStarPlanner(self.config["planner_config"])
        
        # --- NOUVEAU ---
        # Liste temporaire pour les points de passage du path planning
        self.local_checkpoints = [] 
        
        self.last_replan_time = 0.0
        self.replan_interval = 0.5
        self.blocked_until_time = 0.0

        # Pour log / debug capteurs
        self.last_gps_meas = None   # (pos, vel)
        self.last_imu_meas = None   # (specific_force_body, gyro_body)
        
        
        self.detected_obstacles_positions = []
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
            # On passe tout le dictionnaire lidar_cfg
            self.lidar_sensor = LidarSensor(lidar_cfg)
            print(f"[{self.name}] Lidar activé")
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
        """
        Retourne la cible immédiate pour le PID.
        PRIORITÉ :
        1. Le prochain checkpoint local (si une évitement est en cours).
        2. Sinon, le waypoint de mission actuel.
        """
        # PRIORITÉ 1 : Suivre le chemin d'évitement (RRT*)
        if len(self.local_checkpoints) > 0:
            # On vise le premier point de la liste d'évitement
            return self.local_checkpoints[0]
            
        # PRIORITÉ 2 : Suivre la mission principale
        if self.current_wp_idx >= len(self.waypoints):
            return self.waypoints[-1]
            
        return self.waypoints[self.current_wp_idx]

    # ------------------------------------------------------------------
    def _update_waypoint_if_reached(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """
        Vérifie si on a atteint la cible active et passe à la suivante.
        """
        target = self._get_active_target()
        dist = float(np.linalg.norm(target - pos))
        
        # CAS 1 : On chasse des Checkpoints (Évitement)
        if len(self.local_checkpoints) > 0:
            # Tolérance large (ex: 40cm) pour fluidifier le vol (Fly-through)
            # On ne s'arrête pas, on passe juste à proximité
            checkpoint_tolerance = 0.4 
            
            if dist < checkpoint_tolerance:
                # Hop, on a passé ce point, on l'enlève pour viser le suivant
                popped = self.local_checkpoints.pop(0)
                # (Optionnel) print(f"Checkpt atteint. Reste: {len(self.local_checkpoints)}")
                
        # CAS 2 : On chasse des Waypoints (Mission)
        else:
            speed = float(np.linalg.norm(vel))
            # Ici on est précis : on veut être proche ET lent
            if dist < self.pos_tolerance and speed < self.vel_tolerance:
                if self.current_wp_idx < len(self.waypoints) - 1:
                    self.current_wp_idx += 1
                    print(f"[{self.name}] Waypoint validé ! Cap sur le suivant.")
                    # Sécurité : on vide les checkpoints locaux quand on change de mission
                    self.local_checkpoints = []
    # ------------------------------------------------------------------
    

    # ------------------------------------------------------------------
    def get_lidar_data(self, sensor_position: np.ndarray, roll: float, yaw: float, pitch: float):
        """
        Mesure Lidar (si activé). Ne modifie pas le contrôle.
        Retourne une liste de points (np.ndarray) ou [] si désactivé.
        """
        if self.lidar_sensor is None:
            return []
        return self.lidar_sensor.measure(sensor_position, roll, yaw, pitch,self.detected_obstacles_positions)
    
    # -------------------------------------------------------------------
    def _is_path_valid(self, path, obstacles):
        """
        Vérifie si le chemin est sûr avec une marge CRITIQUE (plus faible que la marge de planning).
        Cela évite de rejeter un plan valide juste parce qu'on est un peu près d'un obstacle.
        """
        if not path or not obstacles:
            return True
            
        # Marge critique : on accepte d'être dans la zone de confort (1.5m)
        # tant qu'on est pas en danger immédiat (< 0.6m).
        critical_dist = 0.6
        
        # On reconstitue le chemin complet : Position Actuelle -> Checkpoints
        current_pos = np.array(self.get_ground_truth_state()["pos"])
        full_path = [current_pos] + [p for p in path]
        
        # Vérification segment par segment
        for i in range(len(full_path) - 1):
            if not self._check_segment_safe(full_path[i], full_path[i+1], obstacles, critical_dist):
                return False
        return True

    def _check_segment_safe(self, p_start, p_end, obstacles, dist_limit):
        # Vérification par échantillonnage le long du segment
        seg_len = np.linalg.norm(p_end - p_start)
        if seg_len < 1e-3: return True
        
        # On vérifie tous les 20cm
        steps = int(np.ceil(seg_len / 0.2)) 
        for i in range(steps + 1):
            t = i / steps
            pt = p_start + (p_end - p_start) * t
            
            # Distance minimale aux obstacles
            for obs in obstacles:
                if np.linalg.norm(pt - obs) < dist_limit:
                    return False
        return True

    # ------------------------------------------------------------------
    def _log_state(self, pos_true, vel_true):
        """Enregistre dans le CSV : vérité, GPS, EKF, Lidar (si logging activé)."""
        if not self.logging_enabled or self.log_file_path is None:
            return

        # Par défaut, NaN si mesure absente
        x_gps = y_gps = z_gps = np.nan
        vx_gps = vy_gps = vz_gps = np.nan
        x_ekf = y_ekf = z_ekf = np.nan
        vx_ekf = vy_ekf = vz_ekf = np.nan
        lidar_count = np.nan

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
        }

        file_exists = os.path.isfile(self.log_file_path)
        fieldnames = list(row.keys())

        with open(self.log_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def replan_path(self, current_pos, current_vel, obstacles):
        """
        Calcule un chemin avec Fallback :
        1. Essaie depuis la position future (Lookahead) pour la fluidité.
        2. Si échec, essaie depuis la position actuelle pour la sécurité.
        """
        if not obstacles: return False
        
        # Vérifications temporelles
        if self._sim_time < self.blocked_until_time: return False
        if self._sim_time - self.last_replan_time < self.replan_interval: return False

        current_mission_goal = self.waypoints[self.current_wp_idx]
        
        # Conversion types
        pos_np = np.array(current_pos, dtype=float)
        vel_np = np.array(current_vel, dtype=float)
        speed = np.linalg.norm(vel_np)

        # --- TENTATIVE 1 : Projection Anticipée (Lookahead) ---
        path = None
        if speed > 0.1:
            start_node_future = pos_np + vel_np * 0.5 # 0.5s dans le futur
            path = self.planner.plan(start_node_future, current_mission_goal, obstacles, smooth=True)

        # --- TENTATIVE 2 : Position Actuelle (Fallback) ---
        # Si la tentative 1 a échoué (ou si on est immobile), on essaie depuis ici.
        if path is None:
            # print(f"[{self.name}] Lookahead échoué, essai depuis position actuelle...")
            path = self.planner.plan(pos_np, current_mission_goal, obstacles, smooth=True)

        # Résultat
        if path is not None and len(path) > 0:
            self.local_checkpoints = [np.array(p) for p in path]
            print(f"[{self.name}] SUCCÈS : Nouveau chemin généré ({len(path)} pts).")
            self.last_replan_time = self._sim_time
            self.blocked_until_time = 0.0
            return True 
        else:
            # Échec total -> Blocage temporaire
            print(f"[{self.name}] ÉCHEC : Aucun chemin trouvé. Pause calcul 2.0s.")
            self.last_replan_time = self._sim_time
            self.blocked_until_time = self._sim_time + 2.0 
            return False

    # ------------------------------------------------------------------
    def _check_mission_waypoint_safety(self, obstacles):
        """
        Vérifie si le waypoint de mission actuel est situé dans la zone de sécurité d'un obstacle.
        Si oui, passe au waypoint suivant jusqu'à en trouver un de sûr.
        """
        while self.current_wp_idx < len(self.waypoints):
            target = self.waypoints[self.current_wp_idx]
            
            # Utilisation de la marge de sécurité complète du planner (ex: 0.8m)
            is_safe = self.planner._is_collision_free(
                pos=target,
                obstacles=obstacles,
                custom_margin=self.planner.safe_distance
            )
            
            if is_safe:
                return True
            else:
                # Si le waypoint est en collision, on le saute
                print(f"[{self.name}] AVERTISSEMENT : Waypoint {self.current_wp_idx} est dans un obstacle (marge de sécurité). PASSAGE AU SUIVANT.")
                self.current_wp_idx += 1

        # Si nous avons parcouru toute la liste sans trouver de waypoint sûr
        if self.current_wp_idx >= len(self.waypoints):
            print(f"[{self.name}] ERREUR : Tous les waypoints sont bloqués ou atteints.")
            self.current_wp_idx = len(self.waypoints) - 1 # Se positionne sur le dernier point
            return False


    # ------------------------------------------------------------------
    def think_and_act(self, setpoint: np.ndarray | None = None):
        # ... (Début inchangé : initialisation, capteurs, state...)
        if not p.isConnected(self.physics_client_id): return
        if setpoint: self.set_target_pos(setpoint)
        try:
            state = self.get_ground_truth_state()
        except p.error: return
        pos, vel, orn_q, ang_vel = np.array(state["pos"]), np.array(state["vel"]), state["orn_q"], state["ang_vel"]
        if self.gps and self.ekf:
            m_pos, m_vel = self.gps.measure(pos, vel)
            self.last_gps_meas, self.last_ekf_state = (m_pos, m_vel), self.ekf.step(np.asarray(m_pos), np.asarray(m_vel))

        # --- GESTION OBSTACLES ---
        raw_obstacles = self.lidar_sensor.measure(pos, 0.0, self.current_yaw, 0.0)
        self.detected_obstacles_positions = [p for p in raw_obstacles if p[2]>0.2 and np.linalg.norm(p-pos)>0.5]
        
        emergency_hover = self._sim_time < self.blocked_until_time
        
        if len(self.detected_obstacles_positions) > 0 and self.current_wp_idx < len(self.waypoints):
            self._check_mission_waypoint_safety(self.detected_obstacles_positions)
        
        # Si la vérification a mis fin à la mission (plus de waypoints), on s'arrête
        if self.current_wp_idx >= len(self.waypoints):
            # Le drone va naturellement s'arrêter dans la section PID
            pass

        if len(self.detected_obstacles_positions) > 0 and not emergency_hover:
            
            # 1. J'ai un plan -> Est-il valide ?
            if len(self.local_checkpoints) > 0:
                if not self._is_path_valid(self.local_checkpoints, self.detected_obstacles_positions):
                    print(f"[{self.name}] Chemin coupé par un obstacle !")
                    self.last_replan_time = 0.0 
                    self.replan_path(pos, vel, self.detected_obstacles_positions)
            
            # 2. Pas de plan -> Route directe bloquée ?
            else:
                target_mission = self.waypoints[self.current_wp_idx]
                if not self.planner._is_path_collision_free(pos, target_mission, self.detected_obstacles_positions, custom_margin=0.5):
                    # On ne print que si on n'est pas throttled (pour éviter le spam)
                    if self._sim_time - self.last_replan_time >= self.replan_interval:
                        print(f"[{self.name}] Route directe bloquée. Tentative d'évitement...")
                        self.replan_path(pos, vel, self.detected_obstacles_positions)

        # --- CONTRÔLE (Inchangé) ---
        if self.imu: self.imu.measure(pos, orn_q, vel, ang_vel, self.dt)
        self._update_waypoint_if_reached(pos, vel)
        target = self._get_active_target()
        e_pos, dist = target - pos, float(np.linalg.norm(target - pos))
        
        dir_xy = np.array([e_pos[0], e_pos[1]])
        yaw_des = float(np.arctan2(dir_xy[1], dir_xy[0])) if np.linalg.norm(dir_xy) > 0.1 else self.current_yaw
        yaw_err = np.arctan2(np.sin(yaw_des - self.current_yaw), np.cos(yaw_des - self.current_yaw))
        self.current_yaw += np.clip(self.yaw_align_gain * yaw_err, -self.yaw_rate_max, self.yaw_rate_max) * self.dt
        
        cy, sy = np.cos(self.current_yaw), np.sin(self.current_yaw)
        v_fwd, v_side = cy*vel[0]+sy*vel[1], -sy*vel[0]+cy*vel[1]
        self.current_pitch = (1-self.tilt_smoothing)*self.current_pitch + self.tilt_smoothing*np.clip(self.tilt_gain*v_fwd, -self.pitch_max, self.pitch_max)
        self.current_roll = (1-self.tilt_smoothing)*self.current_roll + self.tilt_smoothing*np.clip(self.tilt_gain*v_side, -self.roll_max, self.roll_max)

        if emergency_hover:
            self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()
            self.pid_x.setpoint, self.pid_y.setpoint, self.pid_z.setpoint = pos[0], pos[1], pos[2]
            a_total = np.array([0.0, 0.0, self.g])
        else:
            stop_cond = (self.current_wp_idx == len(self.waypoints)-1 and len(self.local_checkpoints)==0 and dist<self.pos_tolerance)
            if stop_cond:
                self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()
                a_total = np.array([0.0, 0.0, self.g])
            else:
                ax = np.clip(self.pid_x.compute(e_pos[0], self.dt), -self.acc_limit_xy, self.acc_limit_xy)
                ay = np.clip(self.pid_y.compute(e_pos[1], self.dt), -self.acc_limit_xy, self.acc_limit_xy)
                az = np.clip(self.pid_z.compute(e_pos[2], self.dt), -self.acc_limit_z, self.acc_limit_z)
                if abs(yaw_err) > self.yaw_align_threshold and len(self.local_checkpoints) == 0: ax = ay = 0.0
                a_total = np.array([ax, ay, az + self.g])

        F = self.mass * a_total
        if np.linalg.norm(F) > self.F_max: F *= self.F_max / np.linalg.norm(F)
        p.applyExternalForce(self.bodyId, -1, F.tolist(), [0,0,0], p.WORLD_FRAME, self.physics_client_id)
        
        orn_q_cmd = p.getQuaternionFromEuler([self.current_roll, self.current_pitch, self.current_yaw])
        cur_p, _ = p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
        lin_v, _ = p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(self.bodyId, cur_p, orn_q_cmd, physicsClientId=self.physics_client_id)
        p.resetBaseVelocity(self.bodyId, lin_v, [0,0,0], physicsClientId=self.physics_client_id)
        self._log_state(pos, vel); self._sim_time += self.dt
    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Toute la logique de contrôle est dans think_and_act(),
        on garde cette méthode pour compatibilité.
        """
        pass
