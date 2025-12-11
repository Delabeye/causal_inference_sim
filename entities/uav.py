import os
import csv
import threading
import pybullet as p
import numpy as np

# Imports Utilitaires & Contrôle
from utilities.utilities import point_in_cube, point_in_cylinder
from entities.agent import Agent
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF
from Control.Path_planning import AStarPlanner

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class UAV(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float, known_obstacles_config: list[dict]):
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "UAV")
        
        urdf_path = config.get("urdf_path", "assets/quadrotor.urdf")
        start_pos = config.get("start_pos", [0, 0, 1.0])
        start_orn = p.getQuaternionFromEuler(config.get("start_orn_euler", [0,0,0]))
        super().__init__(urdf_path, start_pos, start_orn, physics_client_id, self.dt)
        self._sim_time = 0.0
        
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
        # On ajoute le point de départ pour stabilisation initiale
        first_wp = np.array(start_pos)+np.array([0,0,1])
        self.waypoints = [first_wp] + [np.array(w) for w in wp_list]
        self.wp_idx = 0
        
        # --- OBSTACLES & PLANNING ---
        self.obs_dic = known_obstacles_config
        # Nuage de points (Statique + Dynamique)
        self.obstacles = self._discretize_obstacles(known_obstacles_config)
        self.target_yaw_cache = 0.0
        
        a_config = {"world_bounds": {"x": [-10, 10], "y": [-10, 10], "z": [0.1, 5.0]}}
        self.planner = AStarPlanner(self.config.get("astar", a_config))
        
        # États du Planning
        self.active_path = []
        self.is_planning = False      # Drapeau : True si un thread calcule
        self.planning_thread = None   # Référence du thread
        self.replan_timer = 0         # Cooldown pour éviter de spammer
        self.Calculation_fail_count = 0

        # --- SWARM CONTROL (MODIFIÉ) ---
        self.swarm_active = False
        self.swarm_target_pos = None
        self.swarm_target_vel = None
        self.swarm_target_yaw = None  # Nouveau : pour stocker le Yaw du leader

        # --- CAPTEURS ---
        sens = config.get("sensors", {})
        self.ekf = GPSEKF(dt); self.ekf.x[:3] = start_pos
        self.gps = GPSSensor(sens.get("gps", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
        self.lidar = LidarSensor(sens.get("lidar", {}))
        
        # Fréquence Lidar (10Hz)
        lidar_freq = sens.get("lidar", {}).get("frequency", 10.0)
        self.lidar_period = 1.0 / lidar_freq
        self.last_lidar_time = -self.lidar_period
        
        # Logs
        self.logging_enabled = True
        self.log_file = os.path.join("logs", f"{self.name}.csv")
        os.makedirs("logs", exist_ok=True)
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        # Damping physique nul (on gère le drag nous-même)
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    # ----------------------------------------------------------------------
    # API SWARM (MODIFIÉE POUR YAW)
    # ----------------------------------------------------------------------
    def set_swarm_command(self, target_pos, target_vel, target_yaw=None):
        """ Appelé par le Swarm pour prendre le contrôle """
        self.swarm_active = True
        self.swarm_target_pos = np.array(target_pos)
        self.swarm_target_vel = np.array(target_vel)
        self.swarm_target_yaw = target_yaw # On stocke le Yaw reçu

    # ----------------------------------------------------------------------
    # GESTION OBSTACLES & PLANNING ASYNCHRONE
    # ----------------------------------------------------------------------

    def _discretize_obstacles(self, obstacles_config):
        points = []
        res = 0.25 
        for obs in obstacles_config:
            center = np.array(obs["center"])
            otype = obs["type"]
            # Discrétisation simplifiée
            if otype == "cube":
                l, w, h = obs["length"], obs["width"], obs["height"]
                xs = np.arange(center[0]-l/2, center[0]+l/2+res, res)
                ys = np.arange(center[1]-w/2, center[1]+w/2+res, res)
                zs = np.arange(center[2]-h/2, center[2]+h/2+res, res)
                for x in xs:
                    for y in ys:
                        for z in zs: points.append([x, y, z])
            elif otype == "sphere":
                r = obs["radius"]
                # Approximation cubique pour aller vite au démarrage
                xs = np.arange(center[0]-r, center[0]+r+res, res)
                ys = np.arange(center[1]-r, center[1]+r+res, res)
                zs = np.arange(center[2]-r, center[2]+r+res, res)
                for x in xs:
                    for y in ys:
                        for z in zs:
                            if np.linalg.norm(np.array([x,y,z])-center) <= r: points.append([x, y, z])
            elif otype == "cylinder":
                r, h = obs["radius"], obs["height"]
                xs = np.arange(center[0]-r, center[0]+r+res, res)
                ys = np.arange(center[1]-r, center[1]+r+res, res)
                zs = np.arange(center[2]-h/2, center[2]+h/2+res, res)
                for x in xs:
                    for y in ys:
                        if np.linalg.norm(np.array([x,y])-center[:2]) <= r:
                            for z in zs: points.append([x, y, z])
        return points

    def _trigger_planning(self, start_pos, target_pos, obstacles):
        """ Lance le calcul A* dans un thread séparé """
        if self.is_planning: 
            return # Déjà occupé

        self.is_planning = True
        print(f"[{self.name}] ⏳ Démarrage Thread A*...")
        
        # On passe une COPIE des obstacles pour éviter les conflits mémoire pendant que le lidar tourne
        obs_copy = list(obstacles) 
        
        self.planning_thread = threading.Thread(
            target=self._run_async_plan, 
            args=(start_pos, target_pos, obs_copy)
        )
        self.planning_thread.daemon = True # Le thread mourra si le programme quitte
        self.planning_thread.start()

    def _run_async_plan(self, start_pos, target_pos, obstacles):
        """ Code exécuté dans le Thread """
        try:
            path = self.planner.plan(start_pos, target_pos, obstacles, smooth=True)
            if path and len(path) > 0:
                self.active_path = path 
                self.Calculation_fail_count = 0
            else:
                self.Calculation_fail_count += 1
        except Exception as e:
            print(f"[{self.name}] 💥 Erreur Thread A*: {e}")
        finally:
            # On libère le drapeau à la fin (succès ou erreur)
            self.is_planning = False 

    def _analyze_target_accessibility(self, start_pos, target_pos, obs):
        """ Vérifie géométriquement si la route est libre """
        # 1. Vérif Obstacles Connus (Dictionnaires)
        for obstacle in self.obs_dic:
            if obstacle["type"] == "cube":
                if point_in_cube(target_pos, obstacle): return "INVALID"
            elif obstacle["type"] == "sphere":
                if np.linalg.norm(np.array(target_pos) - np.array(obstacle["center"])) <= obstacle["radius"]: return "INVALID"
            elif obstacle["type"] == "cylinder":
                if point_in_cylinder(target_pos, obstacle): return "INVALID"
        
        # 2. Vérif Nuage de Points (Lidar)
        # On prend un sous-ensemble si trop de points pour ne pas laguer ici aussi
        check_obs = obs if len(obs) < 2000 else obs[::5] # Optimisation
        
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
                        if dist_ortho < 0.6: # Marge sécurité
                            return "BLOCKED"
        return "CLEAR"

    # ----------------------------------------------------------------------
    # BOUCLE PRINCIPALE (240 Hz)
    # ----------------------------------------------------------------------
    def think_and_act(self):
        if not p.isConnected(self.physics_client_id): return
        
        # 1. État & Perception
        gt = self.get_ground_truth_state()
        pos, vel = gt["pos"], gt["vel"]
        orn_q, ang_vel = gt["orn_q"], gt["ang_vel"]
        rpy = p.getEulerFromQuaternion(orn_q)
        
        # Lidar (10 Hz)
        if (self._sim_time - self.last_lidar_time) >= self.lidar_period:
            self.last_lidar_time = self._sim_time
            new_pts = self.lidar.measure(pos, rpy[0], rpy[2], rpy[1])
            if len(new_pts) > 0:
                self.obstacles += new_pts # Accumulation pour la carte
        
        # ----------------------------------------------
        # LOGIQUE HAUT NIVEAU
        # ----------------------------------------------
        
        target_pos = pos # Par défaut : on reste là
        target_vel = np.zeros(3) # Par défaut : stationnaire
        
        # PRIORITÉ 1 : SWARM (Si activé)
        if self.swarm_active and self.swarm_target_pos is not None:
            target_pos = self.swarm_target_pos
            if self.swarm_target_vel is not None:
                target_vel = self.swarm_target_vel

        # PRIORITÉ 2 : En cours de planification (Thread actif)
        # -> DYNAMIQUE STATIONNAIRE (Freinage actif)
        elif self.is_planning:
            # On demande au contrôleur de freiner et maintenir l'altitude
            target_pos = pos 
            target_vel = -0.5 * vel # Freinage amorti
            
        else:
            # PRIORITÉ 3 : Normal (Navigation autonome Waypoints)
            
            # A. Gestion Waypoints Globaux
            while self.wp_idx < len(self.waypoints):
                global_target = self.waypoints[self.wp_idx]
                
                # Si on n'a pas de chemin local, on vérifie la cible globale
                if len(self.active_path) == 0:
                    status = self._analyze_target_accessibility(pos, global_target, self.obstacles)
                    
                    if status == "INVALID":
                        print(f"[{self.name}] Cible {self.wp_idx} Mur. Skip.")
                        self.wp_idx += 1
                        continue # On re-boucle
                    
                    elif status == "BLOCKED":
                        # Route bloquée -> On lance le thread
                        if self.replan_timer <= 0:
                            self._trigger_planning(pos, global_target, self.obstacles)
                            self.replan_timer = 50 # Cooldown
                        break # On sort de la boucle et on attend (mode stationnaire au prochain tour)
                
                break # Cible valide ou chemin existant

            # B. Suivi de Chemin (Path Following)
            if len(self.active_path) > 0:
                local_target = self.active_path[0]
                dist_local = np.linalg.norm(local_target - pos)
                
                if dist_local < 0.4: # Waypoint local atteint
                    self.active_path.pop(0)
                    if len(self.active_path) > 0:
                        local_target = self.active_path[0]
                    else:
                        # Fin du chemin local, on vise le global
                        if self.wp_idx < len(self.waypoints):
                            local_target = self.waypoints[self.wp_idx]
                
                target_pos = local_target
                target_vel = np.zeros(3) # On laisse le PID gérer la vitesse vers le point
                
            elif self.wp_idx < len(self.waypoints):
                # Pas de chemin, on vise direct le global (si CLEAR)
                target_pos = self.waypoints[self.wp_idx]
                dist_global = np.linalg.norm(target_pos - pos)
                
                if dist_global < 0.2:
                    print(f"[{self.name}] WP {self.wp_idx} Atteint.")
                    self.wp_idx += 1
                    # Le nouveau target sera pris au prochain cycle

        if self.replan_timer > 0: self.replan_timer -= 1

        # ----------------------------------------------
        # COMMANDE BAS NIVEAU (PID)
        # ----------------------------------------------
        
        # Calcul Vitesse/Cap
        direction_vec = target_pos - pos
        dist_final = np.linalg.norm(direction_vec)
        
        # Clamp distance pour éviter survitesse
        if dist_final > 1.0:
            target_pos = pos + (direction_vec / dist_final) * 1.0
            
        # Orientation Yaw (MODIFIÉ)
        if self.swarm_active and self.swarm_target_yaw is not None:
             # Si le Swarm nous donne un angle imposé (celui du leader), on le prend
             self.target_yaw_cache = self.swarm_target_yaw
        elif np.linalg.norm(direction_vec[:2]) > 0.5:
             # Sinon comportement standard : on regarde vers la cible
             self.target_yaw_cache = np.arctan2(direction_vec[1], direction_vec[0])

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        # Appel Contrôleur
        rpms, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.dt, 
            state=state_vec, 
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.array([0, 0, self.target_yaw_cache]) 
        )
        
        self._apply_lib_physics(rpms, gt)
        self.last_rpms = rpms
        self._sim_time += self.dt
        self._log(pos)

    def _apply_lib_physics(self, rpms, gt):
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        forces = np.array(rpms**2) * self.KF
        torques = np.array(rpms**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(self.bodyId, i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        
        try:
            p.applyExternalTorque(self.bodyId, 4, torqueObj=[0, 0, z_torque], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
            # Drag
            rot = np.array(p.getMatrixFromQuaternion(gt["orn_q"])).reshape(3,3)
            drag = -1 * self.DRAG_COEFF * np.sum(2 * np.pi * rpms / 60)
            f_drag = rot @ (drag * (rot.T @ gt["vel"]))
            p.applyExternalForce(self.bodyId, 4, forceObj=f_drag, posObj=[0,0,0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        except:
            pass 

    def _log(self, pos):
        if int(self._sim_time/self.dt)%10==0:
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow([self._sim_time, *pos])