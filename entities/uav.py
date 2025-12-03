import os
import sys
import csv
import threading
import pybullet as p
import numpy as np

from entities.agent import Agent
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF
from Control.Path_planning import RRTStarPlanner

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class UAV(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "UAV")
        
        urdf_path = config.get("urdf_path", "assets/quadrotor.urdf")
        start_pos = [0, 0, 1.0]
        start_orn = p.getQuaternionFromEuler([0,0,0])
        super().__init__(urdf_path, start_pos, start_orn, physics_client_id, self.dt)
        self._sim_time = 0.0
        
        # Physique
        self.KF, self.KM = 3.16e-10, 7.94e-12
        self.G, self.MAX_RPM = 9.8, 22000.0
        self.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])
        
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.last_rpms = np.zeros(4)
        
        # Navigation
        self.waypoints = [np.array(w) for w in config.get("waypoints", [[0,0,1]])]
        self.wp_idx = 0
        
        # --- Mémoire pour le Yaw (Stabilisation) ---
        self.target_yaw_cache = 0.0 # On garde en mémoire le dernier angle valide
        
        # --- PLANIFICATEUR RRT* ---
        rrt_config = {
            "world_bounds": {"x": [-10, 10], "y": [-10, 10], "z": [0.1, 5.0]},
            "step_size": 0.5,
            "max_iter": 500,
            "safe_distance": 0.8,
            "search_radius": 2.0
        }
        self.planner = RRTStarPlanner(rrt_config)
        self.replan_timer = 0 
        self.active_path = [] 
        
        # Multithreading
        self.is_planning = False
        self.planning_thread = None

        # Capteurs
        sens = config.get("sensors", {})
        self.ekf = GPSEKF(dt); self.ekf.x[:3] = start_pos
        self.gps = GPSSensor(sens.get("gps", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
        self.lidar = LidarSensor(sens.get("lidar", {})) 
        
        # Logs
        self.logging_enabled = True
        self.log_file = os.path.join("logs", f"{self.name}.csv")
        os.makedirs("logs", exist_ok=True)
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    def _run_async_plan(self, start_pos, target_pos, obstacles):
        try:
            path = self.planner.plan(start_pos, target_pos, obstacles, smooth=True)
            if path and len(path) > 0:
                self.active_path = path 
                print(f"[{self.name}] RRT* succès : {len(path)} points.")
        except Exception as e:
            print(f"[{self.name}] Erreur planning : {e}")
        finally:
            self.is_planning = False 

    def _analyze_target_accessibility(self, start_pos, target_pos):
        ray = p.rayTest(start_pos, target_pos, physicsClientId=self.physics_client_id)
        if not ray: return "CLEAR"
        hit_obj_id = ray[0][0]
        hit_fraction = ray[0][2]
        hit_pos = np.array(ray[0][3])
        if hit_fraction < 0.99 and hit_obj_id != self.bodyId:
            dist_hit_to_target = np.linalg.norm(target_pos - hit_pos)
            if dist_hit_to_target < 0.6: return "INVALID"
            else: return "BLOCKED"
        return "CLEAR"

    def think_and_act(self):
        if not p.isConnected(self.physics_client_id): return
        
        # 1. État
        gt = self.get_ground_truth_state()
        pos, vel = gt["pos"], gt["vel"]
        orn_q, ang_vel = gt["orn_q"], gt["ang_vel"]
        rpy = p.getEulerFromQuaternion(orn_q)
        
        # 2. Perception
        roll, pitch, yaw = rpy
        obstacles = self.lidar.measure(pos, roll, yaw, pitch)
        
        global_target = self.waypoints[self.wp_idx]

        # --- 3. ANALYSE ET GESTION DU CHEMIN ---
        is_cruising = False 
        
        if len(self.active_path) == 0 and not self.is_planning and self.replan_timer <= 0:
            status = self._analyze_target_accessibility(pos, global_target)
            if status == "INVALID":
                print(f"[{self.name}] ⚠️ Cible {self.wp_idx} inaccessible. SKIP !")
                if self.wp_idx < len(self.waypoints)-1:
                    self.wp_idx += 1
                    global_target = self.waypoints[self.wp_idx]
            elif status == "BLOCKED":
                if len(obstacles) > 0: 
                    print(f"[{self.name}] Chemin bloqué. Calcul RRT*...")
                    self.is_planning = True 
                    self.replan_timer = 50 
                    self.planning_thread = threading.Thread(
                        target=self._run_async_plan, 
                        args=(pos, global_target, obstacles)
                    )
                    self.planning_thread.start()

        # --- SUIVI DU CHEMIN ---
        if len(self.active_path) > 0:
            target_pos = self.active_path[0]
            dist_to_local = np.linalg.norm(target_pos - pos)
            acceptance = 0.6 if len(self.active_path) > 1 else 0.2
            
            if dist_to_local < acceptance:
                self.active_path.pop(0)
                if len(self.active_path) > 0:
                    target_pos = self.active_path[0]
                    is_cruising = True
                else:
                    target_pos = global_target
                    is_cruising = False
            else:
                is_cruising = (len(self.active_path) > 1)
        else:
            target_pos = global_target
            is_cruising = False
            
        if self.replan_timer > 0: self.replan_timer -= 1

        # --- 4. NAVIGATION GLOBALE ---
        if len(self.active_path) == 0 and not self.is_planning:
            dist_to_global = np.linalg.norm(global_target - pos)
            if dist_to_global < 0.2:
                if self.wp_idx < len(self.waypoints)-1:
                    self.wp_idx += 1
                    target_pos = self.waypoints[self.wp_idx]

        # --- 5. COMMANDE STABILISÉE ---
        
        if self.is_planning:
            if len(obstacles) > 0:
                dist_critique = min([np.linalg.norm(pos - obs) for obs in obstacles])
                if dist_critique < 0.6:
                    target_pos = pos 
                    target_vel_request = np.zeros(3) 
                else:
                    target_vel_request = -0.1 * vel 
            else:
                 target_vel_request = -0.1 * vel
        else:
            direction_vec = target_pos - pos
            dist_final = np.linalg.norm(direction_vec)
            
            MAX_TARGET_DIST = 1.0
            if dist_final > MAX_TARGET_DIST:
                target_clamped = pos + (direction_vec / dist_final) * MAX_TARGET_DIST
            else:
                target_clamped = target_pos
            
            target_pos = target_clamped
            
            if is_cruising:
                target_vel_request = np.zeros(3)
            else:
                # Approche finale : Freinage PLUS DOUX (-0.2 au lieu de -0.3)
                target_vel_request = -0.2 * vel

        # --- CALCUL DU YAW AVEC VERROUILLAGE (Stabilisation) ---
        diff_vec = target_pos - pos
        dist_planar = np.linalg.norm(diff_vec[:2])
        
        # [CORRECTIF] : Si on est loin (> 0.5m), on calcule le cap.
        # Sinon, on GARDE le cap précédent pour éviter la toupie.
        if dist_planar > 0.5:
            self.target_yaw_cache = np.arctan2(diff_vec[1], diff_vec[0])
            
        # On utilise toujours la valeur en cache (qui est soit à jour, soit figée si on est proche)
        target_yaw = self.target_yaw_cache

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        rpm_action, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.dt, 
            state=state_vec, 
            target_pos=target_pos,
            target_vel=target_vel_request,
            target_rpy=np.array([0, 0, target_yaw]) 
        )
        
        self._apply_lib_physics(rpm_action, gt)
        self.last_rpms = rpm_action
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