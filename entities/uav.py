import os
import sys
import csv
import pybullet as p
import numpy as np

from entities.agent import Agent
from entities.sensor import GPSSensor, IMUSensor, LidarSensor
from Control.EKF import GPSEKF
# Import Librairie
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
        
        # Capteurs
        sens = config.get("sensors", {})
        self.ekf = GPSEKF(dt); self.ekf.x[:3] = start_pos
        self.gps = GPSSensor(sens.get("gps", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
        self.lidar = LidarSensor(sens.get("lidar", {})) # Lidar paramétré par config
        
        # Logs
        self.logging_enabled = True
        self.log_file = os.path.join("logs", f"{self.name}.csv")
        os.makedirs("logs", exist_ok=True)
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    def think_and_act(self):
        if not p.isConnected(self.physics_client_id): return
        
        # 1. État
        gt = self.get_ground_truth_state()
        pos, vel = gt["pos"], gt["vel"]
        orn_q, ang_vel = gt["orn_q"], gt["ang_vel"]
        rpy = p.getEulerFromQuaternion(orn_q)
        
        # --- 2. SCAN LIDAR ---
        roll, pitch, yaw = rpy
        obstacles = self.lidar.measure(pos, roll, yaw, pitch)
        
        if len(obstacles) > 0:
            dist_min = min([np.linalg.norm(pos - obs) for obs in obstacles])
            if dist_min < 2.0:
                pass 

        # --- 3. Waypoint & Navigation Précise ---
        final_target = self.waypoints[self.wp_idx]
        direction_vec = final_target - pos
        dist_to_target = np.linalg.norm(direction_vec)
        
        # [MODIFICATION 1] Seuil de précision plus strict (0.1m au lieu de 0.2m)
        if dist_to_target < 0.1:
            if self.wp_idx < len(self.waypoints)-1:
                self.wp_idx += 1
                final_target = self.waypoints[self.wp_idx]
                direction_vec = final_target - pos
                dist_to_target = np.linalg.norm(direction_vec)

        # [MODIFICATION 2] "Carrot Chasing" ralenti
        # On réduit la distance max à 1.0m (au lieu de 2.0m) pour limiter la vitesse de pointe
        MAX_TARGET_DIST = 1.0 
        
        if dist_to_target > MAX_TARGET_DIST:
            target_pos_clamped = pos + (direction_vec / dist_to_target) * MAX_TARGET_DIST
        else:
            target_pos_clamped = final_target

        # [MODIFICATION 3] Freinage Actif (Active Braking)
        # On demande une vitesse cible opposée au mouvement actuel (-30% de la vitesse actuelle)
        # Cela augmente artificiellement l'amortissement (Terme D du PID) pour éviter l'overshoot
        braking_vel = -0.3 * vel 

        # 4. Commande
        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        rpm_action, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.dt, 
            state=state_vec, 
            target_pos=target_pos_clamped,
            target_vel=braking_vel  # <-- Injection du freinage
        )
        
        # 5. Physique
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