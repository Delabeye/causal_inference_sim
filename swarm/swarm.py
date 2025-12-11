import numpy as np
from entities.uav import UAV

class Swarm:
    def __init__(self, agents: list[UAV]):
        self.agents = agents
        self.leader = None
        self.followers = []

        for agent in self.agents:
            if agent.name == "drone_0":
                self.leader = agent
            else:
                self.followers.append(agent)

        print(f"[Swarm] Initialisé. Leader: {self.leader.name}, {len(self.followers)} Suiveurs.")

        # Formation en "V" (Delta)
        self.formation_offsets = {
            "drone_1": np.array([-1.5, -1.5, 0.0]),
            "drone_2": np.array([-1.5, 1.5, 0.0]),
        }

    def run_step(self):
        # 1. Leader (Autonome)
        self.leader.think_and_act(setpoint_pos=None, setpoint_vel=None)

        # État du Leader (via EKF ou Vrai pour simu parfaite)
        if self.leader.ekf:
            l_pos = self.leader.ekf.x[0:3]
            l_vel = self.leader.ekf.x[3:6] # On récupère aussi la vitesse !
        else:
            # Fallback (Vérité terrain si pas d'EKF, pour debug)
            l_pos = np.array(self.leader.last_gps_meas[0]) if self.leader.last_gps_meas else np.zeros(3)
            l_vel = np.array(self.leader.last_gps_meas[1]) if self.leader.last_gps_meas else np.zeros(3)

        l_yaw = self.leader.current_yaw

        # 2. Suiveurs
        for follower in self.followers:
            if follower.name in self.formation_offsets:
                offset_local = self.formation_offsets[follower.name]

                # Rotation du offset
                c, s = np.cos(l_yaw), np.sin(l_yaw)
                R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                
                # CIBLE POSITION
                target_pos = l_pos + R_yaw @ offset_local
                
                # CIBLE VITESSE (Feedforward)
                # Le suiveur doit avoir la même vitesse que le leader
                # (On néglige la vitesse de rotation du leader pour simplifier l'effet de bras de levier)
                target_vel = l_vel 

                # Évitement collision simple
                avoidance = np.zeros(3)
                f_pos = follower.ekf.x[0:3] if follower.ekf else np.zeros(3)
                
                for other in self.agents:
                    if other == follower: continue
                    o_pos = other.ekf.x[0:3] if other.ekf else np.zeros(3)
                    diff = f_pos - o_pos
                    dist = np.linalg.norm(diff)
                    if dist < 0.8 and dist > 0.01:
                        avoidance += (diff/dist) * (0.8 - dist) * 2.0
                
                # On envoie Position ET Vitesse au suiveur
                follower.think_and_act(setpoint_pos=target_pos + avoidance, setpoint_vel=target_vel)
            else:
                follower.think_and_act(setpoint_pos=None)