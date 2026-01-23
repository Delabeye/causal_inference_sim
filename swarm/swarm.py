import json
import numpy as np
import pybullet as p
from entities.uav import UAV
import zmq
import threading
import random

class Swarm:
    """
    Classe représentant un essaim de drones (leader + followers).

    - Le leader suit ses propres waypoints / consignes (gérés dans UAV).
    - Les followers se placent en FORMATION TRIANGULAIRE derrière le leader,
      avec des offsets exprimés dans le REPÈRE DU LEADER (axe x vers l'avant).
    - À chaque pas de temps, on met à jour la cible de chaque follower :
        target = position_leader + Rz(yaw_leader) * offset_body
    - On ajoute une correction de "répulsion" pour maintenir une distance
      minimale entre les drones (éviter de se rentrer dedans).
    - On transmet aussi la VITESSE et le YAW du leader pour un suivi fluide.
    """

    def __init__(
        self,
        agents: list[UAV],
        leader_name: str | None = None,
        min_sep: float = 0.6,
        avoid_gain: float = 0.5,
        formation_body_offsets: dict[str, np.ndarray] | None = None,
        port_in: int = 5556,
        port_out: int = 5557,
        ip: str = "localhost",
    ):
        if len(agents) == 0:
            raise ValueError("Swarm nécessite au moins un agent UAV.")
        self.sim_time = 0.0
        self.dt = agents[0].dt
        self.agents = agents
        self.name =  "swarm 1"
        #broadcast a 50 Hz
        self.broadcast_interval = 0.02  # broadcast à chaque step
        self.last_broadcast = -self.broadcast_interval
        self.prev_targets = {}
        #Com latency
        self.perception_delay_mean = 0.1  # 100ms de retard
        self.perception_delay_std = 0.02  # +/- 20ms
        self.message_buffer = []
        agents_names = [a.name for a in agents if a.type == "uav"]
        # Choix du leader
        if leader_name is not None:
            leader_list = [a for a in agents if getattr(a, "name", "") == leader_name]
            if len(leader_list) == 0:
                raise ValueError(f"Aucun UAV avec name='{leader_name}' trouvé pour le leader.")
            self.leader = leader_list[0]
            self.leader.leader = True
        else:
            # par défaut, le premier UAV de la liste est le leader
            self.leader = agents[0]
            self.leader.leader = True
        self.agents_data = {}
        for agent in self.agents:
            self.agents_data[agent.name] = {"name" : agent.name, "pos": agent.start_pos, "vel": [0,0,0], "yaw": agent.start_orn[2]}
            agent.set_swarm_activate()  # Indique que l'agent fait partie d'un essaim
            agent.swarm_name = agents_names
        self.followers_future_state = self.agents_data.copy()
        # Followers = tous les autres
        self.followers: list[UAV] = [a for a in agents if a is not self.leader and a.type == "uav"]
        self.physics_client_id = self.leader.physics_client_id

        self.radar= [a for a in agents if a.type == "radar"]
                
        # ----------------- PARAMÈTRES D'ÉVITAGE -----------------
        self.min_sep = float(min_sep)
        self.avoid_gain = float(avoid_gain)

        # ----------------- PARAMÈTRES RÉSEAU -----------------
        self.port_in = port_in
        self.port_out = port_out
        self.ip = ip
        self.init_proxy()
        self.setup_swarm_com()
        for a in self.agents:
            a.setup_network_swarm(self.ip,self.port_in, self.port_out)
        self.leader.setup_network_swarm(self.ip, self.port_in, self.port_out)
        # ----------------- OFFSETS DE FORMATION -----------------
        self.formation_body_offsets: dict[str, np.ndarray] = {}

        if formation_body_offsets is not None:
            for f in self.followers:
                if f.name not in formation_body_offsets:
                    raise ValueError(
                        f"formation_body_offsets ne contient pas d'offset pour follower '{f.name}'."
                    )
                off = np.array(formation_body_offsets[f.name], dtype=float)
                if off.shape != (3,):
                    raise ValueError("Chaque offset doit être un vecteur 3D [x, y, z].")
                self.formation_body_offsets[f.name] = off
        else:
            self._assign_default_triangular_offsets()

        print(
            f"[Swarm] Essaim créé avec leader='{self.leader.name}', "
            f"{len(self.followers)} follower(s). "
            f"(min_sep={self.min_sep:.2f}, avoid_gain={self.avoid_gain:.2f})"
        )

    # ------------------------------------------------------------------
    def _assign_default_triangular_offsets(self):
        """
        Assigne automatiquement une formation triangulaire derrière le leader.
        """
        spacing_x = 1  # distance entre rangées en x (m)
        spacing_y = 1  # distance latérale en y (m)

        followers = self.followers
        n = len(followers)
        if n == 0:
            return

        idx = 0
        row = 1
        while idx < n:
            num_in_row = row
            center = 0.5 * (num_in_row - 1)
            for j in range(num_in_row):
                if idx >= n:
                    break
                f = followers[idx]

                x_offset = -row * spacing_x
                y_offset = (j - center) * spacing_y
                z_offset = 0.0  # même altitude que le leader

                self.formation_body_offsets[f.name] = np.array(
                    [x_offset, y_offset, z_offset], dtype=float
                )
                idx += 1
            row += 1
    
    # ------------------------------------------------------------------
    def init_proxy(self):
        def run_proxy():
            try:
                # Contexte ZMQ pour le thread proxy
                ctx = zmq.Context()

                # FRONTEND (Entrée) : Utiliser XSUB pour relayer les abonnements
                frontend = ctx.socket(zmq.XSUB)
                frontend.bind(f"tcp://*:{self.port_in}")

                # BACKEND (Sortie) : Utiliser XPUB pour diffuser
                backend = ctx.socket(zmq.XPUB)
                backend.bind(f"tcp://*:{self.port_out}")

                print(f"[Swarm Network] Proxy démarré (In: {self.port_in} -> Out: {self.port_out})")
                
                # Le proxy tourne ici indéfiniment. 
                # On ne stocke PAS les sockets dans 'self' car ils appartiennent à ce thread.
                zmq.proxy(frontend, backend)
                
            except zmq.ContextTerminated:
                print("[Swarm Network] Contexte ZMQ terminé.")
            except Exception as e:
                print(f"[Swarm Network] Erreur dans le proxy : {e}")
            finally:
                # Nettoyage propre au thread
                frontend.close()
                backend.close()
                ctx.term()

        # Démarrage du thread
        self.proxy_thread = threading.Thread(target=run_proxy, daemon=True)
        self.proxy_thread.start()

    def setup_swarm_com(self):
        """
        Configure le Swarm pour qu'il écoute aussi son propre réseau 
        (comme un drone client).
        """
        self.client_ctx = zmq.Context()
        self.sub_socket = self.client_ctx.socket(zmq.SUB)
        self.pub_socket = self.client_ctx.socket(zmq.PUB)
        # On se CONNECTE à localhost (car le proxy est sur la même machine)
        self.sub_socket.connect(f"tcp://localhost:{self.port_out}")
        
        # On s'abonne à tout (ou au topic 'SWARM')
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1) # Timeout 1ms pour ne pas bloquer

        self.pub_socket.connect(f"tcp://localhost:{self.port_in}")
        self.pub_socket.setsockopt(zmq.LINGER, 1)  # Fermeture immédiate

    def broadcast_state(self):
        """Envoie la position et vitesse actuelle au réseau"""
        # Envoi sur le topic 'SWARM'
        # Format: "TOPIC JSON"
        self.pub_socket.send_string("SWARM " + json.dumps(self.agents_data))

    def broadcast_future_pos(self):
        """Envoie la position future calculée des followers au réseau"""
        # Envoi sur le topic 'FUTURE_POS'
        self.pub_socket.send_string("FUTURE_POS " + json.dumps(self.followers_future_state))
        pass

    def listen_swarm(self):
        """
        Vérifie la boite aux lettres et met à jour la liste des voisins.
        À appeler à chaque step.
        """
        while True:
            try:
                # Lecture non-bloquante
                msg = self.sub_socket.recv_string()
                delay = max(0, random.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self.sim_time + delay
                self.message_buffer.append((visible_time,msg))
            except zmq.Again:
                # Plus de messages
                break
            except Exception as e:
                print(f"Erreur réseau sur {self.name}: {e}")
                break
            
        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self.sim_time >= target_time:
                # --- LE MESSAGE EST PRÊT : ON LE TRAITE ---
                if " " in msg:
                    topic, json_str = msg.split(" ", 1)
                    try:
                        if topic == "State":
                            data = json.loads(json_str)
                            if "name" in data:
                                self.agents_data[data["name"]] = data
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
                # --- PAS ENCORE PRÊT : ON LE GARDE ---
        # On remplace l'ancien buffer par ceux qui restent
        self.message_buffer = buffer_remaining
            

    def cleanup(self):
        """Ferme proprement les connexions (important)"""
        self.sub_socket.close()
        self.pub_socket.close()
        self.client_ctx.term()
    # ------------------------------------------------------------------
    def update(self):
        """
        Met à jour la cible avec un LISSAGE DU YAW pour éviter les sauts de position.
        """
        if (self.sim_time - self.last_broadcast) >= self.broadcast_interval:
            if len(self.followers) == 0:
                return
            self.listen_swarm()

            # 1) Lecture de la pose du leader
            try:
                state_leader = self.agents_data[self.leader.name]
            except KeyError:
                return

            pos_leader = np.array(state_leader["pos"])
            vel_leader = np.array(state_leader["vel"])
            target_yaw_leader = state_leader["yaw"]

            # --- INIT DU SMOOTHER ---
            # On stocke le yaw lissé dans self pour la continuité entre les steps
            if not hasattr(self, "smooth_swarm_yaw"):
                self.smooth_swarm_yaw = target_yaw_leader

            # --- ALGORITHME DE LISSAGE (Low Pass Filter sur l'angle) ---
            # On calcule la différence d'angle (en gérant le saut -pi/pi)
            diff_yaw = np.arctan2(np.sin(target_yaw_leader - self.smooth_swarm_yaw), np.cos(target_yaw_leader - self.smooth_swarm_yaw))
            
            # Paramètre de fluidité :
            # 0.1 = très lent (le swarm met du temps à tourner)
            # 0.5 = réactif mais fluide
            # 1.0 = instantané (votre code actuel qui crash)
            alpha_yaw = 0.3 
            
            # On limite aussi la vitesse de rotation max du groupe (ex: 1 rad/s)
            max_rot_speed = 2.0 * self.dt 
            step_yaw = np.clip(diff_yaw * alpha_yaw, -max_rot_speed, max_rot_speed)
            
            self.smooth_swarm_yaw += step_yaw

            # C'est CE yaw lissé qu'on utilise pour la géométrie
            cy = np.cos(self.smooth_swarm_yaw)
            sy = np.sin(self.smooth_swarm_yaw)
            R_yaw = np.array([[cy, -sy, 0.0], [sy,  cy, 0.0], [0.0, 0.0, 1.0]])

            # 2) Récup positions (inchangé)
            all_agents = [self.leader] + self.followers
            positions = {}
            for a in all_agents:
                try:
                    st = a.get_ground_truth_state()
                    positions[a.name] = st["pos"]
                except: continue

            # 3) Calcul des cibles
            dt_swarm = self.sim_time - self.last_broadcast 
            if dt_swarm <= 0: dt_swarm = 0.1

            for follower in self.followers:
                off_body = self.formation_body_offsets.get(follower.name, None)
                if off_body is None: continue

                # Position cible basée sur le YAW LISSÉ
                off_world = R_yaw @ off_body
                base_target = pos_leader + off_world

                # Correction Répulsion (inchangé)
                correction = np.zeros(3)
                pos_f = positions.get(follower.name, base_target)
                for other in all_agents:
                    if other is follower: continue
                    pos_o = positions.get(other.name, None)
                    if pos_o is None: continue
                    diff = base_target - pos_o; diff[2] = 0.0
                    dist = float(np.linalg.norm(diff))
                    if 0 < dist < self.min_sep:
                        correction += (self.min_sep - dist) * (diff / dist)

                final_target = base_target + self.avoid_gain * correction
                final_target[2] = base_target[2]

                # Calcul vitesse (méthode précédente)
                prev_target = self.prev_targets.get(follower.name, final_target) if hasattr(self, "prev_targets") else final_target
                if not hasattr(self, "prev_targets"): self.prev_targets = {}
                
                target_vel_computed = (final_target - prev_target) / dt_swarm
                self.prev_targets[follower.name] = final_target

                self.followers_future_state[follower.name] = {
                    "pos": final_target.tolist(), 
                    "vel": target_vel_computed.tolist(),
                    "yaw": self.smooth_swarm_yaw # On demande au drone de suivre le lissage
                }
            
            self.broadcast_state() 
            self.broadcast_future_pos()
            self.last_broadcast = self.sim_time
            self.sim_time += self.dt
        else:
            self.sim_time += self.dt
            