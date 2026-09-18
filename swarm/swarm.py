import json
import numpy as np
import pybullet as p
from entities.uav import UAV
import zmq
import threading
import random
import copy

class Swarm:
    """
    Formation controller with an explicit directed reference topology.

    Most formations use a leader-star topology.  A line can instead use a
    directed chain, in which each follower tracks its immediate predecessor.
    Collision avoidance remains pairwise and is logged as a separate mechanism.
    """

    def __init__(
        self,
        agents: list[UAV],
        leader_name: str | None = None,
        min_sep: float = 0.6,
        avoid_gain: float = 0.5,
        formation_body_offsets: dict[str, np.ndarray] | None = None,
        reference_by_follower: dict[str, str] | None = None,
        formation_control_mode: str = "relational",
        attraction_gain: float = 0.8,
        velocity_alignment_gain: float = 0.35,
        relational_lookahead_s: float = 0.25,
        max_relational_correction_speed: float = 1.5,
        formation_ready_tolerance: float = 0.25,
        port_in: int = 5556,
        port_out: int = 5557,
        ip: str = "localhost",
        deterministic_communication: bool = False,
        communication_seed: int = 0,
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
        self.last_central_avoidance_raw = {}
        self.last_central_avoidance_applied = {}
        self.last_formation_error_norm = {}
        #Com latency
        self.perception_delay_mean = 0.1  # 100ms de retard
        self.perception_delay_std = 0.02  # +/- 20ms
        self.message_buffer = []
        self.deterministic_communication = bool(deterministic_communication)
        self.communication_rng = random.Random(int(communication_seed))
        self._direct_last_agent_source_time = {}
        self._direct_uav_message_buffer = []
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
        self.leader.formation_hold_active = bool(
            self.followers
            and getattr(self.leader, "formation_hold_leader_until_ready", True)
        )
        self.physics_client_id = self.leader.physics_client_id

        self.radar= [a for a in agents if a.type == "radar"]
                
        # ----------------- PARAMÈTRES D'ÉVITAGE -----------------
        self.min_sep = float(min_sep)
        self.avoid_gain = float(avoid_gain)
        self.formation_control_mode = str(formation_control_mode).strip().lower()
        if self.formation_control_mode not in {"relational", "direct_offset"}:
            raise ValueError(
                "formation_control_mode must be 'relational' or 'direct_offset'."
            )
        self.attraction_gain = float(attraction_gain)
        self.velocity_alignment_gain = float(velocity_alignment_gain)
        self.relational_lookahead_s = float(relational_lookahead_s)
        self.max_relational_correction_speed = float(
            max_relational_correction_speed
        )
        self.formation_ready_tolerance = float(formation_ready_tolerance)
        if min(
            self.attraction_gain,
            self.velocity_alignment_gain,
            self.relational_lookahead_s,
            self.max_relational_correction_speed,
            self.formation_ready_tolerance,
        ) < 0.0:
            raise ValueError("Relational formation gains and limits must be non-negative.")
        if self.relational_lookahead_s == 0.0:
            raise ValueError("relational_lookahead_s must be strictly positive.")

        # ----------------- PARAMÈTRES RÉSEAU -----------------
        self.port_in = port_in
        self.port_out = port_out
        self.ip = ip
        if not self.deterministic_communication:
            self.setup_swarm_com()
            for a in self.agents:
                a.setup_network_swarm(self.ip,self.port_in, self.port_out)
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
        self._configure_reference_topology(reference_by_follower)

        print(
            f"[Swarm] Essaim créé avec leader='{self.leader.name}', "
            f"{len(self.followers)} follower(s). "
            f"(mode={self.formation_control_mode}, min_sep={self.min_sep:.2f}, "
            f"avoid_gain={self.avoid_gain:.2f})"
        )

    def _configure_reference_topology(
        self, reference_by_follower: dict[str, str] | None
    ) -> None:
        """Resolve each follower's control parent and local edge offset."""

        agents_by_name = {str(agent.name): agent for agent in self.agents}
        configured = reference_by_follower or {}
        self.reference_agents: dict[str, UAV] = {}
        self.reference_offsets: dict[str, np.ndarray] = {}

        for follower in self.followers:
            reference_name = str(configured.get(follower.name, self.leader.name))
            if reference_name not in agents_by_name:
                raise ValueError(
                    f"Unknown formation reference '{reference_name}' for '{follower.name}'."
                )
            reference = agents_by_name[reference_name]
            if reference is follower:
                raise ValueError(f"A UAV cannot be its own formation reference: {follower.name}.")
            self.reference_agents[follower.name] = reference
            reference_global_offset = (
                np.zeros(3)
                if reference is self.leader
                else self.formation_body_offsets.get(reference.name)
            )
            if reference_global_offset is None:
                raise ValueError(
                    f"Reference '{reference.name}' must precede '{follower.name}' in the formation."
                )
            self.reference_offsets[follower.name] = (
                self.formation_body_offsets[follower.name]
                - np.asarray(reference_global_offset, dtype=float)
            )

        for follower in self.followers:
            visited = {follower.name}
            reference = self.reference_agents[follower.name]
            while reference is not self.leader:
                if reference.name in visited or reference.name not in self.reference_agents:
                    raise ValueError("Formation references must form an acyclic path to the leader.")
                visited.add(reference.name)
                reference = self.reference_agents[reference.name]

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

    @staticmethod
    def pairwise_avoidance_contributions(
        receiver_name,
        base_target,
        positions,
        agent_names,
        min_sep,
        avoid_gain,
    ):
        """Return exact per-sender terms before and after ``avoid_gain``."""
        raw = {}
        applied = {}
        base_target = np.asarray(base_target, dtype=float)
        for sender_name in agent_names:
            if sender_name == receiver_name:
                continue
            sender_pos = positions.get(sender_name)
            if sender_pos is None:
                continue
            diff = base_target - np.asarray(sender_pos, dtype=float)
            diff = diff.copy()
            diff[2] = 0.0
            dist = float(np.linalg.norm(diff))
            contribution = np.zeros(3)
            if 0.0 < dist < float(min_sep):
                contribution = (float(min_sep) - dist) * (diff / dist)
            raw[sender_name] = contribution
            applied[sender_name] = float(avoid_gain) * contribution
        return raw, applied

    @staticmethod
    def relational_formation_terms(
        receiver_pos,
        receiver_vel,
        reference_pos,
        reference_vel,
        desired_world_offset,
        attraction_gain,
        velocity_alignment_gain,
        maximum_correction_speed,
    ):
        """Compute the directed elastic interaction ``reference -> receiver``.

        The desired offset is an equilibrium displacement, not an absolute
        navigation target. The reference velocity transports the formation;
        attraction and velocity alignment only correct relative error.
        """
        receiver_pos = np.asarray(receiver_pos, dtype=float)
        receiver_vel = np.asarray(receiver_vel, dtype=float)
        reference_pos = np.asarray(reference_pos, dtype=float)
        reference_vel = np.asarray(reference_vel, dtype=float)
        desired_world_offset = np.asarray(desired_world_offset, dtype=float)
        expected_shape = (3,)
        values = (
            receiver_pos,
            receiver_vel,
            reference_pos,
            reference_vel,
            desired_world_offset,
        )
        if any(value.shape != expected_shape for value in values):
            raise ValueError("Relational formation states and offsets must be 3D vectors.")

        relative_position = receiver_pos - reference_pos
        position_error = desired_world_offset - relative_position
        velocity_error = reference_vel - receiver_vel
        attraction_raw = float(attraction_gain) * position_error
        damping_raw = float(velocity_alignment_gain) * velocity_error

        correction = attraction_raw + damping_raw
        correction_norm = float(np.linalg.norm(correction))
        maximum = float(maximum_correction_speed)
        scale = (
            maximum / correction_norm
            if maximum > 0.0 and correction_norm > maximum
            else 1.0
        )
        attraction_applied = scale * attraction_raw
        damping_applied = scale * damping_raw
        correction_applied = attraction_applied + damping_applied
        command_velocity = reference_vel + correction_applied
        return {
            "relative_position": relative_position,
            "position_error": position_error,
            "velocity_error": velocity_error,
            "attraction_raw": attraction_raw,
            "attraction_applied": attraction_applied,
            "damping_raw": damping_raw,
            "damping_applied": damping_applied,
            "correction_scale": float(scale),
            "correction_velocity": correction_applied,
            "transport_velocity": reference_vel.copy(),
            "command_velocity": command_velocity,
        }
    
 

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

    def _sample_communication_delay(self, mean, std):
        return max(0.0, self.communication_rng.gauss(float(mean), float(std)))

    def _enqueue_direct_agent_states(self):
        """Queue newly broadcast UAV states using simulation time only."""
        for agent in sorted(self.agents, key=lambda item: item.name):
            packet = getattr(agent, "last_broadcast_packet", None)
            if not isinstance(packet, dict):
                continue
            source_time = float(packet.get("sim_time", -np.inf))
            if source_time <= self._direct_last_agent_source_time.get(agent.name, -np.inf):
                continue
            self._direct_last_agent_source_time[agent.name] = source_time
            visible_time = self.sim_time + self._sample_communication_delay(
                self.perception_delay_mean, self.perception_delay_std
            )
            self.message_buffer.append(
                (visible_time, agent.name, copy.deepcopy(packet))
            )

    def _deliver_direct_agent_states(self):
        remaining = []
        for visible_time, sender_name, packet in self.message_buffer:
            if self.sim_time + 1e-12 >= visible_time:
                self.agents_data[sender_name] = packet
            else:
                remaining.append((visible_time, sender_name, packet))
        self.message_buffer = remaining

    def _schedule_direct_uav_broadcasts(self):
        """Emulate SWARM/FUTURE_POS delivery without OS/network scheduling."""
        swarm_snapshot = copy.deepcopy(self.agents_data)
        future_snapshot = copy.deepcopy(self.followers_future_state)
        for receiver in sorted(self.agents, key=lambda item: item.name):
            delay = self._sample_communication_delay(
                receiver.perception_delay_mean, receiver.perception_delay_std
            )
            self._direct_uav_message_buffer.append(
                (self.sim_time + delay, "swarm", receiver, swarm_snapshot)
            )
            if receiver in self.followers:
                delay = self._sample_communication_delay(
                    receiver.perception_delay_mean, receiver.perception_delay_std
                )
                self._direct_uav_message_buffer.append(
                    (
                        self.sim_time + delay,
                        "future",
                        receiver,
                        copy.deepcopy(future_snapshot.get(receiver.name)),
                    )
                )

    def _deliver_direct_uav_messages(self):
        remaining = []
        for visible_time, topic, receiver, payload in self._direct_uav_message_buffer:
            if self.sim_time + 1e-12 < visible_time:
                remaining.append((visible_time, topic, receiver, payload))
                continue
            if topic == "future":
                if payload is not None:
                    receiver.future_state = payload
                continue
            for sender_name, sender_state in payload.items():
                if receiver.swarm_name and sender_name not in receiver.swarm_name:
                    continue
                receiver.neighbors_data[sender_name] = sender_state
                receiver.last_message_receive_time_by_sender[sender_name] = float(
                    receiver._sim_time
                )
                try:
                    receiver.last_message_source_time_by_sender[sender_name] = float(
                        sender_state.get("sim_time", np.nan)
                    )
                except (TypeError, ValueError):
                    receiver.last_message_source_time_by_sender[sender_name] = np.nan
                if sender_name != receiver.name and not receiver.leader:
                    receiver.other_agent_pos[sender_name] = np.asarray(
                        sender_state["pos"], dtype=float
                    )
        self._direct_uav_message_buffer = remaining

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
        if self.deterministic_communication:
            self._enqueue_direct_agent_states()
            self._deliver_direct_agent_states()
            return

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
        if self.deterministic_communication:
            return
        self.sub_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.client_ctx.term()
    # ------------------------------------------------------------------
    def update(self):
        """
        Met à jour la cible avec un LISSAGE DU YAW pour éviter les sauts de position.
        """
        if self.leader.formation_hold_active and self._followers_ready_for_mission():
            self.leader.formation_hold_active = False

        if self.deterministic_communication:
            self._deliver_direct_uav_messages()

        if (self.sim_time - self.last_broadcast) >= self.broadcast_interval:
            if len(self.followers) == 0:
                return
            self.listen_swarm()

            # 1) Lecture de la pose du leader
            try:
                state_leader = self.agents_data[self.leader.name]
            except KeyError:
                return

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
            velocities = {}
            for a in all_agents:
                try:
                    st = a.get_ground_truth_state()
                    positions[a.name] = st["pos"]
                    velocities[a.name] = st["vel"]
                except: continue

            # 3) Calcul des cibles
            dt_swarm = self.sim_time - self.last_broadcast 
            if dt_swarm <= 0: dt_swarm = 0.1

            for follower in self.followers:
                reference = self.reference_agents[follower.name]
                state_reference = self.agents_data.get(reference.name)
                off_body = self.reference_offsets.get(follower.name)
                if state_reference is None:
                    continue
                if off_body is None: continue
                if follower.name not in positions or follower.name not in velocities:
                    continue

                if reference is self.leader:
                    reference_rotation = R_yaw
                    reference_yaw = self.smooth_swarm_yaw
                else:
                    reference_yaw = float(state_reference.get("yaw", self.smooth_swarm_yaw))
                    cy_ref = np.cos(reference_yaw)
                    sy_ref = np.sin(reference_yaw)
                    reference_rotation = np.array(
                        [[cy_ref, -sy_ref, 0.0], [sy_ref, cy_ref, 0.0], [0.0, 0.0, 1.0]]
                    )
                off_world = reference_rotation @ off_body
                formation_metadata = {
                    "formation_control_mode": self.formation_control_mode,
                    "desired_relative_offset": off_world.copy(),
                }
                if self.formation_control_mode == "relational":
                    receiver_pos = np.asarray(positions[follower.name], dtype=float)
                    receiver_vel = np.asarray(velocities[follower.name], dtype=float)
                    reference_pos = np.asarray(state_reference["pos"], dtype=float)
                    reference_vel = np.asarray(
                        state_reference.get("vel", np.zeros(3)), dtype=float
                    )
                    terms = self.relational_formation_terms(
                        receiver_pos=receiver_pos,
                        receiver_vel=receiver_vel,
                        reference_pos=reference_pos,
                        reference_vel=reference_vel,
                        desired_world_offset=off_world,
                        attraction_gain=self.attraction_gain,
                        velocity_alignment_gain=self.velocity_alignment_gain,
                        maximum_correction_speed=self.max_relational_correction_speed,
                    )
                    raw_by_sender = {}
                    applied_by_sender = {}
                    target_vel_computed = terms["command_velocity"]
                    final_target = (
                        receiver_pos
                        + self.relational_lookahead_s * target_vel_computed
                    )
                    final_target[2] = max(
                        final_target[2],
                        float(getattr(follower, "formation_safe_altitude", 0.0)),
                    )
                    formation_metadata = {
                        "formation_control_mode": self.formation_control_mode,
                        "desired_relative_offset": off_world.copy(),
                        "relative_position": terms["relative_position"],
                        "relative_position_error_by_sender": {
                            reference.name: terms["position_error"]
                        },
                        "relative_velocity_error_by_sender": {
                            reference.name: terms["velocity_error"]
                        },
                        "formation_attraction_raw_by_sender": {
                            reference.name: terms["attraction_raw"]
                        },
                        "formation_attraction_applied_by_sender": {
                            reference.name: terms["attraction_applied"]
                        },
                        "formation_damping_raw_by_sender": {
                            reference.name: terms["damping_raw"]
                        },
                        "formation_damping_applied_by_sender": {
                            reference.name: terms["damping_applied"]
                        },
                        "formation_transport_velocity_by_sender": {
                            reference.name: terms["transport_velocity"]
                        },
                        "formation_correction_scale": terms["correction_scale"],
                    }
                    self.last_formation_error_norm[follower.name] = float(
                        np.linalg.norm(terms["position_error"])
                    )
                else:
                    base_target = (
                        np.asarray(state_reference["pos"], dtype=float) + off_world
                    )
                    raw_by_sender, applied_by_sender = (
                        self.pairwise_avoidance_contributions(
                            receiver_name=follower.name,
                            base_target=base_target,
                            positions=positions,
                            agent_names=[agent.name for agent in all_agents],
                            min_sep=self.min_sep,
                            avoid_gain=self.avoid_gain,
                        )
                    )
                    correction_applied = sum(
                        applied_by_sender.values(), np.zeros(3)
                    )
                    final_target = base_target + correction_applied
                    final_target[2] = base_target[2]
                    target_vel_computed = None
                    self.last_formation_error_norm[follower.name] = float(
                        np.linalg.norm(base_target - positions[follower.name])
                    )
                self.last_central_avoidance_raw[follower.name] = {
                    name: value.copy() for name, value in raw_by_sender.items()
                }
                self.last_central_avoidance_applied[follower.name] = {
                    name: value.copy() for name, value in applied_by_sender.items()
                }

                # Calcul vitesse (méthode précédente)
                prev_target = self.prev_targets.get(follower.name, final_target) if hasattr(self, "prev_targets") else final_target
                if not hasattr(self, "prev_targets"): self.prev_targets = {}
                
                if target_vel_computed is None:
                    target_vel_computed = (final_target - prev_target) / dt_swarm
                self.prev_targets[follower.name] = final_target

                self.followers_future_state[follower.name] = {
                    "pos": final_target.tolist(), 
                    "vel": target_vel_computed.tolist(),
                    "yaw": reference_yaw,
                    "controller_generated_time": float(self.sim_time),
                    "leader_reference_sender": reference.name,
                    "leader_source_time": state_reference.get("sim_time", None),
                    "leader_message_received_flag": int("sim_time" in state_reference),
                    "central_avoidance_raw_by_sender": {
                        name: value.tolist() for name, value in raw_by_sender.items()
                    },
                    "central_avoidance_applied_by_sender": {
                        name: value.tolist() for name, value in applied_by_sender.items()
                    },
                    **{
                        key: (
                            {
                                name: np.asarray(value).tolist()
                                for name, value in field_value.items()
                            }
                            if isinstance(field_value, dict)
                            else (
                                np.asarray(field_value).tolist()
                                if isinstance(field_value, np.ndarray)
                                else field_value
                            )
                        )
                        for key, field_value in formation_metadata.items()
                    },
                }
            
            if self.deterministic_communication:
                self._schedule_direct_uav_broadcasts()
            else:
                self.broadcast_state()
                self.broadcast_future_pos()
            self.last_broadcast = self.sim_time
            self.sim_time += self.dt
        else:
            self.sim_time += self.dt

    def _followers_ready_for_mission(self):
        """Return true once every follower has joined its formation offset."""
        if not self.followers:
            return True
        lifecycle_ready = all(
            bool(getattr(follower, "formation_takeoff_complete", False))
            and float(getattr(follower, "formation_transition_timer", 0.0))
            >= float(getattr(follower, "formation_transition_duration", 0.0))
            for follower in self.followers
        )
        if (
            not lifecycle_ready
            or getattr(self, "formation_control_mode", "direct_offset")
            != "relational"
        ):
            return lifecycle_ready
        return all(
            self.last_formation_error_norm.get(follower.name, np.inf)
            <= self.formation_ready_tolerance
            for follower in self.followers
        )
