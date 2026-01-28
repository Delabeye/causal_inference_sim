import json
import numpy as np
import pybullet as p
from entities.uav import UAV
import zmq
import threading
import random

class Swarm:
    """
    Manager for multi-UAV collective behavior and formation control.

    This class orchestrates a group of UAVs to maintain a geometric formation 
    (defaulting to triangular) relative to a designated leader. It handles 
    coordinate frame transformations from the leader's body frame to the global 
    world frame and implements a distributed communication network using a 
    ZMQ proxy-based Pub/Sub architecture.

    Key features:
    1. Dynamic Formation: Followers track positions relative to the leader's heading.
    2. Yaw Smoothing: Implements a Low-Pass Filter on the swarm's orientation to 
        prevent erratic positioning during sharp leader turns.
    3. Network Simulation: Injects stochastic latency and jitter into inter-agent 
        message passing to model real-world communication constraints.
    4. Collision Avoidance: Applies repulsive forces between agents to maintain 
        a minimum separation distance.

    Attributes:
        leader (UAV): The primary agent defining the swarm's trajectory and heading.
        followers (list[UAV]): Agents tracking the formation offsets.
        agents_data (dict): Shared blackboard containing the latest perceived 
                        state of all agents.
        min_sep (float): Minimum safety distance (meters) between any two drones.
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
        """
        Initializes the swarm controller and establishes the communication backbone.

        Sets up the internal state, identifies the leader, and triggers the ZMQ 
        proxy thread to handle message routing between agents.

        Args:
            agents (list[UAV]): List of UAV instances participating in the swarm.
            leader_name (str, optional): Name of the specific UAV to act as leader. 
                                 Defaults to the first agent in the list.
            min_sep (float): Safety distance threshold for repulsion logic.
            avoid_gain (float): Strength of the repulsive force when within min_sep.
            formation_body_offsets (dict, optional): Custom [x, y, z] offsets for 
                                             specific followers.
            port_in (int): ZMQ XSUB port for incoming agent messages.
            port_out (int): ZMQ XPUB port for outgoing broadcast.
            ip (str): Network address for the communication proxy.
        """ 
        if len(agents) == 0:
            raise ValueError("Swarm requires at least one UAV agent.")
        self.sim_time = 0.0
        self.dt = agents[0].dt
        self.agents = agents
        self.name =  "swarm 1"
        #broadcast at 50 Hz
        self.broadcast_interval = 0.02  # broadcast at each step
        self.last_broadcast = -self.broadcast_interval
        self.prev_targets = {}
        #Com latency
        self.perception_delay_mean = 0.1  # 100ms latency
        self.perception_delay_std = 0.02  # +/- 20ms
        self.message_buffer = []
        agents_names = [a.name for a in agents if a.type == "uav"]
        # leader choice
        if leader_name is not None:
            leader_list = [a for a in agents if getattr(a, "name", "") == leader_name]
            if len(leader_list) == 0:
                raise ValueError(f"No UAV with name=“{leader_name}” found for the leader.")
            self.leader = leader_list[0]
            self.leader.leader = True
        else:
            # By default, the first UAV in the list is the leader.
            self.leader = agents[0]
            self.leader.leader = True
        self.agents_data = {}
        for agent in self.agents:
            self.agents_data[agent.name] = {"name" : agent.name, "pos": agent.start_pos, "vel": [0,0,0], "yaw": agent.start_orn[2]}
            agent.set_swarm_activate()  # Indicates that the agent is part of a swarm
            agent.swarm_name = agents_names
        self.followers_future_state = self.agents_data.copy()
        # Followers = others 
        self.followers: list[UAV] = [a for a in agents if a is not self.leader and a.type == "uav"]
        self.physics_client_id = self.leader.physics_client_id

        self.radar= [a for a in agents if a.type == "radar"]
                
        # ----------------- AVOIDANCE PARAMETERS -----------------
        self.min_sep = float(min_sep)
        self.avoid_gain = float(avoid_gain)

        # ----------------- NETWORK PARAMETERS -----------------
        self.port_in = port_in
        self.port_out = port_out
        self.ip = ip
        self.init_proxy()
        self.setup_swarm_com()
        # Each UAV should CONNECT to the proxy endpoints.
        # IMPORTANT: do not bind on the UAV side, otherwise ports collide
        # with the proxy on Windows (often reported as "Permission denied").
        for a in self.agents:
            a.setup_network_swarm(self.ip,self.port_in, self.port_out)
        self.leader.setup_network_swarm(self.ip, self.port_in, self.port_out)

        # ----------------- POSITION OFFSETS -----------------
        self.formation_body_offsets: dict[str, np.ndarray] = {}

        if formation_body_offsets is not None:
            for f in self.followers:
                if f.name not in formation_body_offsets:
                    raise ValueError(
                        f"formation_body_offsets does not contain an offset for follower '{f.name}'."
                    )
                off = np.array(formation_body_offsets[f.name], dtype=float)
                if off.shape != (3,):
                    raise ValueError("Each offset must be a 3D vector [x, y, z].")
                self.formation_body_offsets[f.name] = off
        else:
            self._assign_default_triangular_offsets()

        print(
            f"[Swarm] Swarm created with leader='{self.leader.name}', "
            f"{len(self.followers)} follower(s). "
            f"(min_sep={self.min_sep:.2f}, avoid_gain={self.avoid_gain:.2f})"
        )

    # ------------------------------------------------------------------
    def _assign_default_triangular_offsets(self):
        """
        Generates a hierarchical triangular formation pattern behind the leader.

        Calculates grid-based offsets where agents are placed in successive rows 
        behind the leader. Each row increases in width, forming a 'V' or 
        triangle shape. Offsets are stored in the leader's local coordinate system.
        """
        
        spacing_x = 1  # lat distance on x (m)
        spacing_y = 1  # lat distance on y (m)

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
                z_offset = 0.0  # same altitude as leader

                self.formation_body_offsets[f.name] = np.array(
                    [x_offset, y_offset, z_offset], dtype=float
                )
                idx += 1
            row += 1
    
    # ------------------------------------------------------------------
    def init_proxy(self):
        """
        Spawns a dedicated background thread to run a ZMQ XSUB/XPUB proxy.

        The proxy acts as a centralized message broker, allowing distributed agents 
        to publish their states and subscribe to the states of neighbors without 
        direct peer-to-peer coupling.
        """
        def run_proxy():
            try:
                # Context ZMQ for thread proxy
                ctx = zmq.Context()

                # FRONTEND (Input): Use XSUB to relay subscriptions
                frontend = ctx.socket(zmq.XSUB)
                frontend.bind(f"tcp://*:{self.port_in}")

                # BACKEND (Output): Use XPUB to broadcast
                backend = ctx.socket(zmq.XPUB)
                backend.bind(f"tcp://*:{self.port_out}")

                print(f"[Swarm Network] Proxy started (In: {self.port_in} -> Out: {self.port_out})")
                
                # The proxy runs here indefinitely.
                # We do NOT store the sockets in 'self' because they belong to this thread.
                zmq.proxy(frontend, backend)
                
            except zmq.ContextTerminated:
                print("[Swarm Network] Context ZMQ terminated.")
            except Exception as e:
                print(f"[Swarm Network] Error in proxy : {e}")
            finally:
                # Thread-specific cleaning
                frontend.close()
                backend.close()
                ctx.term()

        # Thread starting
        self.proxy_thread = threading.Thread(target=run_proxy, daemon=True)
        self.proxy_thread.start()

    def setup_swarm_com(self):
        """
        Configures the Swarm manager's local network interface.

        Initializes the PUB/SUB sockets used by the manager to broadcast swarm-wide 
        objectives and listen to the telemetry of individual agents.
        """
        self.client_ctx = zmq.Context()
        self.sub_socket = self.client_ctx.socket(zmq.SUB)
        self.pub_socket = self.client_ctx.socket(zmq.PUB)
        # Connection to localhost
        self.sub_socket.connect(f"tcp://localhost:{self.port_out}")
        
        # Subscribe to all topics
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1) # Timeout 1ms 

        self.pub_socket.connect(f"tcp://localhost:{self.port_in}")
        self.pub_socket.setsockopt(zmq.LINGER, 1)  # Immediate closure

    def broadcast_state(self):
        """Sends the current position and speed to the network"""
        # Send on topic 'SWARM'
        # Format: "TOPIC JSON"
        self.pub_socket.send_string("SWARM " + json.dumps(self.agents_data))

    def broadcast_future_pos(self):
        """Sends the calculated future position of followers to the network"""
        # Send on topic 'FUTURE_POS'
        self.pub_socket.send_string("FUTURE_POS " + json.dumps(self.followers_future_state))
        pass

    def listen_swarm(self):
        """
        Polls the network for new messages and simulates perception latency.

        Incoming messages are stored in a time-sorted buffer and only processed 
        once the simulated 'sim_time' exceeds the message's 'visible_time' 
        (reception time + stochastic delay). This ensures the controller operates 
        on realistic, slightly outdated data.
        """
        while True:
            try:
                # Non-blocking reading
                msg = self.sub_socket.recv_string()
                delay = max(0, random.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self.sim_time + delay
                self.message_buffer.append((visible_time,msg))
            except zmq.Again:
                # No more messages
                break
            except Exception as e:
                print(f"Network error on {self.name}: {e}")
                break
            
        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self.sim_time >= target_time:
                # --- Message ready : processing ---
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
                
        # We replace the old buffer with the remaining ones.
        self.message_buffer = buffer_remaining
            

    def cleanup(self):
        """Close connections properly"""
        self.sub_socket.close()
        self.pub_socket.close()
        self.client_ctx.term()
    # ------------------------------------------------------------------
    def update(self):
        """
        Executes the formation control and coordination logic for the current step.

        This method performs the core geometric calculations:
        1. Leader State Retrieval: Obtains the latest known pose of the leader.
        2. Yaw Smoothing: Filters the leader's yaw to produce a stable rotation matrix.
        3. Coordinate Mapping: Transforms body-frame offsets into world-frame targets.
        4. Repulsion Resolution: Adjusts targets to resolve inter-agent proximity conflicts.
        5. Telemetry Broadcast: Disseminates the updated 'future states' to all 
        followers via the ZMQ network.
        """
        if (self.sim_time - self.last_broadcast) >= self.broadcast_interval:
            if len(self.followers) == 0:
                return
            self.listen_swarm()

            # 1) Reading the leader's pose
            try:
                state_leader = self.agents_data[self.leader.name]
            except KeyError:
                return

            pos_leader = np.array(state_leader["pos"])
            vel_leader = np.array(state_leader["vel"])
            target_yaw_leader = state_leader["yaw"]

            # --- SMOOTHER INIT ---
            # We store the smoothed yaw in self for continuity between steps.
            if not hasattr(self, "smooth_swarm_yaw"):
                self.smooth_swarm_yaw = target_yaw_leader

            # --- SMOOTHING ALGORITHM (Low Pass Filter on the angle) ---
            # We calculate the angle difference (managing the -π/π jump)
            diff_yaw = np.arctan2(np.sin(target_yaw_leader - self.smooth_swarm_yaw), np.cos(target_yaw_leader - self.smooth_swarm_yaw))
            
            # Fluidity parameter:
            # 0.1 = very slow (the swarm takes time to turn)
            # 0.5 = responsive but fluid
            # 1.0 = instantaneous (your current code crashes)
            alpha_yaw = 0.3 
            
            # We also limit the maximum rotation speed of the group (e.g. 1 rad/s)
            max_rot_speed = 2.0 * self.dt 
            step_yaw = np.clip(diff_yaw * alpha_yaw, -max_rot_speed, max_rot_speed)
            
            self.smooth_swarm_yaw += step_yaw

            # It is this smoothed yaw that we use for geometry.
            cy = np.cos(self.smooth_swarm_yaw)
            sy = np.sin(self.smooth_swarm_yaw)
            R_yaw = np.array([[cy, -sy, 0.0], [sy,  cy, 0.0], [0.0, 0.0, 1.0]])

            # 2) Recover positions (unchanged)
            all_agents = [self.leader] + self.followers
            positions = {}
            for a in all_agents:
                try:
                    st = a.get_ground_truth_state()
                    positions[a.name] = st["pos"]
                except: continue

            # 3) Target calculation
            dt_swarm = self.sim_time - self.last_broadcast 
            if dt_swarm <= 0: dt_swarm = 0.1

            for follower in self.followers:
                off_body = self.formation_body_offsets.get(follower.name, None)
                if off_body is None: continue

                # Target position based on SMOOTHED YAW
                off_world = R_yaw @ off_body
                base_target = pos_leader + off_world

                # Correction Repulsion (unchanged)
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

                # Speed calculation
                prev_target = self.prev_targets.get(follower.name, final_target) if hasattr(self, "prev_targets") else final_target
                if not hasattr(self, "prev_targets"): self.prev_targets = {}
                
                target_vel_computed = (final_target - prev_target) / dt_swarm
                self.prev_targets[follower.name] = final_target

                self.followers_future_state[follower.name] = {
                    "pos": final_target.tolist(), 
                    "vel": target_vel_computed.tolist(),
                    "yaw": self.smooth_swarm_yaw 
                }
            
            self.broadcast_state() 
            self.broadcast_future_pos()
            self.last_broadcast = self.sim_time
            self.sim_time += self.dt
        else:
            self.sim_time += self.dt
            