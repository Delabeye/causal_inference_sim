import json
import pybullet as p
import numpy as np
from entities.agent import Agent
import zmq

class RadarStation(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.name = self.config.get("name", "Radar")
        self.physics_client_id = physics_client_id
        self.deterministic_communication = bool(
            self.config.get("deterministic_communication", False)
        )
        deterministic_seed = self.config.get("deterministic_seed", None)
        self.rng = (
            np.random.default_rng(int(deterministic_seed))
            if deterministic_seed is not None
            else None
        )
        
        # --- Noise Configuration ---
        self.pos_noise_std = float(self.config.get("position_noise_std", 0.1)) # Position noise (XYZ)
        self.range_noise_std = float(self.config.get("range_noise_std", 0.05)) # Distance noise (Range)
        self.pos_noise_mean = float(self.config.get("position_noise_mean",0.5))
        self.range_noise_mean = float(self.config.get("range_noise_mean",0.5))

        self.bodyId = self.config.get("bodyId", 1000)   
        
        # --- Temporal Configuration ---
        self.radar_period = float(self.config.get("period", 0.1)) # ex: 0.1s = 10Hz
        self.radar_last_time = 0.0
        
        self.type = self.config.get("type", "radar")

        # 1. Physical Configuration
        self.pos = self.config.get("pos", [0, 0, 0]) # Static position defined in config
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = self.config.get("urdf_path", "assets/cube.urdf") # Default cube
        
        super().__init__(urdf_path, self.pos, start_orn, physics_client_id, dt)
        
        # Make the object static (Mass = 0) and phantom
        p.changeDynamics(self.bodyId, -1, mass=0, localInertiaDiagonal=[0,0,0], physicsClientId=self.physics_client_id)
        # Distinctive color (Red semi-transparent)
        p.changeVisualShape(self.bodyId, -1, rgbaColor=[0.8, 0, 0, 0.6], physicsClientId=self.physics_client_id)

        # 2. Sensor Configuration
        self.detection_range = float(self.config.get("range", 15.0)) # Range in meters
        self.targets = [] # Reference to target list (filled by Manager)

        # 3. Network
        ip = self.config.get("ip", "localhost") # Default localhost
        port_out = self.config.get("port_out", 5557)
        if self.deterministic_communication:
            self.zmq_ctx = None
            self.pub_socket = None
        else:
            self.setup_network(ip, port_out)

    def setup_network(self, ip, port_pub):
        """Configure the radar radio (ZeroMQ)"""
        try:
            self.zmq_ctx = zmq.Context()
            self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
            self.pub_socket.setsockopt(zmq.LINGER, 0)
            # Using bind() because radar is infrastructure station (Server)
            # If using central broker, replace with connect()
            self.pub_socket.bind(f"tcp://{ip}:{port_pub}")
            print(f"[{self.name}] Radio active on tcp://{ip}:{port_pub}")
        except Exception as e:
            print(f"[{self.name}] ZMQ Error: {e}")

    def cleanup_network(self):
        """Close the radar publisher and its private ZMQ context."""

        socket = getattr(self, "pub_socket", None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except zmq.ZMQError:
                pass
            self.pub_socket = None

        context = getattr(self, "zmq_ctx", None)
        if context is not None:
            try:
                context.term()
            except zmq.ZMQError:
                pass
            self.zmq_ctx = None

    def publish_detection(self, report, sim_time):
        """Publish complete report via radio (ZeroMQ)"""
        if not report or self.pub_socket is None:
            return 
        
        wrapper = {
            "radar_name": self.name,
            "data": report,
            "timestamp": sim_time
        }
        # Send as JSON string
        try:
            self.pub_socket.send_string("RADAR " + json.dumps(wrapper))
        except Exception as e:
            print(f"[{self.name}] Send Error: {e}")

    def think_and_act(self, sim_time):
        """
        Main radar loop: Scan -> Measure -> Broadcast
        """
        # Internal time update
        detected_report = {}

        # Scan targets
        for agent in self.targets:
            if agent.bodyId == self.bodyId: continue # No self-detection
            
            # Ground truth
            target_pos, _ = p.getBasePositionAndOrientation(agent.bodyId, physicsClientId=self.physics_client_id)
            dist = np.linalg.norm(np.array(target_pos) - np.array(self.pos))
            
            if dist <= self.detection_range:
                # --- TARGET DETECTED ---
                
                # Measurement generation
                normal = self.rng.normal if self.rng is not None else np.random.normal
                meas_dist = dist + normal(self.range_noise_mean, self.range_noise_std)
                est_pos = np.array(target_pos) + normal(
                    self.pos_noise_mean, self.pos_noise_std, 3
                )
                
                # Transponder packet construction
                detected_report[agent.name] = {
                    "type": "radar",
                    
                    # Info for drone EKF (Correction)
                    "anchor_pos": self.pos,          # Radar position (List [x,y,z])
                    "measured_dist": meas_dist,      # Distance scalar
                    
                    # Info for other drones (Avoidance)
                    "pos": est_pos.tolist(),  # Convert numpy -> list for JSON
                }

        # Publish if detections
        if detected_report:
            self.publish_detection(detected_report, sim_time)
        
        # Return report for internal simulator use (if needed)
        return detected_report
