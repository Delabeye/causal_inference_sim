import numpy as np
import pybullet as p

class Sensor:
    """
    Abstract base class for all simulated sensors.

    Defines the standard interface for sensor implementations. All derived 
    classes must implement the 'measure' method to provide data to the 
    agent's control system.
    """
    def __init__(self):
        pass

    def measure(self):
        raise NotImplementedError("The 'measure' method must be implemented.")


class GNSSensor(Sensor):
    """
    Simulates a Global Navigation Satellite System (GNSS) receiver.

    Models the absolute position and velocity estimation of a UAV by applying 
    stochastic noise (Gaussian/White noise) to the simulation's ground truth. 
    This allows for testing the robustness of navigation algorithms against 
    GPS inaccuracies and signal drift.

    Attributes:
        pos_noise_std (float): Standard deviation of position noise (meters).
        vel_noise_std (float): Standard deviation of velocity noise (m/s).
        pos_noise_mean (float): Systematic bias in position measurement.
        vel_noise_mean (float): Systematic bias in velocity measurement.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.pos_noise_std = float(config.get("position_noise_std", 0.0))
        self.vel_noise_std = float(config.get("velocity_noise_std", 0.0))
        self.pos_noise_mean = float(config.get("position_noise_mean", 0.0))
        self.vel_noise_mean = float(config.get("velocity_noise_mean", 0.0))

        if self.pos_noise_std < 0.0:
            self.pos_noise_std = 0.0
        if self.vel_noise_std < 0.0:
            self.vel_noise_std = 0.0

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (measured_position, measured_velocity) with Gaussian noise.
        """
        pos_noise = np.random.normal(self.pos_noise_mean, self.pos_noise_std, 3)
        vel_noise = np.random.normal(self.vel_noise_mean, self.vel_noise_std, 3)

        meas_pos = ground_truth_position + pos_noise
        meas_vel = ground_truth_velocity + vel_noise

        return meas_pos, meas_vel


class IMUSensor:
    """
    Inertial Measurement Unit (IMU) simulating a 6-DOF sensor.

    Calculates 'proper acceleration' (specific force) and angular velocity in 
    the vehicle's body frame. Proper acceleration includes the effects of 
    physical motion ($a = \frac{dv}{dt}$) and the constant opposing force 
    of gravity.

    Attributes:
        dt (float): Measurement frequency (seconds), used for discrete differentiation.
        g_vector (np.ndarray): Gravity vector in the world frame $[0, 0, 9.81]$.
        ast_vel (np.ndarray): Velocity from the previous step to calculate acceleration.
    """
    def __init__(self, config_dict):
        self.last_vel = np.zeros(3)
        self.dt = config_dict.get("dt", 1/100) 
        self.noise_mean = config_dict.get("accel_noise_mean", 0.0)
        self.noise_std = config_dict.get("accel_noise_std", 0.0)
        self.g_vector = np.array([0, 0, 9.81])

    def measure(self, vel, orn_q):
        """
        Simulates the raw accelerometer and gyroscope output.

        Transforms world-frame kinematics into the non-inertial body frame of 
        the drone, accounting for gravity and sensor noise.

        Args:
            vel (np.ndarray): Current world linear velocity $[v_x, v_y, v_z]$.
            orn_q (np.ndarray): Current orientation quaternion $[x, y, z, w]$.

        Returns:
            tuple[np.ndarray, np.ndarray]: (acc_body, gyro_body) vectors.
        """
        # 1. Calculate world acceleration (a = dv/dt)
        # (This is a discrete approximation)
        acc_world = (np.array(vel) - self.last_vel) / self.dt
        self.last_vel = np.array(vel)
        
        # 2. Add "felt gravity" (Proper Acceleration)
        # An accelerometer measures (a - g). Since g points downward (-9.81),
        # a_measured = a_world - (-9.81) = a_world + 9.81
        acc_proper_world = acc_world + self.g_vector
        
        # 3. Rotation to body frame (World -> Drone)
        # Use the inverse rotation matrix
        R_world_to_body = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3, 3).T
        acc_body = R_world_to_body @ acc_proper_world
        
        # Add noise (optional)
        acc_body += np.random.normal(0, 0.1, 3)  # White noise
        
        # Gyro (Angular velocity, set to 0 for simplicity)
        gyro_body = np.zeros(3) 
        
        return acc_body, gyro_body

class LidarSensor:
    """
    Configurable 3D LiDAR sensor simulating ray-based obstacle detection.

    Generates a point cloud by performing a batch of raycasts within a defined 
    conical field of view (FOV). This is highly optimized using PyBullet's 
    `rayTestBatch` for real-time environmental perception.

    Attributes:
        max_distance (float): Range limit of the laser pulses (meters).
        angle_resolution (float): Angular increment between adjacent rays (degrees).
        local_rays (np.ndarray): Pre-computed direction vectors in the drone's frame.
    """
    def __init__(self, config: dict):
        """
        Configurable Lidar with conical field of view (FOV).
        """
        self.max_distance = float(config.get("max_distance", 5.0))
        self.angle_resolution = float(config.get("angle_resolution", 2.0))
        
        # FOV in degrees, converted to half-angles
        fov_h = float(config.get("fov_horizontal", 90.0))
        fov_v = float(config.get("fov_vertical", 30.0))
        
        self.half_fov_h = fov_h / 2.0
        self.half_fov_v = fov_v / 2.0
        
        # Pre-generate ray vectors in the LOCAL frame of the drone
        # X axis = Front, Y = Left, Z = Up
        self.local_rays = self._generate_local_rays()
        print(f"[LidarSensor] Initialized: {len(self.local_rays)} rays (FOV H:{fov_h}°, V:{fov_v}°)")

    def _generate_local_rays(self):
        rays = []
        # Sweep left to right (-fov_h/2 to +fov_h/2)
        for az in np.arange(-self.half_fov_h, self.half_fov_h, self.angle_resolution):
            # Sweep bottom to top (-fov_v/2 to +fov_v/2)
            for el in np.arange(-self.half_fov_v, self.half_fov_v, self.angle_resolution):
                
                # Convert degrees to radians
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Spherical to Cartesian coordinates (X points forward)
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                
                # Normalize (for safety)
                v = np.array([x, y, z])
                v = v / np.linalg.norm(v)
                rays.append(v)
        
        return np.array(rays)

    def measure(self, position, roll, yaw, pitch):
        """
        Performs a LiDAR scan and returns a set of impact coordinates.

        Args:
            position (np.ndarray): Origin of the LiDAR sensor in world coordinates.
            roll, yaw, pitch (float): Orientation angles in radians.

        Returns:
            list[np.ndarray]: List of detected impact points in the world frame.
        """
        # 1. Calculate drone rotation matrix
        # PyBullet uses [roll, pitch, yaw] order for Euler quaternions
        orn_q = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(orn_q)
        
        # Transform flat matrix (9,) to (3,3)
        R = np.array(rot_matrix).reshape(3, 3)
        
        # 2. Rotate all local rays to world frame
        # Formula: Ray_World = R * Ray_Local
        # Vectorized optimization: (N,3) dot (3,3) -> (N,3)
        world_rays_dir = self.local_rays @ R.T 
        
        # 3. Prepare start and end positions
        num_rays = len(world_rays_dir)
        ray_froms = np.tile(position, (num_rays, 1))
        ray_tos = ray_froms + world_rays_dir * self.max_distance
        
        # 4. PyBullet batch raycast (very fast)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 5. Filter hits
        detected_points = []
        for _, res in enumerate(results):
            # res structure: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
            hit_id = res[0]
            if hit_id >= 0:  # If hit an object (id >= 0)
                hit_pos = np.array(res[3])
                if hit_pos[2] > 0.01:  # Filter out points too close
                    detected_points.append(hit_pos)
                
        return detected_points
    
class HeightSensor:
    """
    Simulates a downward-facing altimeter (Sonar or 1D LiDAR).

    Used for precision landing and terrain following. Unlike a global Z-coordinate, 
    this sensor measures the distance to the surface directly beneath the vehicle, 
    accounting for vehicle tilt (Roll/Pitch).

    Attributes:
        noise_std (float): Precision of the distance measurement.
        max_range (float): Maximum sensing distance to the terrain.
    """
    def __init__(self, physics_client_id, noise_std=0.01, max_range=4.0):
        """
        Initializes the height sensor.
        Args:
            physics_client_id (int): PyBullet physics client ID.
            noise_std (float): Standard deviation of measurement noise (meters).
            max_range (float): Maximum sensing distance (meters).
        """
        self.client_id = physics_client_id
        self.noise_std = noise_std
        self.max_range = max_range

    def measure(self, pos, orn_q):
        """
        Calculates the distance to the closest obstacle beneath the sensor.

        Args:
            pos (np.ndarray): World position of the drone.
            orn_q (np.ndarray): Drone orientation quaternion.

        Returns:
            float|None: Measured distance in meters, or None if no surface is detected.
        """
        # 1. Calculate direction vector (Sensor points down from drone)
        # In world frame, drone's down direction changes if drone tilts (Roll/Pitch)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3, 3)
        # The "down" vector in drone frame is [0, 0, -1]
        # Rotate to world frame
        down_vec_world = rot_mat @ np.array([0, 0, -1])

        # 2. Raycast
        start = np.array(pos)
        end = start + (down_vec_world * self.max_range)
        
        results = p.rayTest(start, end, physicsClientId=self.client_id)
        # results[0] contains [objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal]
        
        hit_fraction = results[0][2]
        
        # 3. Process result
        if hit_fraction == 1.0:  # Nothing hit
            return None  # Too high for sensor
        
        # Actual distance = hit_fraction * max_range
        dist = hit_fraction * self.max_range
        
        # Add noise
        dist += np.random.normal(0, self.noise_std)
        
        # Prevent negative values
        return max(0.0, dist)
