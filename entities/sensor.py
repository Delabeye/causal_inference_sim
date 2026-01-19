import math
import numpy as np
import pybullet as p


class Sensor:
    """
    Sensor class for measuring physical quantities.
    A base class that defines the interface for sensor implementations.
    Subclasses must implement the measure method to provide specific
    measurement functionality.
    Attributes:
        None
    Methods:
        measure(*args, **kwargs): Measure physical quantities.
            Must be implemented by subclasses.
    Raises:
        NotImplementedError: If measure method is called on base Sensor class.
    Example:
        >>> class TemperatureSensor(Sensor):
        ...     def measure(self, *args, **kwargs):
        ...         return 25.5
        >>> sensor = TemperatureSensor()
        >>> sensor.measure()
        25.5
    """
    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("The 'measure' method must be implemented.")


class GNSSensor(Sensor):
    """
    Simple GPS Sensor:
    - position_noise_std : standard deviation of position noise (m)
    - velocity_noise_std : standard deviation of velocity noise (m/s)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.pos_noise_std = float(config.get("position_noise_std", 0.0))
        self.vel_noise_std = float(config.get("velocity_noise_std", 0.0))

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
        pos_noise = np.random.normal(1.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)

        meas_pos = ground_truth_position + pos_noise
        meas_vel = ground_truth_velocity + vel_noise

        return meas_pos, meas_vel


# In entities/sensor.py

class IMUSensor:
    """
    IMUSensor class for simulating IMU (Inertial Measurement Unit) sensor measurements.
    This class simulates an accelerometer and gyroscope by computing proper acceleration
    in the sensor body frame, including gravity effects and measurement noise.
    Attributes:
        last_vel (np.ndarray): Previous velocity vector (3D) used for acceleration calculation.
        dt (float): Time step for discrete acceleration approximation (default: 0.01 seconds).
        g_vector (np.ndarray): Gravitational acceleration vector in world frame [0, 0, 9.81] m/s².
    Methods:
        __init__(config_dict=None):
            Initialize the IMUSensor with default parameters.
            Args:
                config_dict (dict, optional): Configuration dictionary for sensor parameters.
        measure(vel, orn_q):
            Simulate accelerometer and gyroscope measurements in the body frame.
            Calculates proper acceleration (what an accelerometer measures) by:
            1. Computing world frame acceleration from velocity change
            2. Adding gravitational acceleration (proper acceleration)
            3. Rotating to body frame using the orientation quaternion
            4. Adding white noise to accelerometer readings
            Args:
                vel (np.ndarray): Linear velocity in world frame (3D vector).
                orn_q (tuple/list): Orientation as a quaternion (4D).
            Returns:
                tuple: (acc_body, gyro_body)
                    - acc_body (np.ndarray): Measured acceleration in body frame with noise (3D).
                    - gyro_body (np.ndarray): Angular velocity in body frame (3D, currently zeros).
    """

    def __init__(self, config_dict=None):
        self.last_vel = np.zeros(3)
        self.dt = 1/100  # Assumed
        self.g_vector = np.array([0, 0, 9.81])

    def measure(self, vel, orn_q):
        """
        Simulates an accelerometer + gyroscope
        Returns: acc_body, gyro_body
        """
        # 1. Calculate World Acceleration (a = dv/dt)
        # (This is a discrete approximation)
        acc_world = (np.array(vel) - self.last_vel) / self.dt
        self.last_vel = np.array(vel)
        
        # 2. Add "Felt Gravity" (Proper Acceleration)
        # An accelerometer measures (a - g). Since g points downwards (-9.81), 
        # a_measured = a_world - (-9.81) = a_world + 9.81
        acc_proper_world = acc_world + self.g_vector
        
        # 3. Rotation to Body Frame (World -> Drone)
        # We use the inverse rotation matrix
        R_world_to_body = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3,3).T
        acc_body = R_world_to_body @ acc_proper_world
        
        # Add noise (optional)
        acc_body += np.random.normal(0, 0.1, 3) # White noise
        
        # Gyro (Angular velocity, here 0 or true value for simplicity)
        gyro_body = np.zeros(3) 
        
        return acc_body, gyro_body

class LidarSensor:
    """
    LidarSensor
    A configurable LiDAR sensor simulator that emulates a conical Field of View (FOV) 
    sensor mounted on a drone. It generates ray vectors in the local drone frame and 
    performs batch raycasting in PyBullet to detect obstacles and environmental features.
    The sensor operates in the drone's local coordinate system where:
    - X-axis: Forward direction
    - Y-axis: Left direction
    - Z-axis: Up direction
    Attributes:
        max_distance (float): Maximum detection range in meters (default: 5.0)
        angle_resolution (float): Angular resolution between rays in degrees (default: 2.0)
        half_fov_h (float): Half of the horizontal field of view in degrees
        half_fov_v (float): Half of the vertical field of view in degrees
        local_rays (np.ndarray): Pre-generated array of ray direction vectors in the drone's 
                                 local frame, shape (N, 3) where N is the number of rays
    Methods:
        _generate_local_rays(): Generates all ray vectors in the local drone frame by sweeping
                               horizontally and vertically within the FOV bounds, converting
                               spherical coordinates to Cartesian unit vectors.
        measure(position, roll, yaw, pitch): Performs batch raycasting from the drone's 
                                             current position and orientation to detect 
                                             obstacles in the environment using PyBullet.
    """
    def __init__(self, config: dict):
        """
        Configurable Lidar with a conical Field of View (FOV).
        """
        self.max_distance = float(config.get("max_distance", 5.0))
        self.angle_resolution = float(config.get("angle_resolution", 2.0))
        
        # FOV in degrees, converted to half-angles
        fov_h = float(config.get("fov_horizontal", 90.0))
        fov_v = float(config.get("fov_vertical", 30.0))
        
        self.half_fov_h = fov_h / 2.0
        self.half_fov_v = fov_v / 2.0
        
        # Pre-generation of ray vectors in the LOCAL frame of the drone
        # X-axis = Forward, Y = Left, Z = Up
        self.local_rays = self._generate_local_rays()
        print(f"[LidarSensor] Initialized : {len(self.local_rays)} rays (FOV H:{fov_h}°, V:{fov_v}°)")

    def _generate_local_rays(self):
        rays = []
        # Sweep from left to right (-fov_h/2 to +fov_h/2)
        for az in np.arange(-self.half_fov_h, self.half_fov_h, self.angle_resolution):
            # Sweep from bottom to top (-fov_v/2 to +fov_v/2)
            for el in np.arange(-self.half_fov_v, self.half_fov_v, self.angle_resolution):
                
                # Convert degrees -> radians
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Spherical to Cartesian coordinates (X is forward)
                # x = cos(el) * cos(az)
                # y = cos(el) * sin(az)
                # z = sin(el)
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                
                # Normalization (safety check)
                v = np.array([x, y, z])
                v = v / np.linalg.norm(v)
                rays.append(v)
        
        return np.array(rays)

    def measure(self, position, roll, yaw, pitch):
        """
        Performs a batch raycast in PyBullet.
        Args:
            position: [x, y, z] of the drone
            roll, yaw, pitch: current orientation in radians
        Returns:
            points: List of np.array [x, y, z] of detected impacts
        """
        # 1. Calculate drone rotation matrix
        # PyBullet uses [roll, pitch, yaw] order for Euler quaternions
        orn_q = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(orn_q)
        
        # Reshape flat matrix (9,) to (3,3)
        R = np.array(rot_matrix).reshape(3, 3)
        
        # 2. Rotate all local rays to world frame
        # Formula: Ray_World = R * Ray_Local
        # Vector optimization: (N,3) dot (3,3) -> (N,3)
        # Note: using transpose to align dimensions correctly
        world_rays_dir = self.local_rays @ R.T 
        
        # 3. Prepare start and end positions
        num_rays = len(world_rays_dir)
        ray_froms = np.tile(position, (num_rays, 1))
        ray_tos = ray_froms + world_rays_dir * self.max_distance
        
        # 4. PyBullet Raycast (Batch = very fast)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 5. Filter impacts
        detected_points = []
        for _, res in enumerate(results):
            # res structure: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
            hit_id = res[0]
            if hit_id >= 0: # If an object was hit (id >= 0)
                hit_pos = np.array(res[3])
                if hit_pos[2] > 0.01:  # Filter to avoid points too close
                    hit_id = [round(coord, 3) for coord in hit_pos]
                    detected_points.append(hit_pos)
                
        return detected_points
    
class HeightSensor:
    """
    HeightSensor class for simulating a downward-facing distance sensor.
    This class simulates a downward-facing LiDAR or Sonar sensor mounted on a drone.
    It performs raycasting to measure the distance to the ground or nearest obstacle below
    the drone, accounting for the drone's orientation and adding realistic noise.
    Attributes:
        client_id (int): PyBullet physics client ID for raycast queries.
        noise_std (float): Standard deviation of Gaussian noise added to measurements (default: 0.01).
        max_range (float): Maximum range of the sensor in meters (default: 4.0).
    Methods:
        measure(pos, orn_q): Measures the distance to the ground below the drone.
            Args:
                pos (tuple or array-like): Position of the drone in world coordinates (x, y, z).
                orn_q (tuple or array-like): Orientation of the drone as a quaternion (x, y, z, w).
            Returns:
                float or None: Measured distance in meters, or None if the ground is out of range.
                              Returns a non-negative value with added Gaussian noise.
    """
    def __init__(self, physics_client_id, noise_std=0.01, max_range=4.0):
        """
        Simulates a downward-facing distance sensor (Down-facing Lidar/Sonar).
        """
        self.client_id = physics_client_id
        self.noise_std = noise_std
        self.max_range = max_range

    def measure(self, pos, orn_q):
        """
        Returns measured distance to ground (or None if out of range).
        """
        # 1. Calculate direction vector (Sensor points to the "Bottom" of the drone)
        # In world frame, drone "down" changes if the drone tilts (Roll/Pitch)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3, 3)
        # The "Down" vector in drone frame is [0, 0, -1]
        # Rotate it to world frame
        down_vec_world = rot_mat @ np.array([0, 0, -1])

        # 2. Raycast (Shoot ray)
        start = np.array(pos)
        end = start + (down_vec_world * self.max_range)
        
        results = p.rayTest(start, end, physicsClientId=self.client_id)
        # results[0] contains [objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal]
        
        hit_fraction = results[0][2]
        
        # 3. Processing
        if hit_fraction == 1.0: # Nothing hit
            return None # Too high for the sensor
        
        # Actual distance = hit_fraction * max_range
        dist = hit_fraction * self.max_range
        
        # Add noise
        dist += np.random.normal(0, self.noise_std)
        
        # Negative value protection
        return max(0.0, dist)