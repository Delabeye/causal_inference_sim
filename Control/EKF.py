import numpy as np

class EKF:
    """
    Extended Kalman Filter (EKF) for UAV State Estimation
    A comprehensive EKF implementation for estimating the 3D position and velocity of an 
    Unmanned Aerial Vehicle (UAV) by fusing Inertial Measurement Unit (IMU) and Global 
    Positioning System (GPS) data.
    State Vector:
        x = [x, y, z, vx, vy, vz]
        where (x, y, z) is position in world frame and (vx, vy, vz) is velocity.
    Key Features:
        - IMU-aided prediction with proper coordinate frame transformations
        - Gravity compensation for accurate acceleration measurement
        - GPS measurement updates for position and velocity corrections
        - Configurable process and measurement noise covariances
        - Robust quaternion-based attitude representation
        - Exception handling for singular matrix operations
    Attributes:
        dt (float): Sampling time interval (seconds)
        x (np.ndarray): State vector [x, y, z, vx, vy, vz] (meters, m/s)
        P (np.ndarray): State covariance matrix (6x6)
        F (np.ndarray): State transition matrix (6x6)
        Q (np.ndarray): Process noise covariance matrix (6x6)
        H (np.ndarray): Measurement matrix (6x6)
        R (np.ndarray): Measurement noise covariance matrix (6x6)
        GRAVITY (np.ndarray): Gravitational acceleration vector [0, 0, 9.81] m/s²
    Methods:
        predict(imu_acc_body, orientation_quat): Predict next state using IMU acceleration
        update(pos_meas, vel_meas): Correct state estimate using GPS measurements
        _quat_to_rot_matrix(q): Convert quaternion to rotation matrix
    Notes:
        - IMU acceleration is assumed to be in the drone body frame
        - GPS measurements are assumed to be in the world frame
        - Gravity is automatically subtracted from IMU measurements
        - Quaternion format: [x, y, z, w]
    """
    def __init__(self, dt):
        """
        Initialize Extended Kalman Filter for 6D state tracking (position and velocity).
        :param dt: Time step for state transition
        """
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State transition matrix (kinematics: x_new = x + v*dt)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        
        # Initial uncertainty
        self.P = np.eye(6) * 1.0
        
        # Process noise (Q)
        # Reduce velocity noise since IMU provides precise info
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.01 
        self.Q[3:6, 3:6] *= 0.05 

        # Measurement matrices GPS (Position + Velocity)
        self.H = np.eye(6)
        self.R = np.eye(6)
        self.R[0:3, 0:3] *= 2.0   # GPS Position confidence (Low)
        self.R[3:6, 3:6] *= 0.5   # GPS Velocity confidence (Medium)

        self.GRAVITY = np.array([0, 0, 9.81])

    def _quat_to_rot_matrix(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix"""
        x, y, z, w = q
        # Standard formula
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R

    def predict(self, imu_acc_body, orientation_quat):
        """
        IMU-aided prediction.
        imu_acc_body: [ax, ay, az] measured by IMU (m/s^2)
        orientation_quat: [x, y, z, w] current orientation
        """
        # 1. Acceleration rotation: Drone frame -> World frame
        # IMU gives acceleration in drone reference frame
        R = self._quat_to_rot_matrix(orientation_quat)
        acc_world = R @ imu_acc_body
        
        # 2. Remove gravity
        # Accelerometer also measures gravity (9.81 upward when flat).
        # Must subtract to get pure motion acceleration.
        acc_linear = acc_world - self.GRAVITY
        
        # 3. Physics prediction (Newton's laws)
        # Future position = Pos + Vel*dt + 0.5*Acc*dt^2
        # Future velocity = Vel + Acc*dt
        
        # First apply velocity part (F*x)
        self.x = self.F @ self.x
        
        # Then add acceleration part (Control matrix B*u)
        # Position term (0.5 * a * dt^2)
        self.x[0:3] += 0.5 * acc_linear * (self.dt**2)
        # Velocity term (a * dt)
        self.x[3:6] += acc_linear * self.dt
        
        # 4. Covariance update
        # Add noise since IMU is imperfect
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, pos_meas, vel_meas):
        """Standard GPS correction"""
        z = np.hstack([pos_meas, vel_meas])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except:
            return self.x[:3], self.x[3:6]
        
        self.x = self.x + (K @ y)
        I = np.eye(6)
        self.P = (I - (K @ self.H)) @ self.P
        
        return self.x[:3], self.x[3:6]
