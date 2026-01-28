import pybullet as p
import numpy as np
import os

class Agent:
    """
    Base class for all physical entities within the PyBullet simulation.

    This class manages the instantiation of a robot or object from a URDF file. 
    It handles path resolution for assets and ensures that the physical 
    properties (mass, inertia tensors) defined in the URDF are correctly 
    interpreted by the physics engine.

    Attributes:
        p (module): Reference to the PyBullet physics module.
        dt (float): The simulation time step (seconds).
        physics_client_id (int): The handle for the specific PyBullet physics server.
        bodyId (int): The unique integer ID assigned to this object by PyBullet.
    """
    def __init__(self, urdf_path, start_pos, start_orn_q, physics_client_id, dt: float):
        """
        Loads the agent into the physics world and initializes its pose.

        Automatically resolves absolute and relative paths to find the URDF file. 
        It uses the URDF_USE_INERTIA_FROM_FILE flag, which is critical for 
        high-fidelity drone simulations where precise moments of inertia 
        dictate flight stability.

        Args:
            urdf_path (str): Path to the .urdf file defining the agent.
            start_pos (list/np.ndarray): Initial Cartesian coordinates [x, y, z].
            start_orn_q (list/np.ndarray): Initial orientation as a quaternion [x, y, z, w].
            physics_client_id (int): The ID of the connected physics client.
            dt (float): The integration time step in seconds.
        """
        self.p = p
        self.dt = dt
        self.physics_client_id = physics_client_id
        
        # Absolute path to avoid issues
        if not os.path.exists(urdf_path):
            base = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(os.path.dirname(base), urdf_path)

        # Flag essential for drone physics
        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client_id,
        )

    def get_ground_truth_state(self):
        """
        Queries the physics engine for the absolute state of the agent.

        This provides the 'Ground Truth' (perfect knowledge) of the agent's 
        kinematics, bypassing any simulated sensor noise or estimation errors.

        Returns:
            dict: A dictionary containing:
                - "pos" (np.ndarray): World position [x, y, z].
                - "orn_q" (np.ndarray): Orientation quaternion [x, y, z, w].
                - "vel" (np.ndarray): Linear velocity [vx, vy, vz].
                - "ang_vel" (np.ndarray): Angular velocity [wx, wy, wz].
            If the physics server is disconnected, returns an empty dictionary.
        """
        if not self.p.isConnected(self.physics_client_id): return {}
        pos, orn = self.p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
        vel, ang_vel = self.p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
        return {
            "pos": np.array(pos), "orn_q": np.array(orn), 
            "vel": np.array(vel), "ang_vel": np.array(ang_vel)
        }