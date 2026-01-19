import pybullet as p
import numpy as np
import os

class Agent:
    """Agent
    Represents an agent loaded into a PyBullet simulation from a URDF file and
    provides convenience access to its base pose and velocity.
    Parameters
    ----------
    urdf_path : str
        Path to the agent URDF file. If a relative path is provided, it will be
        resolved relative to this module's parent directory.
    start_pos : Sequence[float]
        Initial base position as (x, y, z).
    start_orn_q : Sequence[float]
        Initial base orientation as a quaternion (x, y, z, w).
    physics_client_id : int
        The PyBullet physics client id used for all simulation API calls.
    dt : float
        Simulation time step associated with the agent.
    Attributes
    ----------
    p : module
        The PyBullet module or client wrapper used for API calls.
    bodyId : int
        The ID returned by PyBullet when the URDF is loaded.
    physics_client_id : int
        Stored physics client id.
    dt : float
        Stored simulation timestep.
    Methods
    -------
    get_ground_truth_state()
        Query the physics client for the agent's current base state and return a
        dictionary with numpy arrays:
          - "pos": shape (3,) base position
          - "orn_q": shape (4,) base orientation quaternion
          - "vel": shape (3,) base linear velocity
          - "ang_vel": shape (3,) base angular velocity
        If the physics client is not connected, returns an empty dict.
    Notes
    -----
    - The URDF is loaded with the URDF_USE_INERTIA_FROM_FILE flag.
    - The method implementations expect numpy to be available as `np`.
    """
    def __init__(self, urdf_path, start_pos, start_orn_q, physics_client_id, dt: float):
        self.p = p
        self.dt = dt
        self.physics_client_id = physics_client_id
        
        """Resolve URDF path if relative."""
        if not os.path.exists(urdf_path):
            base = os.path.dirname(os.path.abspath(__file__))
            # On remonte de 'entities' vers 'Stage'
            urdf_path = os.path.join(os.path.dirname(base), urdf_path)

        """Load the agent URDF into the simulation."""
        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client_id,
        )

    def get_ground_truth_state(self):
        if not self.p.isConnected(self.physics_client_id): return {}
        pos, orn = self.p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
        vel, ang_vel = self.p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
        return {
            "pos": np.array(pos), "orn_q": np.array(orn), 
            "vel": np.array(vel), "ang_vel": np.array(ang_vel)
        }