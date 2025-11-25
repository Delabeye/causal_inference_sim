# entities/agent.py
import pybullet as p
import numpy as np


class Agent:
    """
    Classe de base pour les agents physiques (UAV, etc.).
    Gère bodyId et la vérité terrain.
    """

    def __init__(self, urdf_path, start_pos, start_orn_q, physics_client_id, dt: float):
        self.p = p
        self.dt = dt
        self.physics_client_id = physics_client_id

        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
            physicsClientId=self.physics_client_id,
        )

        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """Surchargée dans les sous-classes si besoin."""
        pass

    def get_ground_truth_state(self):
        if not self.p.isConnected(self.physics_client_id):
            raise RuntimeError("PyBullet n'est plus connecté au physics server.")

        pos, orn_q = self.p.getBasePositionAndOrientation(
            self.bodyId, physicsClientId=self.physics_client_id
        )
        vel, ang_vel = self.p.getBaseVelocity(
            self.bodyId, physicsClientId=self.physics_client_id
        )

        return {
            "pos": np.array(pos, dtype=float),
            "orn_q": np.array(orn_q, dtype=float),
            "vel": np.array(vel, dtype=float),
            "ang_vel": np.array(ang_vel, dtype=float),
        }
