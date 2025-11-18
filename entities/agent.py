import pybullet as p
import numpy as np


class Agent:
    """
    Classe de base pour tous les agents physiques (UAV, UGV, etc.).
    Gère le bodyId PyBullet et fournit la vérité terrain.
    """

    def __init__(self, urdf_path, start_pos, start_orn_q, dt: float):
        self.p = p
        self.dt = dt

        # On suppose que p.connect(...) a déjà été fait AVANT
        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
        )
        print(f"Agent chargé avec bodyId: {self.bodyId}")

        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """
        Surchargée dans les sous-classes pour créer capteurs/contrôleurs.
        """
        pass

    def get_ground_truth_state(self):
        """
        Retourne l'état parfait depuis PyBullet (position, orientation, vitesses).
        """
        if not self.p.isConnected():
            raise RuntimeError("PyBullet n'est pas connecté (get_ground_truth_state).")

        pos, orn_q = self.p.getBasePositionAndOrientation(self.bodyId)
        vel, ang_vel = self.p.getBaseVelocity(self.bodyId)

        return {
            "pos": np.array(pos),
            "orn_q": np.array(orn_q),
            "vel": np.array(vel),
            "ang_vel": np.array(ang_vel),
        }

    def think_and_act(self, setpoint):
        raise NotImplementedError("La méthode 'think_and_act' doit être implémentée.")

    def apply_physics(self, *args):
        raise NotImplementedError("La méthode 'apply_physics' doit être implémentée.")
