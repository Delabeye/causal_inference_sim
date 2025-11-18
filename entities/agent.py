import pybullet as p
import numpy as np

class Agent:
    """
    Classe de base abstraite pour toutes les entités autonomes (UAV, UGV).
    Un Agent est un objet physique qui possède ses propres capteurs et
    contrôleurs.
    """
    def __init__(self, urdf_path, start_pos, start_orn_q, 
                 physics_client_id, dt):
        
        self.p = p
        self.physics_client_id = physics_client_id
        self.dt = dt
        
        # 1. Charger le corps physique dans PyBullet
        # C'est le changement le plus important par rapport à votre ancien code 
        # L'agent charge son propre modèle [1, 12, 13, 14, 10, 3, 15]
        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
            physicsClientId=self.physics_client_id
        )
        print(f"Agent chargé avec bodyId: {self.bodyId}")

        # 2. Initialiser les composants (Capteurs, Estimateurs, Contrôleurs)
        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """
        Méthode fictive (placeholder) pour instancier les composants
        (GPSSensor, KalmanFilter, PIDController).
        Sera surchargée par les classes enfants (UAV, UGV).
        """
        pass

    def get_ground_truth_state(self):
        """
        Récupère l'état cinématique parfait (vérité terrain) de PyBullet.
        """
        pos, orn_q = self.p.getBasePositionAndOrientation(
            self.bodyId, self.physics_client_id
        )
        vel, ang_vel = self.p.getBaseVelocity(
            self.bodyId, self.physics_client_id
        )
        return {'pos': np.array(pos), 'orn_q': np.array(orn_q),
                'vel': np.array(vel), 'ang_vel': np.array(ang_vel)}

    def think_and_act(self, setpoint):
        """
        Méthode abstraite pour exécuter la boucle de contrôle.[4]
        Doit être implémentée par les classes enfants.
        """
        raise NotImplementedError("La méthode 'think_and_act' doit être implémentée.")

    def apply_physics(self, *args):
        """
        Méthode abstraite pour appliquer les forces/commandes de moteur.
        Doit être implémentée par les classes enfants.
        """
        raise NotImplementedError("La méthode 'apply_physics' doit être implémentée.")