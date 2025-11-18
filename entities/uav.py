import pybullet as p
import numpy as np

from entities.agent import Agent


class UAV(Agent):
    """
    Drone avec contrôleur très simple :
      - on considère le drone comme un point de masse
      - on fait un contrôle PD sur la position 3D
      - on applique une force globale au centre de masse

    Résultat : il décolle et va proprement de start_pos à setpoint,
    sans finir au sol ni partir en vrille.
    """

    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = dt
        self.physics_client_id = physics_client_id

        # ----------- Pose initiale -----------
        urdf_path = self.config["urdf_path"]

        start_pos = self.config.get("start_pos", [0.0, 0.0, 0.2])
        start_pos = [float(v) for v in start_pos]

        start_orn_euler = self.config.get("start_orn_euler", [0.0, 0.0, 0.0])
        start_orn_q = p.getQuaternionFromEuler(start_orn_euler)

        # Charge l'URDF via la classe Agent
        super().__init__(
            urdf_path=urdf_path,
            start_pos=start_pos,
            start_orn_q=start_orn_q,
            physics_client_id=self.physics_client_id,
            dt=dt,
        )

        # ----------- Paramètres physiques -----------
        self.g = 9.81

        # Masse totale = base + tous les liens
        num_joints = p.getNumJoints(self.bodyId, physicsClientId=self.physics_client_id)
        total_mass = p.getDynamicsInfo(
            self.bodyId, -1, physicsClientId=self.physics_client_id
        )[0] or 0.0
        for j in range(num_joints):
            mj = p.getDynamicsInfo(
                self.bodyId, j, physicsClientId=self.physics_client_id
            )[0]
            if mj is not None:
                total_mass += mj

        if total_mass <= 0.0:
            total_mass = 0.03  # fallback si URDF bizarre

        self.mass = float(total_mass)

        # ----------- Gains du contrôleur PD -----------
        # Même gains pour tous les drones ; tu peux les passer en config si tu veux
        self.Kp = np.array([2.0, 2.0, 6.0], dtype=float)  # position
        self.Kd = np.array([3.0, 3.0, 5.0], dtype=float)  # vitesse

        # Limite de la force totale (en fonction du poids)
        self.F_max = 3.0 * self.mass * self.g  # jusqu'à ~3 g

        # Position cible
        self.target_pos = np.array(
            self.config.get("setpoint", start_pos), dtype=float
        )

        print(
            f"UAV '{self.config.get('name', 'unnamed')}' chargé, "
            f"bodyId={self.bodyId}, masse_totale={self.mass:.4f} kg"
        )

    # ------------------------------------------------------------------
    def _initialize_components(self):
        """Pas de capteurs/estimateurs séparés pour ce contrôleur simple."""
        self.components = {}

    # ------------------------------------------------------------------
    def set_target_pos(self, target):
        """Changer la cible pendant la simu (optionnel)."""
        self.target_pos = np.array(target, dtype=float)

    # ------------------------------------------------------------------
    def think_and_act(self, setpoint: np.ndarray | None = None):
        """
        Contrôle PD très simple sur la position :
          - erreur de position : e = x* - x
          - erreur de vitesse : ev = 0 - v
          - accel désirée : a = Kp*e + Kd*ev
          - force : F = m*a + compensation gravité en Z
        """
        if setpoint is not None:
            self.set_target_pos(setpoint)

        # 1. Vérité terrain
        state = self.get_ground_truth_state()
        pos = state["pos"]    # [x, y, z]
        vel = state["vel"]    # [vx, vy, vz]

        # 2. Erreurs
        e_pos = self.target_pos - pos        # erreur de position
        e_vel = -vel                         # on veut v = 0

        # 3. Accélération désirée (sans gravité)
        a_cmd = self.Kp * e_pos + self.Kd * e_vel

        # 4. Ajout de la gravité sur Z
        a_total = a_cmd + np.array([0.0, 0.0, self.g], dtype=float)

        # 5. Force souhaitée
        F = self.mass * a_total

        # 6. Saturation de la force totale pour éviter les coups de bélier
        norm_F = np.linalg.norm(F)
        if norm_F > self.F_max:
            F *= self.F_max / (norm_F + 1e-9)

        # 7. Application de la force au centre de masse (base)
        if p.isConnected(self.physics_client_id):
            p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=-1,               # base
                forceObj=F.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client_id,
            )

    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Avec ce contrôleur simple, toute la physique est gérée dans think_and_act(),
        donc cette méthode ne fait rien. On la garde pour compatibilité.
        """
        pass
