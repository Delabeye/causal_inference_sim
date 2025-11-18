import pybullet as p
import numpy as np

from entities.agent import Agent


class UAV(Agent):
    """
    Drone avec contrôleur très simple :
      - modèle de masse ponctuelle
      - contrôle PD sur la position 3D
      - force appliquée au centre de masse
      - orientation figée (pas de rotation parasite)

    Objectif : aller de start_pos à setpoint proprement,
    sans finir au sol et sans tourner sur lui-même.
    """

    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.dt = dt
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "unnamed_uav")

        # ----------- Pose initiale -----------
        urdf_path = self.config["urdf_path"]

        start_pos = self.config.get("start_pos", [0.0, 0.0, 0.2])
        start_pos = [float(v) for v in start_pos]

        start_orn_euler = self.config.get("start_orn_euler", [0.0, 0.0, 0.0])
        start_orn_q = p.getQuaternionFromEuler(start_orn_euler)
        self.initial_orn_q = start_orn_q  # on garde l'orientation de référence

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
        # Gains plus doux pour limiter les oscillations
        self.Kp = np.array([1.0, 1.0, 3.0], dtype=float)   # X, Y, Z
        self.Kd = np.array([4.0, 4.0, 6.0], dtype=float)   # X, Y, Z

        # Limite de la force totale (en fonction du poids)
        self.F_max = 2.0 * self.mass * self.g  # jusqu'à ~2 g

        # Zone morte autour de la cible (pour éviter les tremblements)
        self.pos_tolerance = 0.05   # 5 cm
        self.vel_tolerance = 0.05   # 5 cm/s

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
          - tant qu'on est loin de la cible : PD complet (forces)
          - une fois proche et lent : on compense juste la gravité
          - orientation figée pour éviter que le drone tourne sur lui-même
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

        dist = np.linalg.norm(e_pos)
        speed = np.linalg.norm(vel)

        # 3. Si on est très proche et presque à l'arrêt -> juste compenser la gravité
        if dist < self.pos_tolerance and speed < self.vel_tolerance:
            F = np.array([0.0, 0.0, self.mass * self.g], dtype=float)
        else:
            # 4. Accélération désirée (sans gravité)
            a_cmd = self.Kp * e_pos + self.Kd * e_vel

            # 5. Ajout de la gravité sur Z
            a_total = a_cmd + np.array([0.0, 0.0, self.g], dtype=float)

            # 6. Force souhaitée
            F = self.mass * a_total

            # 7. Saturation de la force totale
            norm_F = np.linalg.norm(F)
            if norm_F > self.F_max:
                F *= self.F_max / (norm_F + 1e-9)

        # 8. Application de la force au centre de masse (base)
        if p.isConnected(self.physics_client_id):
            p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=-1,               # base
                forceObj=F.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client_id,
            )

            # 9. VERROUILLER L'ORIENTATION (pas de rotation)
            #    - on garde la position actuelle
            #    - on force l'orientation à la valeur initiale
            #    - on annule la vitesse angulaire
            cur_pos, cur_orn = p.getBasePositionAndOrientation(
                self.bodyId, physicsClientId=self.physics_client_id
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                self.bodyId, physicsClientId=self.physics_client_id
            )

            # fixe l'orientation
            p.resetBasePositionAndOrientation(
                self.bodyId,
                cur_pos,
                self.initial_orn_q,
                physicsClientId=self.physics_client_id,
            )

            # annule la rotation
            p.resetBaseVelocity(
                self.bodyId,
                linearVelocity=lin_vel,
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.physics_client_id,
            )

    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Avec ce contrôleur simple, toute la physique est gérée dans think_and_act(),
        donc cette méthode ne fait rien. On la garde pour compatibilité.
        """
        pass
