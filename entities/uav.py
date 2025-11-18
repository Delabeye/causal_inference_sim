import pybullet as p
import numpy as np

from entities.agent import Agent
from Control.PID import PIDController


class UAV(Agent):
    """
    Drone avec contrôleur simple à base de PID sur la position :
      - 3 PID (x, y, z) configurés dans config.yaml
      - sortie des PID = accélération désirée
      - force = m * (a_cmd + gravité) appliquée au centre de masse
      - orientation figée (pas de rotation parasite)
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
        self.initial_orn_q = start_orn_q  # orientation de référence

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

        # ----------- PID à partir du YAML -----------
        components_cfg = self.config.get("components", {})

        def get_pid_cfg(name: str, fallback: dict | None = None) -> dict:
            if name in components_cfg:
                return components_cfg[name]
            if fallback is not None:
                return fallback
            # config par défaut si rien n'est défini
            return {
                "gains": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
                "windup": 0.0,
            }

        cfg_z = get_pid_cfg("controller_z")
        cfg_x = get_pid_cfg("controller_x", fallback=cfg_z)
        cfg_y = get_pid_cfg("controller_y", fallback=cfg_z)

        # limites pour l'intégrale (et comme référence de saturation)
        self.acc_limit_xy = 5.0   # m/s^2 max en x/y
        self.acc_limit_z  = 5.0   # m/s^2 max en z

        self.pid_x = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_x)
        self.pid_y = PIDController(self.acc_limit_xy, self.acc_limit_xy, cfg_y)
        self.pid_z = PIDController(self.acc_limit_z,  self.acc_limit_z,  cfg_z)

        # Limite de la force totale (en fonction du poids)
        self.F_max = 2.5 * self.mass * self.g  # jusqu'à ~2.5 g

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
        Contrôle de position avec PID (x, y, z) :
          - tant qu'on est loin de la cible : on utilise les PID
          - une fois proche et lent : on compense juste la gravité
          - orientation figée pour éviter que le drone tourne sur lui-même
          - si PyBullet est déconnecté -> on ne fait rien
        """
        # Si le serveur n'est plus connecté, on ne fait rien
        if not p.isConnected(self.physics_client_id):
            return

        if setpoint is not None:
            self.set_target_pos(setpoint)

        try:
            # 1. Vérité terrain
            state = self.get_ground_truth_state()
            pos = state["pos"]    # [x, y, z]
            vel = state["vel"]    # [vx, vy, vz]
        except p.error:
            return

        # 2. Erreurs
        e_pos = self.target_pos - pos        # erreur de position
        e_vel = -vel                         # v_cible = 0
        dist = np.linalg.norm(e_pos)
        speed = np.linalg.norm(vel)

        # 3. Zone morte : si on est arrivé et quasi immobile
        if dist < self.pos_tolerance and speed < self.vel_tolerance:
            # On remet les PID à zéro pour éviter l'accumulation
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            # Force uniquement pour tenir en l'air
            F = np.array([0.0, 0.0, self.mass * self.g], dtype=float)
        else:
            # 4. PID par axe -> accélérations désirées
            ex, ey, ez = e_pos

            ax_cmd = self.pid_x.compute(float(ex), self.dt)
            ay_cmd = self.pid_y.compute(float(ey), self.dt)
            az_cmd = self.pid_z.compute(float(ez), self.dt)

            # Saturation des accélérations
            ax_cmd = float(np.clip(ax_cmd, -self.acc_limit_xy, self.acc_limit_xy))
            ay_cmd = float(np.clip(ay_cmd, -self.acc_limit_xy, self.acc_limit_xy))
            az_cmd = float(np.clip(az_cmd, -self.acc_limit_z,  self.acc_limit_z))

            # 5. Ajout de la gravité sur Z
            a_total = np.array([ax_cmd, ay_cmd, az_cmd + self.g], dtype=float)

            # 6. Force souhaitée
            F = self.mass * a_total

            # 7. Saturation de la force totale
            norm_F = np.linalg.norm(F)
            if norm_F > self.F_max:
                F *= self.F_max / (norm_F + 1e-9)

        # 8. Application de la force + verrouillage orientation
        if not p.isConnected(self.physics_client_id):
            return

        try:
            # appliquer la force au centre de masse
            p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=-1,               # base
                forceObj=F.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client_id,
            )

            # Verrouiller l'orientation + annuler la rotation
            cur_pos, cur_orn = p.getBasePositionAndOrientation(
                self.bodyId, physicsClientId=self.physics_client_id
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                self.bodyId, physicsClientId=self.physics_client_id
            )

            p.resetBasePositionAndOrientation(
                self.bodyId,
                cur_pos,
                self.initial_orn_q,
                physicsClientId=self.physics_client_id,
            )

            p.resetBaseVelocity(
                self.bodyId,
                linearVelocity=lin_vel,
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.physics_client_id,
            )
        except p.error:
            return

    # ------------------------------------------------------------------
    def apply_physics(self, *args, **kwargs):
        """
        Avec ce contrôleur simple, toute la physique est gérée dans think_and_act(),
        donc cette méthode ne fait rien. On la garde pour compatibilité.
        """
        pass
