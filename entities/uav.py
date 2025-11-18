import pybullet as p
import numpy as np

from entities.agent import Agent
from entities.sensor import GPSSensor


# ----------------------------------------------------------------------
# PID minimal interne (on n'utilise plus Control.PID)
# ----------------------------------------------------------------------
class PIDController:
    def __init__(self, config: dict):
        """
        config:
          gains: {Kp: ..., Ki: ..., Kd: ...}
          windup: valeur max de l'intégrale (optionnel)
        """
        gains = config.get("gains", {})
        self.Kp = float(gains.get("Kp", 0.0))
        self.Ki = float(gains.get("Ki", 0.0))
        self.Kd = float(gains.get("Kd", 0.0))

        self.integral = 0.0
        self.prev_error = 0.0
        self.windup = float(config.get("windup", 0.0))  # 0.0 => pas de clamp

    def compute(self, error: float, dt: float) -> float:
        # Intégrale
        self.integral += error * dt
        if self.windup > 0.0:
            self.integral = max(-self.windup, min(self.windup, self.integral))

        # Dérivée
        derivative = (error - self.prev_error) / dt if dt > 0.0 else 0.0
        self.prev_error = error

        # Sortie PID
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# ----------------------------------------------------------------------
# Kalman fictif (placeholder)
# ----------------------------------------------------------------------
class PlaceholderKalmanFilter:
    def __init__(self, config: dict):
        # état [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=float)
        self.process_noise = float(config.get("process_noise", 0.0))

    def predict(self):
        # Stub : ne fait rien pour l'instant
        pass

    def update(self, z):
        """
        z : position mesurée (np.array de taille 3)
        Pour l'instant on copie juste la position dans l'état.
        """
        z = np.asarray(z, dtype=float)
        self.x[0:3] = z
        # vitesses laissées à 0


# ----------------------------------------------------------------------
# UAV (drone)
# ----------------------------------------------------------------------
class UAV(Agent):
    """
    Implémentation d'un agent UAV (drone).
    Tous ses paramètres sont lus depuis son bloc 'config'.
    """

    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config  # stocker le bloc de config complet

        # ----------- Lecture des paramètres de pose -----------
        urdf_path = self.config["urdf_path"]

        start_pos = self.config.get("start_pos")
        if start_pos is None:
            start_pos = [0.0, 0.0, 0.2]
        start_pos = [float(v) for v in start_pos]

        start_orn_euler = self.config.get("start_orn_euler")
        if start_orn_euler is None:
            start_orn_euler = [0.0, 0.0, 0.0]
        start_orn_q = p.getQuaternionFromEuler(start_orn_euler)

        # ----------- Appel du constructeur de Agent -----------
        super().__init__(
            urdf_path=urdf_path,
            start_pos=start_pos,
            start_orn_q=start_orn_q,
            physics_client_id=physics_client_id,
            dt=dt,
        )

        self.dt = dt
        self.physics_client_id = physics_client_id

        # ----------- Paramètres physiques du drone -----------
        physics_cfg = self.config.get("physics", {})

        self.thrust_coeff = float(physics_cfg.get("thrust_coeff", 2.2e-8))
        self.max_rpm = float(physics_cfg.get("max_rpm", 25000.0))

        raw_indices = physics_cfg.get("motor_link_indices")
        if raw_indices is None:
            self.motor_link_indices = []
        else:
            self.motor_link_indices = list(raw_indices)

        # Masse issue de l'URDF
        dyn = p.getDynamicsInfo(
            self.bodyId,
            -1,
            physicsClientId=self.physics_client_id,
        )
        mass = dyn[0] if dyn is not None else None

        if mass is None or mass <= 0:
            mass = 0.027  # masse par défaut

        self.mass = float(mass)

        self.g = 9.81
        self.hover_thrust_per_motor = (self.mass * self.g) / 4.0
        self.hover_rpm = np.sqrt(self.hover_thrust_per_motor / self.thrust_coeff)

        # Dernières consignes moteurs
        self.last_rpms = np.zeros(4, dtype=float)

        print(
            f"UAV '{self.config.get('name', 'unnamed')}' "
            f"chargé, bodyId={self.bodyId}, masse={self.mass}"
        )

    # ------------------------------------------------------------------
    # Initialisation des composants (capteurs, estimateurs, contrôleurs)
    # ------------------------------------------------------------------
    def _initialize_components(self):
        """
        Surcharge pour créer les composants de l'UAV en lisant son bloc config.
        Appelée automatiquement par Agent.__init__().
        """
        components_cfg = self.config.get("components", {})

        # 1. Capteur GPS
        gps_cfg = components_cfg.get("gps", {})
        self.components["gps"] = GPSSensor(gps_cfg)

        # 2. Estimateur (Kalman fictif)
        est_cfg = components_cfg.get("estimator", {})
        self.components["estimator"] = PlaceholderKalmanFilter(est_cfg)

        # 3. Contrôleur PID sur l'axe Z
        pid_z_cfg = components_cfg.get("controller_z", {})
        self.components["pid_z"] = PIDController(pid_z_cfg)

        print(
            f"Composants pour '{self.config.get('name', 'unnamed')}' "
            f"(ID: {self.bodyId}) initialisés."
        )

    # ------------------------------------------------------------------
    # Boucle de décision + action
    # ------------------------------------------------------------------
    def think_and_act(self, setpoint: np.ndarray):
        """
        Exécute la boucle de contrôle en Z pour le drone.
        setpoint : [x, y, z] (on n'utilise ici que z)
        """
        # 1. Vérité terrain
        true_state = self.get_ground_truth_state()
        true_pos = true_state["pos"]
        true_vel = true_state["vel"]

        # 2. Mesure bruitée via GPS
        gps: GPSSensor = self.components["gps"]
        noisy_pos, noisy_vel = gps.measure(true_pos, true_vel)

        # 3. Estimation
        estimator: PlaceholderKalmanFilter = self.components["estimator"]
        estimator.predict()
        estimator.update(noisy_pos)
        estimated_pos = estimator.x[0:3]

        # 4. Contrôle en Z
        desired_z = float(setpoint[2])
        error_z = desired_z - float(estimated_pos[2])

        pid_z: PIDController = self.components["pid_z"]
        correction_force = pid_z.compute(error_z, self.dt)

        # 5. Traduction en poussée globale puis par moteur
        total_thrust = self.hover_thrust_per_motor * 4.0 + correction_force
        thrust_per_motor = max(0.0, total_thrust / 4.0)

        target_rpm = np.sqrt(thrust_per_motor / self.thrust_coeff)
        target_rpm = min(target_rpm, self.max_rpm)

        n_motors = len(self.motor_link_indices) if self.motor_link_indices else 4
        self.last_rpms = np.full(n_motors, target_rpm, dtype=float)

        # 6. Application de la physique
        self.apply_physics(self.last_rpms)

    # ------------------------------------------------------------------
    # Application des forces de poussée aux moteurs
    # ------------------------------------------------------------------
    def apply_physics(self, rpms: np.ndarray):
        """
        Applique les forces de poussée (thrust) aux liens moteurs dans PyBullet.
        """
        if not self.motor_link_indices:
            return

        num_joints = p.getNumJoints(self.bodyId, physicsClientId=self.physics_client_id)

        for i, motor_link_index in enumerate(self.motor_link_indices):
            if i >= len(rpms):
                break

            if motor_link_index < 0 or motor_link_index >= num_joints:
                continue

            thrust = self.thrust_coeff * (rpms[i] ** 2)

            p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=motor_link_index,
                forceObj=[0.0, 0.0, float(thrust)],
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME,
                physicsClientId=self.physics_client_id,
            )
