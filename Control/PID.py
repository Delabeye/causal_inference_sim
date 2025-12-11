# Control/PID.py

import math


class PIDController:
    """
    Contrôleur PID basique, paramétré par un bloc de config :

      gains:
        Kp: ...
        Ki: ...
        Kd: ...

      windup: valeur non nulle => anti-windup activé
      output_min / output_max : bornes pour l'intégrale (anti-windup).

    Le contrôleur est volontairement générique : il ne sait pas s'il contrôle
    une position, une vitesse, un angle... il ne voit que "error".
    """

    def __init__(self, output_min, output_max, config: dict):
        gains = config.get("gains", {})
        self.Kp = float(gains.get("Kp", 0.0))
        self.Ki = float(gains.get("Ki", 0.0))
        self.Kd = float(gains.get("Kd", 0.0))

        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.prev_error = 0.0

        # Bornes pour l'intégrale / la sortie
        self.output_min = float(output_min)
        self.output_max = float(output_max)

        # windup peut être 0/1, True/False, etc.
        self.windup = bool(config.get("windup", 0))

    # ------------------------------------------------------------------
    def reset(self):
        """Remet le PID à zéro (utile quand on est arrivé à la cible)."""
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.prev_error = 0.0

    # ------------------------------------------------------------------
    def compute(self, error: float, dt: float) -> float:
        """
        Calcule la sortie du PID pour un échantillon.

        error : consigne - mesure
        dt    : pas de temps
        """
        if dt <= 0.0:
            dt = 1e-6

        # Proportionnel
        self.P = self.Kp * error

        # Intégral + anti-windup
        self.I += self.Ki * error * dt
        self._windup_guard()

        # Dérivé (sur l'erreur)
        d_err = error - self.prev_error
        self.D = self.Kd * (d_err / dt)
        self.prev_error = error

        # Sortie totale
        return self.P + self.I + self.D

    # ------------------------------------------------------------------
    def _windup_guard(self):
        """Limite l'intégrale si l'anti-windup est activé."""
        if not self.windup:
            return

        if self.I > self.output_max:
            self.I = self.output_max
        elif self.I < -self.output_min:
            self.I = -self.output_min


class AnglePIDController(PIDController):
    """
    PID spécialisé pour les ANGLES (en radian).

    Avant d'appeler la logique PID de base, on "wrap" l'erreur dans [-pi, pi]
    pour éviter les sauts à 2*pi près (ex : -179° -> +179°).
    Parfait pour contrôler un yaw autour de z.
    """

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute(self, error: float, dt: float) -> float:
        # on ramène l'erreur dans [-pi, pi]
        wrapped_error = self._wrap_angle(error)
        return super().compute(wrapped_error, dt)
    
    