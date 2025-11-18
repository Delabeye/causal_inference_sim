class PIDController:
    """Contrôleur PID simple pour l'altitude (ou autre)."""

    def __init__(self, output_min, output_max, config):
        # Gains PID
        self.Kp = config["gains"]["Kp"]
        self.Ki = config["gains"]["Ki"]
        self.Kd = config["gains"]["Kd"]

        # États internes
        self.I = 0.0
        self.P = 0.0
        self.D = 0.0
        self.prev_error = 0.0

        # Anti-windup limits (valeurs POSITIVES)
        # L'intégrale sera bornée dans [-output_min, output_max]
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.windup = config.get("windup", None)

    def compute(self, error, dt):
        # Proportionnel
        self.P = self.Kp * error

        # Intégral
        self.I += self.Ki * error * dt
        self.windup_guard()

        # Dérivé (sur l'erreur, pas juste error/dt)
        if dt > 0.0:
            self.D = self.Kd * ((error - self.prev_error) / dt)
        else:
            self.D = 0.0

        self.prev_error = error

        output = self.P + self.I + self.D
        return output

    def windup_guard(self):
        if self.windup:
            # clamp de l'intégrale entre [-output_min, output_max]
            if self.I > self.output_max:
                self.I = self.output_max
            elif self.I < -self.output_min:
                self.I = -self.output_min
