class PlaceholderPIDController:
    """Stub pour le contrôleur PID de l'Étape 3."""
    def __init__(self, output_min, output_max, config):
        
        self.Kp = config['gains']['Kp'] # Lit depuis la config
        self.Ki = config['gains']['Ki'] # Lit depuis la config
        self.Kd = config['gains']['Kd'] # Lit depuis la config
        self.I = 0.0
        self.P = 0.0
        self.D = 0.0    
        self.prev_error = 0.0

        # Anti-windup limits
        self.output_min = output_min
        self.output_max = output_max
        self.windup = config['windup'] # Lit depuis la config
    
    def compute(self, error, dt):
        self.P = self.Kp * error
        
        self.I += self.Ki * error * dt
        self.windup_guard()

        self.D = self.Kd * (error / dt) 

        Output = self.P + self.I + self.D

        return Output
    
    def windup_guard(self):

        if self.windup:
            if self.I > self.output_max:
                self.I = self.output_max
            elif self.I < -self.output_min:
                self.I = -self.output_min
