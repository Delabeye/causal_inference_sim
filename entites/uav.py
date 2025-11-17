import pybullet as p
import numpy as np
from.agent import Agent # L'import vient de.agent_base

# Importer vos définitions de capteurs 
from entites.sensor import GPSSensor

# --- PLACEHOLDERS POUR L'ÉTAPE 3 (maintenant ils lisent la config) ---
class PlaceholderKalmanFilter:
<<<<<<< HEAD
    """Stub pour le filtre de Kalman de l'Étape 3."""
    def __init__(self):
        self.x = np.zeros(6) # [pos, vel]
    def predict(self): 
        pass
    def update(self, z): self.x = np.array()

class PlaceholderPIDController:
    """Stub pour le contrôleur PID de l'Étape 3."""
    def __init__(self, Kp, Ki, Kd, output_min, output_max, windup):
        
        self.Kp = Kp # N/m
        self.Ki = Ki # N/(m·s)
        self.Kd = Kd # N·s/m
        self.I = 0.0
        self.P = 0.0
        self.D = 0.0    
        self.prev_error = 0.0

        # Anti-windup limits
        self.output_min = output_min
        self.output_max = output_max
        self.windup = windup
    
    def compute(self, error, dt):
        self.P = self.Kp * error
        
        self.I += self.Ki * error * dt
        self.windup_guard()

        self.D = self.Kd * (error / dt) 

        Output = self.P + self.I + self.D

        return Output
    
    def windup_guard(self):

        if self.windup != 0:
            if self.I > self.output_max:
                self.I = self.output_max
            elif self.I < -self.output_min:
                self.I = -self.output_min



=======
    def __init__(self, config):
        self.x = np.zeros(6)
        self.process_noise = config['process_noise'] # Lit depuis la config
    def predict(self): pass
    def update(self, z): self.x = np.array(z, z[1], z[2])

class PlaceholderPIDController:
    def __init__(self, config):
        self.Kp = config['gains']['Kp'] # Lit depuis la config
        self.Ki = config['gains']['Ki']
        self.Kd = config['gains']['Kd']
    def compute(self, error, dt): return self.Kp * error
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec
# --- FIN DES PLACEHOLDERS ---


class UAV(Agent):
    """
    Implémentation d'un agent UAV.
    Tous ses paramètres sont lus depuis son objet 'config'.
    """
<<<<<<< HEAD
    def __init__(self, urdf_path, start_pos, start_orn_q, 
                 physics_client_id, dt, mass):
=======
    def __init__(self, config, physics_client_id, dt):
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec
        
        self.config = config # Stocke son propre bloc de config
        
<<<<<<< HEAD
        # Paramètres physiques (doivent être ajustés pour votre URDF)
        # Basé sur gym-pybullet-drones [7, 16, 3]
        self.thrust_coeff = 2.2e-8  # N / (RPM^2)
        self.max_rpm = 25000
        self.mass = mass # Masse (kg) - À AJUSTER
=======
        # Initialise AgentBase avec les valeurs de la config
        super().__init__(
            urdf_path=config['urdf_path'],
            start_pos=config['start_pos'],
            start_orn_q=p.getQuaternionFromEuler(config['start_orn_euler']), 
            physics_client_id=physics_client_id, 
            dt=dt
        )
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec
        
        # --- Paramètres physiques (lus depuis la config) ---
        physics_cfg = self.config['physics']
        self.thrust_coeff = physics_cfg['thrust_coeff']
        self.max_rpm = physics_cfg['max_rpm']
        self.motor_link_indices = physics_cfg['motor_link_indices']
        
        self.mass = p.getDynamicsInfo(self.bodyId, -1, physicsClientId=self.physics_client_id)
        if self.mass <= 0: self.mass = 0.027 # Fallback
            
        self.g = 9.81
        self.hover_thrust_per_motor = (self.mass * self.g) / 4.0
        self.hover_rpm = np.sqrt(self.hover_thrust_per_motor / self.thrust_coeff)
        
        self.last_rpms = np.zeros(4)

<<<<<<< HEAD
    def _initialize_components(self, position_noise_std, velocity_noise_std, Kp, Ki, Kd):
        """Surcharge pour créer les composants réels de l'UAV."""
        
        # 1. Capteur (de votre code existant )
        self.components['gps'] = GPSSensor(position_noise_std, 
                                           velocity_noise_std)
=======
    def _initialize_components(self):
        """
        Surcharge pour créer les composants de l'UAV en lisant son bloc config.
        """
        components_cfg = self.config['components']
        
        # 1. Capteur (lit la config 'gps') 
        self.components['gps'] = GPSSensor(components_cfg['gps'])
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec
        
        # 2. Estimateur (lit la config 'estimator') [Image 1]
        self.components['estimator'] = PlaceholderKalmanFilter(components_cfg['estimator'])
        
<<<<<<< HEAD
        # 3. Contrôleur (bloc 'controller' de [4])
        # TODO: Remplacer par la vraie classe PIDController de l'Étape 3
        # Nous créons un contrôleur P-I-D juste pour l'altitude (axe Z)
        self.components['pid_z'] = PlaceholderPIDController(Kp, Ki, Kd)
        
        print(f"Composants de l'UAV {self.identifier} initialisés (GPS, Estimateur Stub, PID Stub).")
=======
        # 3. Contrôleur (lit la config 'controller_z') [Image 1]
        self.components['pid_z'] = PlaceholderPIDController(components_cfg['controller_z'])
        
        print(f"Composants pour '{self.config['name']}' (ID: {self.bodyId}) initialisés.")
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec


    def think_and_act(self, setpoint):
        """
        Exécute la boucle de contrôle complète [Image 1] pour le drone.
        """
        # 1. PERCEPTION [19, 20, 21, 22, 9]
        true_state = self.get_ground_truth_state()
        
        # 2. SENSING 
        noisy_pos, noisy_vel = self.components['gps'].measure(
            true_state['pos'], true_state['vel']
        )
        
        # 3. ESTIMATION [Image 1]
        estimator = self.components['estimator']
        estimator.predict()
        estimator.update(noisy_pos)
        estimated_pos = estimator.x[0:3]
        
        # 4. CONTRÔLE [Image 1]
        error_z = setpoint[2] - estimated_pos[2]
        pid_z = self.components['pid_z']
<<<<<<< HEAD
        # La sortie est une "force" de correction
        # Un simple contrôleur P pour cet exemple
=======
>>>>>>> c4a9312370d54549b783e637ee085acc81f8c1ec
        correction_force = pid_z.compute(error_z, self.dt)
        
        total_thrust = self.hover_thrust_per_motor * 4 + correction_force
        thrust_per_motor = max(0, total_thrust / 4.0)
        
        target_rpm = np.sqrt(thrust_per_motor / self.thrust_coeff) [3, 4, 6]
        target_rpm = min(target_rpm, self.max_rpm)
        
        self.last_rpms = np.array([target_rpm] * 4)
        
        # 5. ACTIONNEMENT
        self.apply_physics(self.last_rpms)

    def apply_physics(self, rpms):
        """
        Applique les forces de poussée (thrust) aux liens moteurs dans PyBullet.
        """
        for i, motor_link_index in enumerate(self.motor_link_indices):
            
            thrust = self.thrust_coeff * (rpms[i]**2)
            
            self.p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=motor_link_index,
                forceObj='',
                posObj='',
                flags=self.p.LINK_FRAME, # Crucial pour le contrôle [20, 23, 8, 24, 9, 13]
                physicsClientId=self.physics_client_id
            )