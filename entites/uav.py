import pybullet as p
import numpy as np
from entites import Agent # L'import vient de.agent_base
import utilities.config as config

# Importer vos définitions de capteurs 
from entites.sensor import GPSSensor

# --- PLACEHOLDERS POUR L'ÉTAPE 3 (maintenant ils lisent la config) ---
class PlaceholderKalmanFilter:
    def __init__(self, config):
        self.x = np.zeros(6)
        self.process_noise = config['process_noise'] # Lit depuis la config
    def predict(self): pass
    def update(self, z): self.x = np.array(z, z[1], z[2])


# --- FIN DES PLACEHOLDERS ---

class UAV(Agent):
    """
    Implémentation d'un agent UAV.
    Tous ses paramètres sont lus depuis son objet 'config'.
    """
    def __init__(self, config, physics_client_id, dt):
        
        self.config = config # Stocke son propre bloc de config
        
        # Initialise AgentBase avec les valeurs de la config
        super().__init__(
            urdf_path=config['urdf_path'],
            start_pos=config['start_pos'],
            start_orn_q=p.getQuaternionFromEuler(config['start_orn_euler']), 
            physics_client_id=physics_client_id, 
            dt=dt
        )
        
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

    def _initialize_components(self):
        """
        Surcharge pour créer les composants de l'UAV en lisant son bloc config.
        """
        components_cfg = self.config['components']
        
        # 1. Capteur (lit la config 'gps') 
        self.components['gps'] = GPSSensor(components_cfg['gps'])
        
        # 2. Estimateur (lit la config 'estimator') [Image 1]
        self.components['estimator'] = PlaceholderKalmanFilter(components_cfg['estimator'])
        
        # 3. Contrôleur (lit la config 'controller_z') [Image 1]
        self.components['pid_z'] = PlaceholderPIDController(components_cfg['controller_z'])
        
        print(f"Composants pour '{self.config['name']}' (ID: {self.bodyId}) initialisés.")


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