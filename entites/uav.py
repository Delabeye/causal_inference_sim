import pybullet as p
import numpy as np
from agent import Agent

# Importer vos définitions de capteurs existantes 
from entites.sensor import GPSSensor

# --- PLACEHOLDERS POUR L'ÉTAPE 3 ---
# Normalement, ceux-ci seraient dans components/estimators.py et components/controllers.py
# Nous les mettons ici temporairement pour que le code de l'Étape 2 puisse s'importer.
class PlaceholderKalmanFilter:
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



# --- FIN DES PLACEHOLDERS ---


class UAV(Agent):
    """
    Implémentation d'un agent UAV (Quadricoptère) pour PyBullet.
    Cette classe REMPLACE votre ancienne classe Drone.
    """
    def __init__(self, urdf_path, start_pos, start_orn_q, 
                 physics_client_id, dt, mass):
        
        super().__init__(urdf_path, start_pos, start_orn_q, 
                         physics_client_id, dt)
        
        # Paramètres physiques (doivent être ajustés pour votre URDF)
        # Basé sur gym-pybullet-drones [7, 16, 3]
        self.thrust_coeff = 2.2e-8  # N / (RPM^2)
        self.max_rpm = 25000
        self.mass = mass # Masse (kg) - À AJUSTER
        
        # Gravité pour la compensation (hover)
        self.g = 9.81
        self.hover_thrust_per_motor = (self.mass * self.g) / 4.0
        self.hover_rpm = np.sqrt(self.hover_thrust_per_motor / self.thrust_coeff)
        
        # Indices des liens moteurs de l'URDF (C'EST LA PARTIE DÉLICATE)
        # Vous DEVEZ trouver les bons indices de liens pour vos moteurs/hélices
        # dans votre fichier URDF. Ceci est un EXEMPLE.
        # 0, 1, 2, 3 sont souvent les indices des liens moteurs/hélices
        self.motor_link_indices = '' # [prop0, prop1, prop2, prop3]
        if len(self.motor_link_indices)!= 4:
            print("ERREUR : L'URDF ne correspond pas, les indices des moteurs sont incorrects.")
            
        self.last_rpms = np.zeros(4)

    def _initialize_components(self, position_noise_std, velocity_noise_std, Kp, Ki, Kd):
        """Surcharge pour créer les composants réels de l'UAV."""
        
        # 1. Capteur (de votre code existant )
        self.components['gps'] = GPSSensor(position_noise_std, 
                                           velocity_noise_std)
        
        # 2. Estimateur (Filtre - bloc 'sensor filter' de [4])
        # TODO: Remplacer par la vraie classe KalmanFilter de l'Étape 3
        self.components['estimator'] = PlaceholderKalmanFilter()
        
        # 3. Contrôleur (bloc 'controller' de [4])
        # TODO: Remplacer par la vraie classe PIDController de l'Étape 3
        # Nous créons un contrôleur P-I-D juste pour l'altitude (axe Z)
        self.components['pid_z'] = PlaceholderPIDController(Kp, Ki, Kd)
        
        print(f"Composants de l'UAV {self.identifier} initialisés (GPS, Estimateur Stub, PID Stub).")


    def think_and_act(self, setpoint):
        """
        Exécute la boucle de contrôle complète [4] pour le drone.
        """
        # 1. PERCEPTION : Obtenir la vérité terrain de PyBullet
        true_state = self.get_ground_truth_state()
        
        # 2. SENSING : (Bloc 'sensor camera' [4])
        # Obtenir une mesure bruitée de notre composant GPSSensor 
        noisy_pos, noisy_vel = self.components['gps'].measure(
            true_state['pos'], true_state['vel']
        )
        
        # 3. ESTIMATION : (Bloc 'sensor filter' [4])
        estimator = self.components['estimator']
        estimator.predict()
        estimator.update(noisy_pos)
        # Pour cet exemple, trichons et utilisons la vérité terrain
        estimated_pos = true_state['pos']
        
        # 4. CONTRÔLE : (Bloc 'controller' et Sommation [4])
        
        # Calculer l'erreur (Consigne - État Estimé)
        # Nous ne contrôlons que l'altitude (Z) pour l'instant
        error_z = setpoint[1] - estimated_pos[1]
        
        # Calculer la commande (PID)
        pid_z = self.components['pid_z']
        # La sortie est une "force" de correction
        # Un simple contrôleur P pour cet exemple
        correction_force = pid_z.compute(error_z, self.dt)
        
        # Convertir la force en RPM pour les 4 moteurs
        # Commande totale = compensation de gravité + correction PID
        total_thrust = self.hover_thrust_per_motor * 4 + correction_force
        
        # Distribuer aux 4 moteurs
        thrust_per_motor = total_thrust / 4.0
        
        # Gérer la saturation
        thrust_per_motor = max(0, thrust_per_motor)
        
        # Poussée = K * RPM^2  =>  RPM = sqrt(Poussée / K) [7, 16, 17]
        target_rpm = np.sqrt(thrust_per_motor / self.thrust_coeff)
        target_rpm = min(target_rpm, self.max_rpm) # Plafonner
        
        self.last_rpms = np.array([target_rpm] * 4)
        
        # 5. ACTIONNEMENT : Appliquer la physique
        self.apply_physics(self.last_rpms)

    def apply_physics(self, rpms):
        """
        Applique les forces de poussée (thrust) aux liens moteurs dans PyBullet.
        """
        for i, motor_link_index in enumerate(self.motor_link_indices):
            
            # Calculer la force de poussée [7, 16, 17]
            thrust = self.thrust_coeff * (rpms[i]**2)
            
            # Appliquer la force dans le REPÈRE DU LIEN (LINK_FRAME)
            # La force est [0, 0, thrust] car elle pousse "vers le haut"
            # par rapport au moteur lui-même.
            self.p.applyExternalForce(
                objectUniqueId=self.bodyId,
                linkIndex=motor_link_index,
                forceObj=[0, 0, thrust],    # Force dans le repère local
                posObj='',          # Position dans le repère local
                flags=self.p.LINK_FRAME, # C'EST LE POINT LE PLUS IMPORTANT [5, 6, 7, 8, 9]
                physicsClientId=self.physics_client_id)