import numpy as np
import matplotlib.pyplot as plt
from entites.drone import Drone

# --- Paramètres de Simulation ---
SIM_DURATION_S = 10.0   # Durée de la simulation en secondes
SIM_DT_S = 0.01         # Pas de temps (100 Hz)
NUM_STEPS = int(SIM_DURATION_S / SIM_DT_S)

# --- Initialisation ---
print("Initialisation de la simulation...")
# Crée un drone à la position [0, 0, 0]
drone1 = Drone(identifier="drone_01", initial_pos=np.array([0.0, 0.0, 0.0]), dt=SIM_DT_S)

# Définit un objectif : aller à (x=5, y=5, z=2) et y rester
target_pos = np.array([5.0, 5.0, 2.0])
drone1.set_target(target_pos)

# --- Pour stocker les données (votre futur "dataset") ---
log_p_true = []
log_p_measured = []
log_p_target = []

print(f"Lancement de la simulation ({NUM_STEPS} pas)...")

# --- Boucle de Simulation Principale ---
for step in range(NUM_STEPS):
    
    # Met à jour l'état du drone (Mesure -> Contrôle -> Physique)
    p_true, v_true, p_measured, accel = drone1.update_state()
    
    # Enregistre les données
    log_p_true.append(p_true)
    log_p_measured.append(p_measured)
    log_p_target.append(target_pos)
    
    # (Optionnel) Changer la cible à mi-chemin
    if step == NUM_STEPS // 2:
        print("Changement de cible !")
        target_pos = np.array([-5.0, 0.0, 3.0])
        drone1.set_target(target_pos)

print("Simulation terminée.")

# --- Analyse simple ---
log_p_true = np.array(log_p_true)
log_p_measured = np.array(log_p_measured)
log_p_target = np.array(log_p_target)

# Afficher les trajectoires (exemple pour X)
plt.figure()
plt.title("Trajectoire sur l'axe X")
plt.plot(log_p_true[:, 0], label="Position Réelle (X)")
plt.plot(log_p_measured[:, 0], 'x', markersize=2, label="Position Mesurée (X)")
plt.plot(log_p_target[:, 0], '--', label="Cible (X)")
plt.xlabel("Pas de temps")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)
plt.show()