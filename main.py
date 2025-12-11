import time
import os
import yaml
import pybullet as p
import pybullet_data

# --- CORRECTION : On retire l'import de SimulatorManager qui ne sert plus ---
# from simulator.simulator_manager import SimulatorManager 

from entities.uav import UAV
from swarm.swarm import Swarm 

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Chargement Config
    cfg = load_config()
    
    # 2. Init PyBullet (Ceci remplace SimulatorManager)
    physics_client_mode = p.GUI if cfg["simulation"].get("physics_client") == "GUI" else p.DIRECT
    physics_client = p.connect(physics_client_mode)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, float(cfg["simulation"]["gravity"]))
    p.loadURDF("plane.urdf") # Sol

    # 3. Création des Agents (Drones)
    agents_list = []
    dt = float(cfg["simulation"]["dt"])
    
    for agent_cfg in cfg["agents"]:
        if agent_cfg["type"] == "uav":
            uav = UAV(agent_cfg, physics_client, dt)
            uav._initialize_components() # Init Capteurs/EKF
            agents_list.append(uav)

    # 4. Init Manager d'Essaim
    my_swarm = Swarm(agents_list)

    # 5. Boucle de Simulation
    sim_duration = 30.0 
    steps = int(sim_duration / dt)

    print("Début de la simulation SWARM...")
    
    try:
        for i in range(steps):
            # L'intelligence de l'essaim gère tous les drones
            my_swarm.run_step()
            
            p.stepSimulation()
            
            if physics_client_mode == p.GUI:
                time.sleep(dt) # Ralentir pour voir en temps réel

    except KeyboardInterrupt:
        print("Arrêt utilisateur.")
    except Exception as e:
        print(f"Erreur en cours de simulation : {e}")
    finally:
        try:
            p.disconnect()
        except:
            pass
        print("Simulation terminée.")

if __name__ == "__main__":
    main()