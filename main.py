import sys
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config

def main():
    # 1. Charger la configuration depuis le fichier YAML
    config = load_config('config.yaml')
    if config is None:
        print("Échec du chargement de la configuration. Arrêt.")
        sys.exit(1)
        
    # 2. Initialiser le manager avec la configuration
    sim = SimulationManager(config)
    
    # 3. Lancer la simulation
    try:
        sim.run()
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    finally:
        sim.stop()

if __name__ == "__main__":
    main()
    