import sys
import time
import pybullet as p

from simulator.simulator_manager import SimulationManager
from utilities.config import load_config


def main():
    # 1. Charger la configuration
    config = load_config("config.yaml")
    if config is None:
        print("Échec du chargement de la configuration. Arrêt.")
        sys.exit(1)

    sim = SimulationManager(config)

    try:
        # 2. Lancer la simulation (boucle principale)
        sim.run()

        # 3. Si on est en GUI, garder la fenêtre ouverte
        mode_str = str(config["simulation"]["connect_mode"]).strip().lower()
        if mode_str == "gui":
            print("Simulation terminée. Ferme la fenêtre PyBullet pour quitter.")
            while p.isConnected(sim.physics_client_id):
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
