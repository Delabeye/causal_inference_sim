import sys
import time

import pybullet as p

from simulator.simulator_manager import SimulationManager
from utilities.config import load_config


def main():
    # 1. Charger la configuration depuis le fichier YAML
    config = load_config("config.yaml")
    if config is None:
        print("Échec du chargement de la configuration. Arrêt.")
        sys.exit(1)

    # 2. Initialiser le manager avec la configuration
    sim = SimulationManager(config)

    # 3. Lancer la simulation
    try:
        sim.run()

        # 4. Si on est en mode GUI, garder la fenêtre ouverte
        mode_str = str(config["simulation"]["connect_mode"]).strip().lower()
        if mode_str == "gui":
            print("Simulation terminée. Ferme la fenêtre PyBullet pour quitter.")
            # On regarde juste si PyBullet est encore connecté (pas besoin de physics_client_id)
            while p.isConnected():
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
