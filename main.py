# main.py
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config
import pybullet as p
import time


def main():
    # Charge la config
    config = load_config("config.yaml")

    # Par sécurité, on vérifie quand même que c’est bien un dict
    if config is None:
        raise RuntimeError(
            "La configuration n'a pas été chargée (config=None). "
            "Vérifie le fichier config.yaml."
        )

    sim = SimulationManager(config)

    try:
        sim.run()

        # garder la fenêtre GUI ouverte quand la simu est terminée
        mode = str(config["simulation"]["connect_mode"]).lower()
        if mode == "gui":
            print("Simulation terminée. Ferme la fenêtre PyBullet pour quitter.")
            while p.isConnected():
                time.sleep(0.1)

    finally:
        sim.stop()


if __name__ == "__main__":
    main()