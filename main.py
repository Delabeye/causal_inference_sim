from simulator.simulator_manager import SimulationManager
from utilities.config import load_config
import pybullet as p
import time

def main():
    config = load_config("config.yaml")
    sim = SimulationManager(config)
    try:
        sim.run()
        # garder la fenêtre GUI ouverte
        if config["simulation"]["connect_mode"].lower() == "gui":
            print("Simulation terminée. Ferme la fenêtre pour quitter.")
            while p.isConnected():
                time.sleep(0.1)
    finally:
        sim.stop()

if __name__ == "__main__":
    main()
