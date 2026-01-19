# main.py
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config
import pybullet as p
import time


def main():
    # Load the config
    config = load_config("config.yaml")

    # For safety, we verify it's a dict
    if config is None:
        raise RuntimeError(
            "The configuration was not loaded (config=None). "
            "Check the config.yaml file."
        )

    sim = SimulationManager(config)

    try:
        sim.run()

        # keep the GUI window open when simulation is finished
        mode = str(config["simulation"]["connect_mode"]).lower()
        if mode == "gui":
            print("Simulation finished. Close the PyBullet window to quit.")
            while p.isConnected():
                time.sleep(0.1)

    finally:
        sim.stop()


if __name__ == "__main__":
    main()