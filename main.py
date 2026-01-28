# main.py
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config
import pybullet as p
import time


def main():
    """
    Primary entry point for the autonomous system simulator.

    This function orchestrates the high-level execution flow:
    1. Configuration Loading: Retrieves environment, agent, and physics parameters 
        from the 'config.yaml' file.
    2. Initialization: Instantiates the SimulationManager, which connects to 
        PyBullet and builds the world.
    3. Execution: Launches the main simulation loop (physics + control).
    4. Persistence: If running in GUI mode, maintains the window after the 
        simulation time expires to allow for visual inspection.
    5. Resource Management: Ensures a clean disconnection from the physics 
        server and network proxies via a 'finally' block, regardless of 
        execution success or failure.

    Raises:
        RuntimeError: If the configuration file fails to load or returns null.
    """
    # Load the configuration
    config = load_config("config.yaml")

    # To be on the safe side, we still check that it is indeed a dict.
    if config is None:
        raise RuntimeError(
            "Configuration not loaded (config=None). "
            "Check the config.yaml file."
        )

    sim = SimulationManager(config)

    try:
        sim.run()

        # keep the GUI window open when the simulation is finished
        mode = str(config["simulation"]["connect_mode"]).lower()
        if mode == "gui":
            print("Simulation terminated. Close PyBullet window to quit.")
            while p.isConnected():
                time.sleep(0.1)

    finally:
        sim.stop()


if __name__ == "__main__":
    main()
