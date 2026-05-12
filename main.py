# main.py
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config, save_config
import pybullet as p
import time
import os


def run_batch_simu(start_config, n_run):
    if n_run == 1:
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
                sim.stop()
                while p.isConnected():
                    time.sleep(0.1)

        finally:
            sim.stop()

    else:
        config = load_config(start_config)
        ini_compteur = config["simulation"]["run_config"]
        if start_config == "config.yaml":
            config["simulation"]["connect_mode"] = "direct"

        for ind in range(1, n_run+1):
            run_config = config.copy()

            print(f"\n SIMULATION {ind}")

            sim = SimulationManager(run_config)

            try:
                sim.run()

            except Exception as e:
                print(f"Erreur pendant la simulation {ind} : {e}")

            finally:
                run_config = sim.it_stop(ini_compteur + ind)
                print(f"Simulation {ind} finished.")
    
                # Mettre à jour la config pour continuer de print
                config_saving_path = os.path.join("logs", f"config_{ini_compteur + ind}.yaml")
                config = load_config(save_config(run_config, config_saving_path))

        print("Fin de toutes les simulations.")


def main():

    run_batch_simu("config.yaml", 2)

if __name__ == "__main__":
    main()