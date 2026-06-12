# main.py
from simulator.simulator_manager import SimulationManager
from utilities.config import load_config, save_config
import pybullet as p
import time
import os


BASIC_CONFIG_PATH = "config.yaml"
RUN_SIM_GUI = 1


LOG_NUMBERS = 1


def run_batch_simu(n_run):

    config = load_config("config.yaml")

# Simulation avec interface 
    if n_run == 1:
        
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
                return

        finally:
            if p.isConnected():
                sim.stop()

# Création de logs en batch

    else:
        config["simulation"]["connect_mode"] = "direct"
        sim = SimulationManager(config)
        init_compteur = config["simulation"]["run_config"]
        
        for ind in range(1, n_run+1):

            print(f"\n SIMULATION {ind}")
            

            try:
                sim.run()

            except Exception as e:
                print(f"Erreur pendant la simulation {ind} : {e}")

            finally:
                print(f"Simulation {ind} finished.")
                sim.it_stop(init_compteur + ind)
    
                # Mettre à jour la config pour continuer de print
            # config_saving_path = os.path.join("logs/config_save", f"config_{ini_compteur + ind}.yaml")
            # config = load_config(save_config(run_config, config_saving_path))
                sim.reset()

        print("Fin de toutes les simulations.")
        sim.disconnect()




# ------------------------------------------------------------------
def main():

    run_batch_simu(LOG_NUMBERS)


if __name__ == "__main__":
    main()