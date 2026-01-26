# causal_inference_sim
Causal inference simulation tool for coupled dynamical systems

This project is an advanced simulation platform built on **PyBullet** designed to study coupled dynamical systems—specifically autonomous drone swarms. It provides high-fidelity physics, procedural environment generation, and a robust pipeline for **causal inference** and failure attribution.

---

## 1. Project Architecture

The simulator follows a modular design to separate physics, control, and data analysis:

* **Core Orchestrator**: `SimulationManager` handles the PyBullet world, agent lifecycle, and the main loop.
* **Environment**: A procedural city generator (`world.py`) creates blocks, roads, and obstacles.
* **Agents (UAVs)**: Quadrotors with realistic motor physics, state estimation (EKF), and PID-based stabilization.
* **Analysis Suite**: Dedicated tools for trajectory visualization and causal relationship discovery.

---
## Installation and Dependencies

To run this simulator and its analysis suite, ensure you have **Python 3.10+** installed.

### A. Core Dependencies
The project relies on the following third-party libraries:

* **Physics & Simulation**: `pybullet`.
* **Data Processing**: `numpy`, `pandas`, and `PyYAML`.
* **Pathfinding**: `pathfinding`.
* **Causal Analysis & ML**: `torch` (PyTorch), `scikit-learn`, `networkx`, and `statsmodels`.
* **Visualization**: `matplotlib` and `scipy`.

### B. Setup Instructions
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Delabeye/causal_inference_sim.git
    cd causal_inference_sim
    ```
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install requirements**:
    ```bash
    pip install pybullet numpy pandas PyYAML matplotlib networkx torch scikit-learn statsmodels scipy pathfinding
    ```
---

## 3. Configuration (`config.yaml`)

All parameters are centralized in `config.yaml` to ensure experiment reproducibility.

| Section | Parameter | Description |
| :--- | :--- | :--- |
| **simulation** | `connect_mode` | Visual mode: "gui" for 3D interface or "direct" for headless/fast mode |
| | `dt` | Physics simulation time-step (e.g., 0.00416 s for 2240 Hz) |
| | `max_sim_time` | Total simulation duration in seconds |
| **physics** | `gravity` | Gravity vector $[x, y, z]$ applied to the world (e.g., $[0, 0, -9.81]$) |
| **world** | `type` | Environment type ("generated" for procedural city or "custom") |
| | `city` | Procedural parameters: block count (`n_blocks`), road width, and building density |
| | `res` | Grid resolution in meters for the heightmap and path planner |
| | `Astar` | Safety margins (`safety_margin`) and world bounds for trajectory calculation |
| **swarm** | `leader` | NAme of the designated leader drone followed by others |
| | `min_sep` | Minimum separation distance maintained between swarm members |
| | `avoid_gain` | Strength of the gain applied for inter-drone collision avoidance |
| | `port_in` / `port_out` | Network configuration for UDP communication between agents |
| **agents** | `type` | Agent type: "uav" for drones or "radar" for fixed stations |
| |`name`| Agent name for communications |
| |`mass` | Physical mass of the aircraft in kilograms |
| | `physics` | Thrust coefficients, torque, and max motor RPM |
| | `waypoints` | List of $[x, y, z]$ coordinates for the leader drone to follow |
| | `sensors` | Noise parameters and frequencies for GNSS, IMU, and LiDAR |
| | `wind` | Wind model including mean speed and turbulence intensity |

---

## 4. Running the Simulator

To start the simulation with the current configuration run:

`main.py`

---

## 5. Causal Analysis
The defining feature of this simulator is its ability to reconstruct the "causal chain" of events during a mission.

### A. Key Analysis Features
The `causal_analysis.py` script implements several advanced techniques to explain swarm behavior:

Failure Detection: Automatically identifies five types of events: collisions, formation loss, GNSS degradation, wind-induced loss, and suboptimal trajectories.

Neural Relational Inference (NRI): An unsupervised Graph Neural Network (GNN) that infers a latent interaction graph between drones by minimizing prediction error of their future states.

Granger Causality: Specifically evaluates how exogenous factors (like wind gusts or GNSS noise) statistically "cause" failures in specific drones.

Root Cause Attribution: A logistic regression model that calculates "interaction pressure" to provide a probabilistic distribution of causes for every detected failure.

### B. How to Use the Analysis Tool
Follow these steps to generate a causal report from your simulation logs:

Run a Simulation: Ensure logs are generated in the logs/ folder.

Execute the Pipeline: Use the CLI to process the data:


`python analysis/causal_analysis.py --log_dir logs --output_dir results --nri_steps 2000 --downsample 2` 

Review the Outputs: The script generates several files in your output directory:

nri_inferred_graph.png: A directed graph showing which drones influence others.

nri_edge_probs_heatmap.png: A matrix displaying the probability of interaction between every pair of agents.

event_root_cause_explanations.json: A detailed breakdown of why each specific failure happened (e.g., "Drone 2 crashed due to 75% interaction pressure from Drone 1").

---

## 6. Technical Implementation

### A. Navigation: Heightmap A*
The `Path_planning.py` module handles navigation in constrained urban environments:
* **Geometric Inflation**: Buildings are "inflated" by a `safety_margin` in the 2D heightmap to ensure drones do not clip corners.
* **String Pulling**: A post-processing step smooths the A* path to create more efficient trajectories.



### B. State Estimation: Extended Kalman Filter (EKF)
Each agent uses an EKF to fuse asynchronous sensor data:
* **IMU Prediction**: Uses high-frequency acceleration and orientation to predict future states.
* **Gravity Compensation**: The filter removes the gravity vector from IMU readings to isolate linear motion.
* **GNSS/Radar Updates**: Corrects the state estimate using absolute position measurements.



### C. Collision Avoidance
Real-time repulsive forces are calculated based on the distance to obstacles and other agents in the swarm or detected by radar.

---

## 7. Conclusion & Future Work

This simulator provides a rigorous framework for auditing the resilience of autonomous swarms. By combining high-fidelity physics with advanced AI for causal inference, it allows researchers to move beyond simple failure detection toward deep systemic understanding. 

Future updates will include:
* Heterogeneous swarms (mixed UAV/UGV).
* Dynamic obstacle support for complex urban traffic.
* Advanced network modeling for communication-denied environments.
