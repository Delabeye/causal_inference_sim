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

## 2. Configuration (`config.yaml`)

All parameters are centralized in `config.yaml` to ensure experiment reproducibility.

| Section | Key Parameters | Purpose |
| :--- | :--- | :--- |
| **simulation** | `dt`, `max_sim_time`, `connect_mode` | Physics step timing and UI settings. |
| **world** | `city`, `Astar` | City dimensions and pathfinding safety margins. |
| **swarm** | `leader`, `min_sep`, `avoid_gain` | Interaction rules for the drone swarm. |
| **agents** | `sensors`, `waypoints`, `noise` | Individual drone hardware and mission profile. |

---

## 3. Running the Simulator

To start the simulation with the current configuration:

```bash
python main.py
