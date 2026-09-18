# Simulator data contract

The simulator has three distinct responsibilities:

1. `simulator_manager.py` orchestrates control and physics.
2. `interventions.py` parses, validates and applies controlled interventions.
3. The two loggers export learning data and relational ground truth without
   changing the simulated dynamics.

## Per-run artifacts

- `run_<id>_learning_trace.npz`: control-rate state, context, validity masks and
  node-local intervention tensors.
- `run_<id>_intervention_events.csv`: stable mapping from `event_index` to the
  event id, target, timing, frame and planned force.
- `run_<id>_interactions.csv`: long-form directed interaction audit.
- `run_<id>_ground_truth_matrices.npz`: receiver-by-sender graph tensors.
- `run_<id>_summary.json`: seeds, formation, pairing metadata and artifact paths.
- `run_<id>_counterfactual_forks.json`: parent/branch manifest for physical
  counterfactual forks.
- `run_<id>_fork_<k>_learning_trace.npz`: intervention rollout restored from
  checkpoint `k`.

The learning trace uses sorted `drone_names`. Its main arrays are:

- `state[T,N,6] = [position, velocity]`;
- `target[T,N,6] = [target_position, target_velocity]`;
- `attitude[T,N,6] = [roll, pitch, yaw, angular_velocity]`;
- `desired_offset`, `wind`, `obstacle`, `repulsion`, `is_leader`;
- communication age/reception, nearest-neighbour distance, collision and motor
  saturation;
- `intervention_active`, `intervention_onset`, `intervention_event_index`;
- `intervention_force_command` and `intervention_force_world`;
- `intervention_elapsed`, `intervention_remaining`, `intervention_frame`;
- `valid`, `crashed` and `control_update_index`.

The decoder must route the intervention only to nodes where
`intervention_active == 1`. Propagation to other drones must occur through the
learned graph, not by broadcasting the intervention vector.

## Formation topology

`line` is a directed predecessor chain:

```text
drone_0 -> drone_1 -> drone_2 -> drone_3
```

Every follower tracks the state and yaw of its immediate predecessor with a
local one-spacing offset. `trail`, `triangle` and `v` retain a leader-star
control topology. The selected topology and `parent_by_follower` mapping are
stored in each `run_<id>_formation_<swarm>.json` artifact.

## Stratified intervention schedule

The paired-plan generator defaults to a 60 s run, 8 s warm-up, 8 s future
horizon and seeded draws in `[10,12]`, `[20,22]`, `[30,32]`, `[40,42]` and
`[50,52]`. Exact times and the schedule seed are written to the plan and run
summary. Baseline and perturbed rows sharing a simulation seed receive the same
schedule.

## Physical counterfactual forks

Paired plans enable `counterfactual_forks` on intervention rows. The simulator:

1. runs the complete parent trajectory without applying interventions;
2. stores PyBullet and Python checkpoints at the stratified times;
3. restores every checkpoint after the parent is complete;
4. executes one short intervention rollout;
5. restores the final parent state before writing summaries.

The checkpoint includes Bullet bodies, PID/EKF/sensor state, swarm message
queues, controller timers and Python/NumPy RNG state. Branches have independent
learning traces and never write into the parent UAV CSV buffers. Fork mode is
restricted to deterministic DIRECT simulations.

## Valid intervention configuration

Every enabled event needs a unique id, at least one existing UAV target, a
strictly positive duration and a finite 3-D force. Events targeting the same UAV
cannot overlap. `frame` accepts `world` or `link` (`body` is normalized to
`link`). The trace always supplies `intervention_force_world`, including for
body-frame interventions.
