"""Synchronous per-run tensors consumed by relational learning models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pybullet as p

from simulator.interventions import AppliedIntervention, InterventionEvent


EVENT_CATALOG_COLUMNS = [
    "event_index",
    "event_id",
    "target",
    "start_time",
    "end_time",
    "duration",
    "type",
    "frame",
    "link_id",
    "force_x",
    "force_y",
    "force_z",
]


def _vector(value, size: int, fill: float = 0.0) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        return np.full(size, fill, dtype=float)
    return array


class LearningTraceLogger:
    """Collect aligned state, context and intervention tensors.

    A row is emitted whenever the simulator emits relational ground truth, i.e.
    on control updates.  The arrays are deliberately model-agnostic: pairing
    baseline and intervened futures remains a dataset concern.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        log_dir: str,
        run_id: int,
        uavs: Sequence,
        events: Sequence[InterventionEvent],
        artifact_stem: str | None = None,
    ):
        self.run_id = int(run_id)
        stem = artifact_stem or f"run_{self.run_id}"
        self.path = Path(log_dir) / f"{stem}_learning_trace.npz"
        self.event_catalog_path = Path(log_dir) / f"{stem}_intervention_events.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.uavs = sorted(uavs, key=lambda item: str(item.name))
        self.drone_names = [str(item.name) for item in self.uavs]
        self.events = list(events)
        self.rows_written = 0
        self._closed = False
        self._samples: list[dict[str, np.ndarray | float]] = []
        self._previous_event_index = np.full(len(self.uavs), -1, dtype=np.int32)
        self._write_event_catalog()

    def _write_event_catalog(self) -> None:
        with self.event_catalog_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_CATALOG_COLUMNS)
            writer.writeheader()
            for event_index, event in enumerate(self.events):
                for target in event.targets:
                    writer.writerow(
                        {
                            "event_index": event_index,
                            "event_id": event.id,
                            "target": target,
                            "start_time": event.start_time,
                            "end_time": event.end_time,
                            "duration": event.duration,
                            "type": event.type,
                            "frame": event.frame,
                            "link_id": event.link_id,
                            "force_x": event.force[0],
                            "force_y": event.force[1],
                            "force_z": event.force[2],
                        }
                    )

    def log_step(
        self,
        sim_time: float,
        applications: Sequence[AppliedIntervention],
    ) -> None:
        if self._closed:
            return
        node_count = len(self.uavs)
        by_target = {application.target: application for application in applications}

        state = np.full((node_count, 6), np.nan, dtype=np.float32)
        attitude = np.full((node_count, 6), np.nan, dtype=np.float32)
        target = np.full((node_count, 6), np.nan, dtype=np.float32)
        desired_offset = np.zeros((node_count, 3), dtype=np.float32)
        wind = np.zeros((node_count, 3), dtype=np.float32)
        obstacle = np.full((node_count, 4), np.nan, dtype=np.float32)
        leader_message_age = np.full(node_count, np.nan, dtype=np.float32)
        leader_message_received = np.zeros(node_count, dtype=np.uint8)
        nearest_neighbor_distance = np.full(node_count, np.nan, dtype=np.float32)
        repulsion = np.zeros((node_count, 4), dtype=np.float32)
        formation_error = np.zeros((node_count, 4), dtype=np.float32)
        formation_attraction = np.zeros((node_count, 4), dtype=np.float32)
        formation_damping = np.zeros((node_count, 4), dtype=np.float32)
        formation_scales = np.zeros((node_count, 2), dtype=np.float32)
        motor_saturation_ratio = np.zeros(node_count, dtype=np.float32)
        collision = np.zeros(node_count, dtype=np.uint8)
        is_leader = np.zeros(node_count, dtype=np.uint8)
        valid = np.ones(node_count, dtype=np.uint8)
        crashed = np.zeros(node_count, dtype=np.uint8)
        control_update_index = np.zeros(node_count, dtype=np.int64)

        intervention_active = np.zeros(node_count, dtype=np.uint8)
        intervention_onset = np.zeros(node_count, dtype=np.uint8)
        intervention_event_index = np.full(node_count, -1, dtype=np.int32)
        intervention_force_command = np.zeros((node_count, 3), dtype=np.float32)
        intervention_force_world = np.zeros((node_count, 3), dtype=np.float32)
        intervention_elapsed = np.zeros(node_count, dtype=np.float32)
        intervention_remaining = np.zeros(node_count, dtype=np.float32)
        intervention_frame = np.full(node_count, -1, dtype=np.int8)

        for index, uav in enumerate(self.uavs):
            ground_truth = uav.get_ground_truth_state()
            if ground_truth:
                state[index, :3] = _vector(ground_truth.get("pos"), 3, np.nan)
                state[index, 3:] = _vector(ground_truth.get("vel"), 3, np.nan)
                quaternion = _vector(ground_truth.get("orn_q"), 4, np.nan)
                if np.isfinite(quaternion).all():
                    attitude[index, :3] = p.getEulerFromQuaternion(quaternion)
                attitude[index, 3:] = _vector(ground_truth.get("ang_vel"), 3, np.nan)
            else:
                valid[index] = 0

            target[index, :3] = _vector(getattr(uav, "current_target_pos", []), 3, np.nan)
            target[index, 3:] = _vector(getattr(uav, "current_target_vel", []), 3, np.nan)
            desired_offset[index] = _vector(getattr(uav, "desired_formation_offset", []), 3)
            wind[index] = _vector(getattr(uav, "current_wind", []), 3)
            obstacle[index, 0] = float(getattr(uav, "nearest_obstacle_dist", np.nan))
            obstacle[index, 1:] = _vector(getattr(uav, "nearest_obstacle_dir", []), 3, np.nan)
            leader_message_age[index] = float(getattr(uav, "last_leader_message_age_s", np.nan))
            leader_message_received[index] = int(
                bool(getattr(uav, "last_leader_message_received", False))
            )
            nearest_neighbor_distance[index] = float(
                getattr(uav, "dist_to_nearest_neighbor", np.nan)
            )
            repulsive_force = _vector(getattr(uav, "last_repulsive_force", []), 3)
            repulsion[index, 0] = float(np.linalg.norm(repulsive_force))
            repulsion[index, 1:] = repulsive_force
            relative_error = sum(
                (
                    _vector(value, 3)
                    for value in getattr(
                        uav, "last_formation_relative_error_by_sender", {}
                    ).values()
                ),
                np.zeros(3),
            )
            attraction = sum(
                (
                    _vector(value, 3)
                    for value in getattr(
                        uav, "last_formation_attraction_applied_by_sender", {}
                    ).values()
                ),
                np.zeros(3),
            )
            damping = sum(
                (
                    _vector(value, 3)
                    for value in getattr(
                        uav, "last_formation_damping_applied_by_sender", {}
                    ).values()
                ),
                np.zeros(3),
            )
            formation_error[index, 0] = float(np.linalg.norm(relative_error))
            formation_error[index, 1:] = relative_error
            formation_attraction[index, 0] = float(np.linalg.norm(attraction))
            formation_attraction[index, 1:] = attraction
            formation_damping[index, 0] = float(np.linalg.norm(damping))
            formation_damping[index, 1:] = damping
            formation_scales[index, 0] = float(
                getattr(uav, "last_formation_correction_scale", 1.0)
            )
            formation_scales[index, 1] = float(
                getattr(uav, "last_formation_application_scale", 0.0)
            )
            rpms = np.asarray(getattr(uav, "last_rpms", []), dtype=float).reshape(-1)
            max_rpm = float(getattr(uav, "MAX_RPM", 0.0))
            if rpms.size and max_rpm > 0.0:
                motor_saturation_ratio[index] = float(np.mean(rpms >= 0.98 * max_rpm))
            collision_fn = getattr(uav, "compute_collision_flag", None)
            if callable(collision_fn):
                collision[index] = int(bool(collision_fn()))
            is_leader[index] = int(bool(getattr(uav, "leader", False)))
            crashed[index] = int(bool(getattr(uav, "crashed", False)))
            valid[index] = int(valid[index] and not crashed[index])
            control_update_index[index] = int(getattr(uav, "control_update_index", 0))

            application = by_target.get(str(uav.name))
            if application is not None:
                intervention_active[index] = 1
                intervention_onset[index] = int(
                    self._previous_event_index[index] != application.event_index
                )
                intervention_event_index[index] = application.event_index
                intervention_force_command[index] = application.command_force
                intervention_force_world[index] = application.world_force
                intervention_elapsed[index] = application.elapsed
                intervention_remaining[index] = application.remaining
                intervention_frame[index] = 0 if application.frame == "world" else 1

        self._previous_event_index = intervention_event_index.copy()

        self._samples.append(
            {
                "time": float(sim_time),
                "state": state,
                "attitude": attitude,
                "target": target,
                "desired_offset": desired_offset,
                "wind": wind,
                "obstacle": obstacle,
                "leader_message_age": leader_message_age,
                "leader_message_received": leader_message_received,
                "nearest_neighbor_distance": nearest_neighbor_distance,
                "repulsion": repulsion,
                "formation_error": formation_error,
                "formation_attraction": formation_attraction,
                "formation_damping": formation_damping,
                "formation_scales": formation_scales,
                "motor_saturation_ratio": motor_saturation_ratio,
                "collision": collision,
                "is_leader": is_leader,
                "valid": valid,
                "crashed": crashed,
                "control_update_index": control_update_index,
                "intervention_active": intervention_active,
                "intervention_onset": intervention_onset,
                "intervention_event_index": intervention_event_index,
                "intervention_force_command": intervention_force_command,
                "intervention_force_world": intervention_force_world,
                "intervention_elapsed": intervention_elapsed,
                "intervention_remaining": intervention_remaining,
                "intervention_frame": intervention_frame,
            }
        )
        self.rows_written += 1

    def close(self) -> None:
        if self._closed:
            return
        node_count = len(self.drone_names)
        dynamic_specs = {
            "state": ((node_count, 6), np.float32),
            "attitude": ((node_count, 6), np.float32),
            "target": ((node_count, 6), np.float32),
            "desired_offset": ((node_count, 3), np.float32),
            "wind": ((node_count, 3), np.float32),
            "obstacle": ((node_count, 4), np.float32),
            "leader_message_age": ((node_count,), np.float32),
            "leader_message_received": ((node_count,), np.uint8),
            "nearest_neighbor_distance": ((node_count,), np.float32),
            "repulsion": ((node_count, 4), np.float32),
            "formation_error": ((node_count, 4), np.float32),
            "formation_attraction": ((node_count, 4), np.float32),
            "formation_damping": ((node_count, 4), np.float32),
            "formation_scales": ((node_count, 2), np.float32),
            "motor_saturation_ratio": ((node_count,), np.float32),
            "collision": ((node_count,), np.uint8),
            "is_leader": ((node_count,), np.uint8),
            "valid": ((node_count,), np.uint8),
            "crashed": ((node_count,), np.uint8),
            "control_update_index": ((node_count,), np.int64),
            "intervention_active": ((node_count,), np.uint8),
            "intervention_onset": ((node_count,), np.uint8),
            "intervention_event_index": ((node_count,), np.int32),
            "intervention_force_command": ((node_count, 3), np.float32),
            "intervention_force_world": ((node_count, 3), np.float32),
            "intervention_elapsed": ((node_count,), np.float32),
            "intervention_remaining": ((node_count,), np.float32),
            "intervention_frame": ((node_count,), np.int8),
        }
        arrays = {
            key: (
                np.stack([sample[key] for sample in self._samples])
                if self._samples
                else np.empty((0, *shape), dtype=dtype)
            )
            for key, (shape, dtype) in dynamic_specs.items()
        }
        np.savez_compressed(
            self.path,
            schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int32),
            times=np.asarray(
                [sample["time"] for sample in self._samples], dtype=np.float64
            ),
            drone_names=np.asarray(self.drone_names),
            swarm_ids=np.asarray([str(getattr(uav, "swarm_id", "")) for uav in self.uavs]),
            formation_types=np.asarray(
                [str(getattr(uav, "formation_type", "none")) for uav in self.uavs]
            ),
            state_fields=np.asarray(["x", "y", "z", "vx", "vy", "vz"]),
            target_fields=np.asarray(
                ["target_x", "target_y", "target_z", "target_vx", "target_vy", "target_vz"]
            ),
            obstacle_fields=np.asarray(
                ["distance", "direction_x", "direction_y", "direction_z"]
            ),
            repulsion_fields=np.asarray(["magnitude", "x", "y", "z"]),
            formation_error_fields=np.asarray(["magnitude", "x", "y", "z"]),
            formation_attraction_fields=np.asarray(["magnitude", "x", "y", "z"]),
            formation_damping_fields=np.asarray(["magnitude", "x", "y", "z"]),
            formation_scale_fields=np.asarray(
                ["correction_saturation", "transition_application"]
            ),
            formation_control_modes=np.asarray(
                [
                    str(getattr(uav, "formation_control_mode", "none"))
                    for uav in self.uavs
                ]
            ),
            intervention_frame_convention=np.asarray("-1=inactive,0=world,1=link"),
            **arrays,
        )
        self._closed = True
