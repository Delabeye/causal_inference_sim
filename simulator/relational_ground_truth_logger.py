"""Auditable per-pair relational ground truth emitted by the simulator."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pybullet as p


INTERACTION_LOG_COLUMNS = [
    "time",
    "run_id",
    "sender",
    "receiver",
    "same_swarm",
    "leader_reference_edge",
    "message_received_flag",
    "message_age_s",
    "central_avoidance_x",
    "central_avoidance_y",
    "central_avoidance_z",
    "local_repulsion_x",
    "local_repulsion_y",
    "local_repulsion_z",
    "contact_flag",
    "contact_force",
    "delta_target_norm",
    "delta_rpm_norm",
    # Audit columns: distinguish configured structure from mechanisms used now.
    "sender_swarm_id",
    "receiver_swarm_id",
    "leader_reference_active",
    "message_used_for_control_flag",
    "state_access_mode",
    "central_avoidance_raw_x",
    "central_avoidance_raw_y",
    "central_avoidance_raw_z",
    "central_avoidance_norm",
    "local_repulsion_raw_x",
    "local_repulsion_raw_y",
    "local_repulsion_raw_z",
    "local_repulsion_norm",
    "local_repulsion_source_age_s",
    "repulsion_normalization_scale",
    "building_repulsion_raw_x",
    "building_repulsion_raw_y",
    "building_repulsion_raw_z",
    "building_repulsion_applied_x",
    "building_repulsion_applied_y",
    "building_repulsion_applied_z",
    "total_repulsion_x",
    "total_repulsion_y",
    "total_repulsion_z",
    "contact_count",
    "interaction_active",
    "counterfactual_status",
    # Counterfactual vectors are appended for backward CSV compatibility.
    "delta_target_x",
    "delta_target_y",
    "delta_target_z",
    "delta_target_vel_x",
    "delta_target_vel_y",
    "delta_target_vel_z",
    "delta_rpm_0",
    "delta_rpm_1",
    "delta_rpm_2",
    "delta_rpm_3",
    "control_counterfactual_valid",
    # Directed elastic-formation decomposition (append-only schema extension).
    "formation_control_mode",
    "relative_position_error_x",
    "relative_position_error_y",
    "relative_position_error_z",
    "relative_position_error_norm",
    "relative_velocity_error_x",
    "relative_velocity_error_y",
    "relative_velocity_error_z",
    "formation_attraction_raw_x",
    "formation_attraction_raw_y",
    "formation_attraction_raw_z",
    "formation_attraction_x",
    "formation_attraction_y",
    "formation_attraction_z",
    "formation_attraction_norm",
    "formation_damping_x",
    "formation_damping_y",
    "formation_damping_z",
    "formation_damping_norm",
    "formation_transport_vx",
    "formation_transport_vy",
    "formation_transport_vz",
    "formation_correction_scale",
    "formation_application_scale",
]


class RelationalGroundTruthLogger:
    """Stream one row per directed UAV pair and control update.

    Rows use receiver-by-sender semantics: a row ``sender=i, receiver=j``
    corresponds to matrix entry ``A[j, i]``. Self-pairs are never emitted.
    Counterfactual PID calls start from a snapshot of the real controller state
    and restore its post-command state, so logging cannot alter the simulation.
    """

    SCHEMA_VERSION = 2

    def __init__(self, log_dir: str, run_id: int, physics_client_id: int):
        self.run_id = int(run_id)
        self.physics_client_id = int(physics_client_id)
        self.path = Path(log_dir) / f"run_{self.run_id}_interactions.csv"
        self.matrix_path = (
            Path(log_dir) / f"run_{self.run_id}_ground_truth_matrices.npz"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=INTERACTION_LOG_COLUMNS)
        self._writer.writeheader()
        self._closed = False
        self.rows_written = 0
        self._matrix_times = []
        self._matrix_drone_names = None
        self._matrix_channels = {
            "structural": [],
            "active": [],
            "message_used": [],
            "target_delta_norm": [],
            "rpm_delta_norm": [],
            "contact_force": [],
            "counterfactual_valid": [],
            "attraction_norm": [],
            "damping_norm": [],
            "repulsion_norm": [],
            "relative_error_norm": [],
        }

    @staticmethod
    def _swarm_id(uav) -> str:
        return str(getattr(uav, "swarm_id", "") or "")

    @staticmethod
    def _vector(mapping, sender: str) -> np.ndarray:
        value = mapping.get(sender, np.zeros(3)) if isinstance(mapping, dict) else np.zeros(3)
        vector = np.asarray(value, dtype=float)
        return vector if vector.shape == (3,) else np.zeros(3)

    def _contact_features(self, receiver, sender) -> tuple[int, float, int]:
        try:
            contacts = p.getContactPoints(
                bodyA=receiver.bodyId,
                bodyB=sender.bodyId,
                physicsClientId=self.physics_client_id,
            ) or ()
        except Exception:
            contacts = ()
        normal_force = float(sum(max(0.0, float(contact[9])) for contact in contacts))
        return int(bool(contacts)), normal_force, len(contacts)

    def _message_features(self, receiver, sender_name: str, leader_edge: bool):
        received_times = getattr(receiver, "last_message_receive_time_by_sender", {})
        source_times = getattr(receiver, "last_message_source_time_by_sender", {})
        received = sender_name in received_times
        source_time = source_times.get(sender_name, math.nan)
        if received and np.isfinite(source_time):
            age = max(0.0, float(receiver._sim_time) - float(source_time))
        else:
            age = math.nan

        local_raw = getattr(receiver, "last_inter_uav_repulsion_raw", {})
        used = sender_name in local_raw
        if leader_edge and bool(getattr(receiver, "last_leader_reference_active", False)):
            used = True
            leader_received = bool(getattr(receiver, "last_leader_message_received", False))
            if leader_received:
                received = True
                leader_age = getattr(receiver, "last_leader_message_age_s", math.nan)
                if np.isfinite(leader_age):
                    age = float(leader_age)
        return int(received), age, int(used)

    @staticmethod
    def _counterfactual_features(receiver, sender_name: str):
        valid = bool(getattr(receiver, "last_counterfactual_control_valid", False))
        records = getattr(receiver, "last_counterfactual_control_by_sender", {})
        record = records.get(sender_name) if isinstance(records, dict) else None
        if record is None and valid:
            return {
                "delta_target": np.zeros(3),
                "delta_target_vel": np.zeros(3),
                "delta_rpm": np.zeros(4),
                "valid": 1,
                "status": "valid_no_active_control_path",
            }
        if record is None:
            return {
                "delta_target": np.full(3, math.nan),
                "delta_target_vel": np.full(3, math.nan),
                "delta_rpm": np.full(4, math.nan),
                "valid": 0,
                "status": "unavailable_no_control_update",
            }
        return {
            "delta_target": np.asarray(record.get("delta_target", np.zeros(3)), dtype=float),
            "delta_target_vel": np.asarray(
                record.get("delta_target_vel", np.zeros(3)), dtype=float
            ),
            "delta_rpm": np.asarray(record.get("delta_rpm", np.zeros(4)), dtype=float),
            "valid": 1,
            "status": str(record.get("status", "valid_pid_command_ablation")),
        }

    def _pair_row(self, sim_time: float, sender, receiver) -> dict:
        sender_name = str(sender.name)
        receiver_name = str(receiver.name)
        sender_swarm = self._swarm_id(sender)
        receiver_swarm = self._swarm_id(receiver)
        same_swarm = bool(sender_swarm and sender_swarm == receiver_swarm)
        leader_ref = getattr(receiver, "swarm_leader_ref", None)
        leader_edge = bool(same_swarm and leader_ref is sender and receiver is not sender)
        leader_active = bool(leader_edge and getattr(receiver, "last_leader_reference_active", False))

        central = self._vector(
            getattr(receiver, "last_central_avoidance_applied_by_sender", {}), sender_name
        )
        central_raw = self._vector(
            getattr(receiver, "last_central_avoidance_raw_by_sender", {}), sender_name
        )
        local = self._vector(
            getattr(receiver, "last_inter_uav_repulsion_applied", {}), sender_name
        )
        local_raw = self._vector(
            getattr(receiver, "last_inter_uav_repulsion_raw", {}), sender_name
        )
        local_age = getattr(receiver, "last_inter_uav_repulsion_source_age_s", {}).get(
            sender_name, math.nan
        )
        relative_error = self._vector(
            getattr(receiver, "last_formation_relative_error_by_sender", {}),
            sender_name,
        )
        relative_velocity_error = self._vector(
            getattr(
                receiver,
                "last_formation_relative_velocity_error_by_sender",
                {},
            ),
            sender_name,
        )
        attraction_raw = self._vector(
            getattr(receiver, "last_formation_attraction_raw_by_sender", {}),
            sender_name,
        )
        attraction = self._vector(
            getattr(receiver, "last_formation_attraction_applied_by_sender", {}),
            sender_name,
        )
        damping = self._vector(
            getattr(receiver, "last_formation_damping_applied_by_sender", {}),
            sender_name,
        )
        transport = self._vector(
            getattr(receiver, "last_formation_transport_velocity_by_sender", {}),
            sender_name,
        )
        message_flag, message_age, message_used = self._message_features(
            receiver, sender_name, leader_edge
        )
        contact_flag, contact_force, contact_count = self._contact_features(receiver, sender)
        counterfactual = self._counterfactual_features(receiver, sender_name)
        delta_target = counterfactual["delta_target"]
        delta_target_vel = counterfactual["delta_target_vel"]
        delta_rpm = counterfactual["delta_rpm"]
        building_raw = np.asarray(
            getattr(receiver, "last_building_repulsive_force_raw", np.zeros(3)),
            dtype=float,
        )
        building_applied = np.asarray(
            getattr(receiver, "last_building_repulsive_force_applied", np.zeros(3)),
            dtype=float,
        )
        total_repulsion = np.asarray(
            getattr(receiver, "last_repulsive_force", np.zeros(3)), dtype=float
        )

        mechanism_active = bool(
            leader_active
            or np.linalg.norm(central) > 0.0
            or np.linalg.norm(local) > 0.0
            or contact_flag
        )
        if np.linalg.norm(local) > 0.0:
            access_mode = "delayed_swarm_message"
        elif np.linalg.norm(central) > 0.0:
            access_mode = "central_controller_direct_state"
        elif leader_active:
            access_mode = "delayed_leader_message"
        elif contact_flag:
            access_mode = "physical_contact"
        else:
            access_mode = "none"

        return {
            "time": round(float(sim_time), 6),
            "run_id": self.run_id,
            "sender": sender_name,
            "receiver": receiver_name,
            "same_swarm": int(same_swarm),
            "leader_reference_edge": int(leader_edge),
            "message_received_flag": message_flag,
            "message_age_s": message_age,
            "central_avoidance_x": central[0],
            "central_avoidance_y": central[1],
            "central_avoidance_z": central[2],
            "local_repulsion_x": local[0],
            "local_repulsion_y": local[1],
            "local_repulsion_z": local[2],
            "contact_flag": contact_flag,
            "contact_force": contact_force,
            "delta_target_norm": float(np.linalg.norm(delta_target)),
            "delta_rpm_norm": float(np.linalg.norm(delta_rpm)),
            "sender_swarm_id": sender_swarm,
            "receiver_swarm_id": receiver_swarm,
            "leader_reference_active": int(leader_active),
            "message_used_for_control_flag": message_used,
            "state_access_mode": access_mode,
            "central_avoidance_raw_x": central_raw[0],
            "central_avoidance_raw_y": central_raw[1],
            "central_avoidance_raw_z": central_raw[2],
            "central_avoidance_norm": float(np.linalg.norm(central)),
            "local_repulsion_raw_x": local_raw[0],
            "local_repulsion_raw_y": local_raw[1],
            "local_repulsion_raw_z": local_raw[2],
            "local_repulsion_norm": float(np.linalg.norm(local)),
            "local_repulsion_source_age_s": local_age,
            "repulsion_normalization_scale": float(
                getattr(receiver, "last_repulsion_normalization_scale", 1.0)
            ),
            "building_repulsion_raw_x": building_raw[0],
            "building_repulsion_raw_y": building_raw[1],
            "building_repulsion_raw_z": building_raw[2],
            "building_repulsion_applied_x": building_applied[0],
            "building_repulsion_applied_y": building_applied[1],
            "building_repulsion_applied_z": building_applied[2],
            "total_repulsion_x": total_repulsion[0],
            "total_repulsion_y": total_repulsion[1],
            "total_repulsion_z": total_repulsion[2],
            "contact_count": contact_count,
            "interaction_active": int(mechanism_active),
            "counterfactual_status": counterfactual["status"],
            "delta_target_x": delta_target[0],
            "delta_target_y": delta_target[1],
            "delta_target_z": delta_target[2],
            "delta_target_vel_x": delta_target_vel[0],
            "delta_target_vel_y": delta_target_vel[1],
            "delta_target_vel_z": delta_target_vel[2],
            "delta_rpm_0": delta_rpm[0],
            "delta_rpm_1": delta_rpm[1],
            "delta_rpm_2": delta_rpm[2],
            "delta_rpm_3": delta_rpm[3],
            "control_counterfactual_valid": counterfactual["valid"],
            "formation_control_mode": str(
                getattr(receiver, "formation_control_mode", "none")
            ),
            "relative_position_error_x": relative_error[0],
            "relative_position_error_y": relative_error[1],
            "relative_position_error_z": relative_error[2],
            "relative_position_error_norm": float(np.linalg.norm(relative_error)),
            "relative_velocity_error_x": relative_velocity_error[0],
            "relative_velocity_error_y": relative_velocity_error[1],
            "relative_velocity_error_z": relative_velocity_error[2],
            "formation_attraction_raw_x": attraction_raw[0],
            "formation_attraction_raw_y": attraction_raw[1],
            "formation_attraction_raw_z": attraction_raw[2],
            "formation_attraction_x": attraction[0],
            "formation_attraction_y": attraction[1],
            "formation_attraction_z": attraction[2],
            "formation_attraction_norm": float(np.linalg.norm(attraction)),
            "formation_damping_x": damping[0],
            "formation_damping_y": damping[1],
            "formation_damping_z": damping[2],
            "formation_damping_norm": float(np.linalg.norm(damping)),
            "formation_transport_vx": transport[0],
            "formation_transport_vy": transport[1],
            "formation_transport_vz": transport[2],
            "formation_correction_scale": float(
                getattr(receiver, "last_formation_correction_scale", 1.0)
            ),
            "formation_application_scale": float(
                getattr(receiver, "last_formation_application_scale", 0.0)
            ),
        }

    def _append_matrix_step(self, sim_time: float, names: list[str], rows: list[dict]):
        node_index = {name: index for index, name in enumerate(names)}
        matrices = {
            key: np.zeros((len(names), len(names)), dtype=np.float32)
            for key in self._matrix_channels
        }
        matrices["target_delta_norm"].fill(np.nan)
        matrices["rpm_delta_norm"].fill(np.nan)
        for index in range(len(names)):
            matrices["target_delta_norm"][index, index] = 0.0
            matrices["rpm_delta_norm"][index, index] = 0.0

        for row in rows:
            receiver = node_index[row["receiver"]]
            sender = node_index[row["sender"]]
            matrices["structural"][receiver, sender] = float(row["leader_reference_edge"])
            matrices["active"][receiver, sender] = float(row["interaction_active"])
            matrices["message_used"][receiver, sender] = float(
                row["message_used_for_control_flag"]
            )
            matrices["target_delta_norm"][receiver, sender] = float(
                row["delta_target_norm"]
            )
            matrices["rpm_delta_norm"][receiver, sender] = float(row["delta_rpm_norm"])
            matrices["contact_force"][receiver, sender] = float(row["contact_force"])
            matrices["counterfactual_valid"][receiver, sender] = float(
                row["control_counterfactual_valid"]
            )
            matrices["attraction_norm"][receiver, sender] = float(
                row["formation_attraction_norm"]
            )
            matrices["damping_norm"][receiver, sender] = float(
                row["formation_damping_norm"]
            )
            matrices["repulsion_norm"][receiver, sender] = float(
                row["local_repulsion_norm"]
            )
            matrices["relative_error_norm"][receiver, sender] = float(
                row["relative_position_error_norm"]
            )

        self._matrix_times.append(float(sim_time))
        self._matrix_drone_names = list(names)
        for key, matrix in matrices.items():
            self._matrix_channels[key].append(matrix)

    def log_step(self, sim_time: float, uavs) -> None:
        if self._closed:
            return
        ordered = sorted(uavs, key=lambda item: str(item.name))
        rows = []
        for receiver in ordered:
            for sender in ordered:
                if sender is receiver:
                    continue
                row = self._pair_row(sim_time, sender, receiver)
                self._writer.writerow(row)
                rows.append(row)
                self.rows_written += 1
        self._append_matrix_step(sim_time, [str(item.name) for item in ordered], rows)
        if self.rows_written % 10000 == 0:
            self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        names = self._matrix_drone_names or []
        node_count = len(names)
        arrays = {
            key: (
                np.stack(values).astype(np.float32, copy=False)
                if values
                else np.empty((0, node_count, node_count), dtype=np.float32)
            )
            for key, values in self._matrix_channels.items()
        }
        np.savez_compressed(
            self.matrix_path,
            schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int32),
            times=np.asarray(self._matrix_times, dtype=np.float64),
            drone_names=np.asarray(names),
            matrix_convention=np.asarray("row=receiver,column=sender"),
            **arrays,
        )
        self._closed = True
