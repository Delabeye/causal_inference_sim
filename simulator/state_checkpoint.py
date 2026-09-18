"""Exact in-memory checkpoints for deterministic counterfactual forks."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pybullet as p


UAV_REFERENCE_FIELDS = {
    "p",
    "planning_thread",
    "planner",
    "swarm_leader_ref",
    "sub_socket",
    "pub_socket",
    "radar_sub_socket",
    "zmq_ctx",
    # Branches use their own tensor logger; copying the cumulative CSV buffer
    # at every snapshot wastes memory and has no effect on dynamics.
    "log_data_buffer",
}
SWARM_REFERENCE_FIELDS = {
    "agents",
    "leader",
    "followers",
    "radar",
    "reference_agents",
    "_direct_uav_message_buffer",
    "client_ctx",
    "sub_socket",
    "pub_socket",
}
RADAR_REFERENCE_FIELDS = {"p", "targets"}


def _copy_attributes(obj, excluded: set[str]) -> dict:
    copied = {}
    for name, value in obj.__dict__.items():
        if name in excluded:
            continue
        try:
            copied[name] = copy.deepcopy(value)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot checkpoint {type(obj).__name__}.{name}: {exc}"
            ) from exc
    return copied


def _restore_attributes(obj, state: dict, excluded: set[str]) -> None:
    removable = set(obj.__dict__) - set(state) - excluded
    for name in removable:
        del obj.__dict__[name]
    for name, value in state.items():
        obj.__dict__[name] = copy.deepcopy(value)


@dataclass
class SimulationCheckpoint:
    """PyBullet and Python state captured at one simulator instant."""

    manager: object
    bullet_state_id: int
    sim_time: float
    random_state: object
    numpy_random_state: tuple
    uav_states: dict[str, dict]
    radar_states: dict[str, dict]
    swarm_states: list[dict]
    swarm_direct_queues: list[list[tuple]]
    last_control_indices: object
    last_intervention_applications: list
    runtime_active_keys: set
    _released: bool = False

    @classmethod
    def capture(cls, manager) -> "SimulationCheckpoint":
        if not bool(getattr(manager, "deterministic_execution", False)):
            raise RuntimeError(
                "Counterfactual checkpoints require deterministic_execution=true."
            )
        bullet_state_id = p.saveState(
            physicsClientId=int(manager.physics_client_id)
        )
        uav_states = {}
        radar_states = {}
        for agent in manager.agents:
            name = str(agent.name)
            if getattr(agent, "type", "") == "uav":
                uav_states[name] = _copy_attributes(agent, UAV_REFERENCE_FIELDS)
            else:
                radar_states[name] = _copy_attributes(agent, RADAR_REFERENCE_FIELDS)

        swarm_states = []
        swarm_direct_queues = []
        for swarm in manager.swarms:
            swarm_states.append(_copy_attributes(swarm, SWARM_REFERENCE_FIELDS))
            queue = []
            for visible_time, topic, receiver, payload in getattr(
                swarm, "_direct_uav_message_buffer", []
            ):
                queue.append(
                    (
                        float(visible_time),
                        str(topic),
                        str(receiver.name),
                        copy.deepcopy(payload),
                    )
                )
            swarm_direct_queues.append(queue)

        return cls(
            manager=manager,
            bullet_state_id=int(bullet_state_id),
            sim_time=float(manager.sim_time),
            random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            uav_states=uav_states,
            radar_states=radar_states,
            swarm_states=swarm_states,
            swarm_direct_queues=swarm_direct_queues,
            last_control_indices=copy.deepcopy(
                manager._last_relational_control_indices
            ),
            last_intervention_applications=copy.deepcopy(
                manager._last_intervention_applications
            ),
            runtime_active_keys=set(
                getattr(manager.intervention_runtime, "_previously_active", set())
            ),
        )

    def restore(self) -> None:
        if self._released:
            raise RuntimeError("Cannot restore a released simulation checkpoint.")
        manager = self.manager
        p.restoreState(
            stateId=self.bullet_state_id,
            physicsClientId=int(manager.physics_client_id),
        )
        agents_by_name = {str(agent.name): agent for agent in manager.agents}
        for name, state in self.uav_states.items():
            _restore_attributes(
                agents_by_name[name], state, UAV_REFERENCE_FIELDS
            )
        for name, state in self.radar_states.items():
            _restore_attributes(
                agents_by_name[name], state, RADAR_REFERENCE_FIELDS
            )

        for swarm, state, queue in zip(
            manager.swarms, self.swarm_states, self.swarm_direct_queues
        ):
            _restore_attributes(swarm, state, SWARM_REFERENCE_FIELDS)
            swarm._direct_uav_message_buffer = [
                (
                    visible_time,
                    topic,
                    agents_by_name[receiver_name],
                    copy.deepcopy(payload),
                )
                for visible_time, topic, receiver_name, payload in queue
            ]

        manager.sim_time = self.sim_time
        manager._last_relational_control_indices = copy.deepcopy(
            self.last_control_indices
        )
        manager._last_intervention_applications = copy.deepcopy(
            self.last_intervention_applications
        )
        manager.intervention_runtime._previously_active = set(
            self.runtime_active_keys
        )
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)

    def release(self) -> None:
        if self._released:
            return
        try:
            p.removeState(
                self.bullet_state_id,
                physicsClientId=int(self.manager.physics_client_id),
            )
        finally:
            self._released = True
