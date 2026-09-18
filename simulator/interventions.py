"""Configuration and runtime application of controlled UAV interventions.

The simulator uses one canonical representation for an intervention.  Keeping
normalisation and validation here prevents the batch runner, the simulator and
the learning data exporter from silently interpreting the same event
differently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pybullet as p


_FRAME_ALIASES = {
    "world": "world",
    "global": "world",
    "link": "link",
    "body": "link",
    "local": "link",
}


@dataclass(frozen=True)
class InterventionEvent:
    """A validated, time-bounded force intervention."""

    id: str
    targets: tuple[str, ...]
    start_time: float
    end_time: float
    force: tuple[float, float, float]
    type: str = "controlled_external_force"
    frame: str = "world"
    link_id: int = -1

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def bullet_frame(self) -> int:
        return p.WORLD_FRAME if self.frame == "world" else p.LINK_FRAME

    def is_active(self, sim_time: float) -> bool:
        return self.start_time <= sim_time < self.end_time

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "targets": list(self.targets),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "force": list(self.force),
            "type": self.type,
            "frame": self.frame,
            "link_id": self.link_id,
        }


@dataclass(frozen=True)
class AppliedIntervention:
    """One event-target application at a simulator step."""

    event_index: int
    event_id: str
    target: str
    command_force: tuple[float, float, float]
    world_force: tuple[float, float, float]
    elapsed: float
    remaining: float
    onset: bool
    frame: str


def _targets_from_raw(raw: Mapping) -> tuple[str, ...]:
    targets = (
        raw.get("targets")
        or raw.get("agents")
        or raw.get("agent")
        or raw.get("drone")
        or raw.get("target")
    )
    if isinstance(targets, str):
        targets = [targets]
    if targets is None:
        return ()
    return tuple(
        dict.fromkeys(
            str(target).strip() for target in targets if str(target).strip()
        )
    )


def load_intervention_events(config: object) -> list[InterventionEvent]:
    """Parse the ``interventions`` configuration into canonical events.

    Invalid configured events raise an explicit error instead of being silently
    skipped.  A disabled section still produces an empty event list.
    """

    if isinstance(config, list):
        enabled, raw_events = True, config
    elif isinstance(config, dict):
        enabled = bool(config.get("enabled", False))
        raw_events = config.get("events", [])
    elif config in (None, False):
        return []
    else:
        raise ValueError("interventions must be a mapping or a list of events.")

    if not enabled:
        return []
    if not isinstance(raw_events, list):
        raise ValueError("interventions.events must be a list.")

    events: list[InterventionEvent] = []
    for index, raw in enumerate(raw_events):
        label = f"interventions.events[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a mapping.")

        targets = _targets_from_raw(raw)
        if not targets:
            raise ValueError(f"{label} must contain at least one target.")

        start_time = float(
            raw.get("start_time", raw.get("t_start", raw.get("time", 0.0)))
        )
        if "end_time" in raw:
            end_time = float(raw["end_time"])
        else:
            end_time = start_time + float(raw.get("duration", 0.0))
        if not np.isfinite([start_time, end_time]).all() or start_time < 0.0:
            raise ValueError(f"{label} times must be finite and non-negative.")
        if end_time <= start_time:
            raise ValueError(f"{label} must have a strictly positive duration.")

        force_array = np.asarray(raw.get("force", [0.0, 0.0, 0.0]), dtype=float)
        if force_array.shape != (3,) or not np.isfinite(force_array).all():
            raise ValueError(f"{label}.force must be a finite vector [fx, fy, fz].")

        raw_frame = str(raw.get("frame", "world")).strip().lower()
        if raw_frame not in _FRAME_ALIASES:
            raise ValueError(f"{label}.frame must be 'world' or 'link'.")

        event_id = str(raw.get("id", f"intervention_{index}")).strip()
        if not event_id:
            raise ValueError(f"{label}.id cannot be empty.")
        events.append(
            InterventionEvent(
                id=event_id,
                targets=targets,
                start_time=start_time,
                end_time=end_time,
                force=tuple(float(value) for value in force_array),
                type=str(raw.get("type", "controlled_external_force")),
                frame=_FRAME_ALIASES[raw_frame],
                link_id=int(raw.get("link_id", -1)),
            )
        )

    _validate_unique_ids(events)
    return sorted(events, key=lambda event: (event.start_time, event.id))


def _validate_unique_ids(events: Sequence[InterventionEvent]) -> None:
    counts = Counter(event.id for event in events)
    duplicates = sorted(event_id for event_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate intervention ids: {duplicates}.")


def validate_intervention_targets(
    events: Sequence[InterventionEvent], known_targets: Iterable[str]
) -> None:
    """Validate target names and reject ambiguous overlapping assignments."""

    known = set(known_targets)
    unknown = sorted(
        {target for event in events for target in event.targets if target not in known}
    )
    if unknown:
        raise ValueError(f"Unknown intervention targets: {unknown}.")

    intervals: dict[str, list[InterventionEvent]] = {}
    for event in events:
        for target in event.targets:
            intervals.setdefault(target, []).append(event)
    for target, target_events in intervals.items():
        ordered = sorted(target_events, key=lambda event: event.start_time)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_time < previous.end_time:
                raise ValueError(
                    "Overlapping interventions on the same target are ambiguous: "
                    f"'{previous.id}' and '{current.id}' on '{target}'."
                )


class InterventionRuntime:
    """Apply events and return an auditable record of each applied force."""

    def __init__(self, events: Sequence[InterventionEvent]):
        self.events = list(events)
        self._previously_active: set[tuple[int, str]] = set()

    @staticmethod
    def _force_in_world(agent, event: InterventionEvent) -> np.ndarray:
        force = np.asarray(event.force, dtype=float)
        if event.frame == "world":
            return force
        if event.link_id == -1:
            quaternion = p.getBasePositionAndOrientation(
                agent.bodyId, physicsClientId=agent.physics_client_id
            )[1]
        else:
            quaternion = p.getLinkState(
                agent.bodyId,
                event.link_id,
                physicsClientId=agent.physics_client_id,
            )[1]
        rotation = np.asarray(
            p.getMatrixFromQuaternion(quaternion), dtype=float
        ).reshape(3, 3)
        return rotation @ force

    def apply(self, sim_time: float, uavs: Sequence) -> list[AppliedIntervention]:
        by_name = {str(agent.name): agent for agent in uavs}
        for agent in uavs:
            agent.clear_intervention_state()

        active_keys: set[tuple[int, str]] = set()
        applications: list[AppliedIntervention] = []
        for event_index, event in enumerate(self.events):
            if not event.is_active(sim_time):
                continue
            for target in event.targets:
                agent = by_name[target]
                key = (event_index, target)
                active_keys.add(key)
                world_force = self._force_in_world(agent, event)
                agent.apply_logged_external_force(
                    event.force,
                    intervention_type=event.type,
                    link_id=event.link_id,
                    frame=event.bullet_frame,
                )
                applications.append(
                    AppliedIntervention(
                        event_index=event_index,
                        event_id=event.id,
                        target=target,
                        command_force=event.force,
                        world_force=tuple(float(value) for value in world_force),
                        elapsed=max(0.0, sim_time - event.start_time),
                        remaining=max(0.0, event.end_time - sim_time),
                        onset=key not in self._previously_active,
                        frame=event.frame,
                    )
                )
        self._previously_active = active_keys
        return applications
