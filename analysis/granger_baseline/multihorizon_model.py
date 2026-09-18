"""Conditional Ridge regressions for multi-horizon state changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ridge_fit


def _target_columns(
    node_count: int,
    state_per_node: int,
    components: tuple[int, int, int],
) -> np.ndarray:
    if state_per_node < 6:
        raise ValueError(
            "Multi-horizon Granger expects 3D position followed by 3D velocity."
        )
    return np.asarray(
        [
            node * state_per_node + component
            for node in range(node_count)
            for component in components
        ],
        dtype=int,
    )


def state_delta_design(
    states: np.ndarray,
    controls: np.ndarray | None,
    lags: int,
    horizon: int,
    node_count: int,
    state_per_node: int,
    target_components: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a 3D state change from lagged states and controls at time t."""

    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if len(states) <= lags + horizon:
        raise ValueError(
            f"Need more than lags+horizon={lags + horizon} samples."
        )
    target_columns = _target_columns(
        node_count, state_per_node, target_components
    )
    rows = []
    targets = []
    for time_index in range(lags - 1, len(states) - horizon):
        row = [states[time_index - lag] for lag in range(lags)]
        if controls is not None and controls.shape[1]:
            row.append(controls[time_index])
        row.append(np.ones(1, dtype=np.float64))
        rows.append(np.concatenate(row))
        targets.append(
            states[time_index + horizon, target_columns]
            - states[time_index, target_columns]
        )
    return np.stack(rows), np.stack(targets)


def velocity_delta_design(
    states: np.ndarray,
    controls: np.ndarray | None,
    lags: int,
    horizon: int,
    node_count: int,
    state_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict v[t+h]-v[t] from lagged states and current controls."""

    return state_delta_design(
        states,
        controls,
        lags,
        horizon,
        node_count,
        state_per_node,
        (3, 4, 5),
    )


def position_delta_design(
    states: np.ndarray,
    controls: np.ndarray | None,
    lags: int,
    horizon: int,
    node_count: int,
    state_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict p[t+h]-p[t] from lagged states and current controls."""

    return state_delta_design(
        states,
        controls,
        lags,
        horizon,
        node_count,
        state_per_node,
        (0, 1, 2),
    )


def _concatenate_designs(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    lags: int,
    horizon: int,
    node_count: int,
    state_per_node: int,
    target_components: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    designs = []
    targets = []
    for states, controls in runs:
        design, target = state_delta_design(
            states,
            controls,
            lags,
            horizon,
            node_count,
            state_per_node,
            target_components,
        )
        designs.append(design)
        targets.append(target)
    return np.concatenate(designs), np.concatenate(targets)


@dataclass
class HorizonDeltaRegressor:
    full_coefficients: np.ndarray
    restricted_coefficients: list[np.ndarray]
    restricted_columns: list[np.ndarray]
    lags: int
    horizon: int
    node_count: int
    state_per_node: int
    target_components: tuple[int, int, int]

    def design(
        self, states: np.ndarray, controls: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        return state_delta_design(
            states,
            controls,
            self.lags,
            self.horizon,
            self.node_count,
            self.state_per_node,
            self.target_components,
        )

    def effect_matrix_for_runs(
        self, runs: list[tuple[np.ndarray, np.ndarray | None]]
    ) -> np.ndarray:
        full_sse = np.zeros(self.node_count, dtype=np.float64)
        restricted_sse = np.zeros(
            (self.node_count, self.node_count), dtype=np.float64
        )
        for states, controls in runs:
            design, target = self.design(states, controls)
            full_error = target - design @ self.full_coefficients
            for receiver in range(self.node_count):
                output_slice = slice(3 * receiver, 3 * (receiver + 1))
                full_sse[receiver] += np.square(
                    full_error[:, output_slice]
                ).sum()
            for sender in range(self.node_count):
                keep = self.restricted_columns[sender]
                restricted_error = (
                    target - design[:, keep] @ self.restricted_coefficients[sender]
                )
                for receiver in range(self.node_count):
                    output_slice = slice(3 * receiver, 3 * (receiver + 1))
                    restricted_sse[receiver, sender] += np.square(
                        restricted_error[:, output_slice]
                    ).sum()

        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float64)
        for receiver in range(self.node_count):
            for sender in range(self.node_count):
                if receiver == sender:
                    continue
                raw = np.log(
                    (restricted_sse[receiver, sender] + 1e-12)
                    / (full_sse[receiver] + 1e-12)
                )
                matrix[receiver, sender] = max(
                    0.0, 1.0 - np.exp(-max(raw, 0.0))
                )
        return matrix

    def effect_matrix(
        self, states: np.ndarray, controls: np.ndarray | None
    ) -> np.ndarray:
        return self.effect_matrix_for_runs([(states, controls)])


@dataclass
class MultiHorizonDeltaGranger:
    regressors: dict[int, HorizonDeltaRegressor]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.regressors))

    def effect_matrices_for_runs(
        self, runs: list[tuple[np.ndarray, np.ndarray | None]]
    ) -> dict[int, np.ndarray]:
        return {
            horizon: self.regressors[horizon].effect_matrix_for_runs(runs)
            for horizon in self.horizons
        }

    def effect_matrices(
        self, states: np.ndarray, controls: np.ndarray | None
    ) -> dict[int, np.ndarray]:
        return self.effect_matrices_for_runs([(states, controls)])


def fit_multi_horizon_delta_granger(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    lags: int,
    horizons: tuple[int, ...],
    ridge: float,
    node_count: int,
    state_per_node: int,
    target_components: tuple[int, int, int],
) -> MultiHorizonDeltaGranger:
    state_dim = runs[0][0].shape[1]
    regressors = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        design, target = _concatenate_designs(
            runs,
            lags,
            horizon,
            node_count,
            state_per_node,
            target_components,
        )
        full = ridge_fit(design, target, ridge)
        restricted_coefficients = []
        restricted_columns = []
        for sender in range(node_count):
            keep = np.ones(design.shape[1], dtype=bool)
            sender_columns = []
            for lag in range(lags):
                start = lag * state_dim + sender * state_per_node
                sender_columns.extend(range(start, start + state_per_node))
            keep[np.asarray(sender_columns, dtype=int)] = False
            restricted_columns.append(keep)
            restricted_coefficients.append(
                ridge_fit(design[:, keep], target, ridge)
            )
        regressors[horizon] = HorizonDeltaRegressor(
            full_coefficients=full,
            restricted_coefficients=restricted_coefficients,
            restricted_columns=restricted_columns,
            lags=lags,
            horizon=horizon,
            node_count=node_count,
            state_per_node=state_per_node,
            target_components=target_components,
        )
    return MultiHorizonDeltaGranger(regressors=regressors)


def fit_multi_horizon_velocity_granger(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    lags: int,
    horizons: tuple[int, ...],
    ridge: float,
    node_count: int,
    state_per_node: int,
) -> MultiHorizonDeltaGranger:
    return fit_multi_horizon_delta_granger(
        runs, lags, horizons, ridge, node_count, state_per_node, (3, 4, 5)
    )


def fit_multi_horizon_position_granger(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    lags: int,
    horizons: tuple[int, ...],
    ridge: float,
    node_count: int,
    state_per_node: int,
) -> MultiHorizonDeltaGranger:
    return fit_multi_horizon_delta_granger(
        runs, lags, horizons, ridge, node_count, state_per_node, (0, 1, 2)
    )


# Backward-compatible names for imports of the previous velocity-only API.
HorizonVelocityRegressor = HorizonDeltaRegressor
MultiHorizonVelocityGranger = MultiHorizonDeltaGranger


__all__ = [
    "HorizonDeltaRegressor",
    "HorizonVelocityRegressor",
    "MultiHorizonDeltaGranger",
    "MultiHorizonVelocityGranger",
    "fit_multi_horizon_delta_granger",
    "fit_multi_horizon_position_granger",
    "fit_multi_horizon_velocity_granger",
    "position_delta_design",
    "state_delta_design",
    "velocity_delta_design",
]
