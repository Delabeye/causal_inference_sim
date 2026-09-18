"""Ridge VAR estimator used to compute conditional Granger scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "Standardizer":
        values = np.concatenate(arrays, axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(values.mean(axis=0), scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Multi-output ridge regression with an unpenalized final intercept."""

    gram = design.T @ design
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    system = gram + penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def var_design(
    states: np.ndarray,
    controls: np.ndarray | None,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict x[t+1] from x[t], ..., x[t-lags+1] and current controls."""

    if len(states) <= lags:
        raise ValueError(f"Need more than {lags} samples for a VAR({lags}).")
    rows = []
    targets = []
    for time_index in range(lags - 1, len(states) - 1):
        row = [states[time_index - lag] for lag in range(lags)]
        if controls is not None and controls.shape[1]:
            row.append(controls[time_index])
        row.append(np.ones(1, dtype=np.float64))
        rows.append(np.concatenate(row))
        targets.append(states[time_index + 1])
    return np.stack(rows), np.stack(targets)


def concatenate_designs(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    builder,
    *builder_args,
) -> tuple[np.ndarray, np.ndarray]:
    designs = []
    targets = []
    for states, controls in runs:
        design, target = builder(states, controls, *builder_args)
        designs.append(design)
        targets.append(target)
    return np.concatenate(designs), np.concatenate(targets)


@dataclass
class GrangerVAR:
    full_coefficients: np.ndarray
    restricted_coefficients: list[np.ndarray]
    restricted_columns: list[np.ndarray]
    lags: int
    state_dim: int
    control_dim: int
    node_count: int
    state_per_node: int

    def design(self, states: np.ndarray, controls: np.ndarray | None):
        return var_design(states, controls, self.lags)

    def effect_matrix(
        self, states: np.ndarray, controls: np.ndarray | None
    ) -> np.ndarray:
        """Return one run's directed conditional-Granger score matrix."""

        return self.effect_matrix_for_runs([(states, controls)])

    def effect_matrix_for_runs(
        self, runs: list[tuple[np.ndarray, np.ndarray | None]]
    ) -> np.ndarray:
        """Pool residual sums without creating transitions across run boundaries."""

        full_sse = np.zeros(self.node_count, dtype=np.float64)
        restricted_sse = np.zeros(
            (self.node_count, self.node_count), dtype=np.float64
        )
        for states, controls in runs:
            design, target = self.design(states, controls)
            full_error = target - design @ self.full_coefficients
            for receiver in range(self.node_count):
                output_slice = slice(
                    receiver * self.state_per_node,
                    (receiver + 1) * self.state_per_node,
                )
                full_sse[receiver] += np.square(
                    full_error[:, output_slice]
                ).sum()
            for sender in range(self.node_count):
                keep = self.restricted_columns[sender]
                restricted_error = (
                    target - design[:, keep] @ self.restricted_coefficients[sender]
                )
                for receiver in range(self.node_count):
                    output_slice = slice(
                        receiver * self.state_per_node,
                        (receiver + 1) * self.state_per_node,
                    )
                    restricted_sse[receiver, sender] += np.square(
                        restricted_error[:, output_slice]
                    ).sum()

        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float64)
        for sender in range(self.node_count):
            for receiver in range(self.node_count):
                if sender != receiver:
                    raw = np.log(
                        (restricted_sse[receiver, sender] + 1e-12)
                        / (full_sse[receiver] + 1e-12)
                    )
                    matrix[receiver, sender] = max(0.0, 1.0 - np.exp(-max(raw, 0.0)))
        return matrix


def fit_granger_var(
    runs: list[tuple[np.ndarray, np.ndarray | None]],
    lags: int,
    ridge: float,
    node_count: int,
    state_per_node: int,
) -> GrangerVAR:
    design, target = concatenate_designs(runs, var_design, lags)
    full = ridge_fit(design, target, ridge)
    state_dim = runs[0][0].shape[1]
    control_dim = 0 if runs[0][1] is None else runs[0][1].shape[1]
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
        restricted_coefficients.append(ridge_fit(design[:, keep], target, ridge))
    return GrangerVAR(
        full_coefficients=full,
        restricted_coefficients=restricted_coefficients,
        restricted_columns=restricted_columns,
        lags=lags,
        state_dim=state_dim,
        control_dim=control_dim,
        node_count=node_count,
        state_per_node=state_per_node,
    )


__all__ = [
    "GrangerVAR",
    "Standardizer",
    "fit_granger_var",
    "ridge_fit",
    "var_design",
]
