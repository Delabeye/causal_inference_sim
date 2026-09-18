"""RiTINI-inspired Graph Neural ODE implemented with stock PyTorch.

The architecture follows the public RiTINI implementation:

* an LSTM encodes a per-node history into the initial latent ODE state;
* a directed multi-head graph-attention layer computes the vector field;
* Euler or fixed-step RK4 integrates that field in continuous time;
* the attention coefficients are exposed as the dynamic interaction graph.

The upstream package uses torch-geometric and torchdiffeq.  This local version
keeps the same modeling ingredients without adding those dependencies, and
supports batched UAV trajectories in the repository's native tensor layout.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def complete_directed_mask(node_count: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a receiver-by-sender mask with every off-diagonal edge enabled."""

    return ~torch.eye(node_count, dtype=torch.bool, device=device)


def off_diagonal_values(matrices: torch.Tensor) -> torch.Tensor:
    """Flatten receiver-major off-diagonal entries from [..., N, N]."""

    node_count = matrices.shape[-1]
    mask = complete_directed_mask(node_count, matrices.device)
    return matrices[..., mask]


class TemporalHistoryEncoder(nn.Module):
    """Encode recent node histories and expose a learned lag distribution."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        history_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.history_steps = int(history_steps)
        self.lag_logits = nn.Parameter(torch.zeros(history_steps))
        self.input_projection = nn.Linear(state_dim, latent_dim)
        self.lstm = nn.LSTM(
            latent_dim,
            latent_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # history: [batch, nodes, history, state_features]
        if history.shape[2] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} history steps, got {history.shape[2]}."
            )
        batch, nodes, steps, _ = history.shape
        lag_attention = F.softmax(self.lag_logits, dim=0)
        projected = torch.tanh(self.input_projection(history))
        # Multiplication by H preserves the input scale under a uniform prior.
        projected = projected * lag_attention.view(1, 1, steps, 1) * steps
        projected = self.dropout(projected).reshape(batch * nodes, steps, -1)
        _, (hidden, _) = self.lstm(projected)
        return hidden[-1].reshape(batch, nodes, -1), lag_attention


class DirectedGraphAttention(nn.Module):
    """Dense directed GAT with A[receiver, sender] semantics."""

    def __init__(
        self,
        latent_dim: int,
        heads: int,
        dropout: float,
        negative_slope: float,
        candidate_mask: torch.Tensor,
    ):
        super().__init__()
        if candidate_mask.ndim != 2 or candidate_mask.shape[0] != candidate_mask.shape[1]:
            raise ValueError("candidate_mask must be a square receiver-by-sender matrix.")
        if not candidate_mask.bool().any(dim=-1).all():
            raise ValueError("Every receiver needs at least one candidate sender.")
        self.latent_dim = int(latent_dim)
        self.heads = int(heads)
        self.head_dim = latent_dim // heads
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.output = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = float(negative_slope)
        self.register_buffer("candidate_mask", candidate_mask.bool().clone())

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = latent.shape
        query = self.query(latent).reshape(batch, nodes, self.heads, self.head_dim)
        key = self.key(latent).reshape(batch, nodes, self.heads, self.head_dim)
        value = self.value(latent).reshape(batch, nodes, self.heads, self.head_dim)
        # receiver i queries sender j: [batch, head, receiver, sender]
        scores = torch.einsum("bihd,bjhd->bhij", query, key)
        scores = scores / math.sqrt(self.head_dim)
        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)
        scores = scores.masked_fill(
            ~self.candidate_mask.view(1, 1, nodes, nodes),
            torch.finfo(scores.dtype).min,
        )
        attention = F.softmax(scores, dim=-1)
        # Dropout belongs to message aggregation, not to the graph exported for
        # interpretation/regularization; the latter remains row-normalized.
        message_attention = self.dropout(attention)
        aggregated = torch.einsum("bhij,bjhd->bihd", message_attention, value)
        aggregated = aggregated.reshape(batch, nodes, self.latent_dim)
        return self.output(aggregated), attention.mean(dim=1)


class GraphODEVectorField(nn.Module):
    """Compute dz/dt with a GAT followed by RiTINI's nonlinear field MLP."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int,
        heads: int,
        dropout: float,
        negative_slope: float,
        candidate_mask: torch.Tensor,
    ):
        super().__init__()
        self.attention = DirectedGraphAttention(
            latent_dim,
            heads,
            dropout,
            negative_slope,
            candidate_mask,
        )
        self.field = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.nfe = 0

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.nfe += 1
        relational, attention = self.attention(latent)
        derivative = self.field(torch.cat([latent, relational], dim=-1))
        return derivative, attention


class FixedStepODEBlock(nn.Module):
    """Differentiable Euler/RK4 integration with optional internal substeps."""

    def __init__(self, field: GraphODEVectorField, method: str, substeps: int):
        super().__init__()
        if method not in {"euler", "rk4"}:
            raise ValueError("method must be 'euler' or 'rk4'.")
        if substeps < 1:
            raise ValueError("substeps must be positive.")
        self.field = field
        self.method = method
        self.substeps = int(substeps)

    @staticmethod
    def _scaled(step: torch.Tensor, derivative: torch.Tensor) -> torch.Tensor:
        if step.ndim == 0:
            return step * derivative
        return step.reshape(-1, 1, 1) * derivative

    def _single_step(
        self, latent: torch.Tensor, step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k1, attention = self.field(latent)
        if self.method == "euler":
            return latent + self._scaled(step, k1), attention
        k2, _ = self.field(latent + self._scaled(step * 0.5, k1))
        k3, _ = self.field(latent + self._scaled(step * 0.5, k2))
        k4, _ = self.field(latent + self._scaled(step, k3))
        increment = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return latent + self._scaled(step, increment), attention

    def forward(
        self, latent: torch.Tensor, step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        substep = step / self.substeps
        attention = None
        for _ in range(self.substeps):
            latent, attention = self._single_step(latent, substep)
        assert attention is not None
        return latent, attention


class RiTINISwarm(nn.Module):
    """Batched RiTINI baseline for trajectories shaped [B, N, T, F]."""

    def __init__(
        self,
        state_dim: int,
        node_count: int,
        history_steps: int,
        latent_dim: int,
        attention_heads: int,
        field_hidden: int,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        ode_method: str = "rk4",
        ode_substeps: int = 1,
        integration_dt: float = 0.1,
        use_observed_dt: bool = False,
        time_scale: float = 1.0,
        candidate_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.node_count = int(node_count)
        self.history_steps = int(history_steps)
        self.integration_dt = float(integration_dt)
        self.use_observed_dt = bool(use_observed_dt)
        self.time_scale = float(time_scale)
        if candidate_mask is None:
            candidate_mask = complete_directed_mask(node_count)
        if tuple(candidate_mask.shape) != (node_count, node_count):
            raise ValueError("candidate_mask does not match node_count.")
        self.history_encoder = TemporalHistoryEncoder(
            state_dim,
            latent_dim,
            history_steps,
            dropout,
        )
        vector_field = GraphODEVectorField(
            latent_dim,
            field_hidden,
            attention_heads,
            dropout,
            negative_slope,
            candidate_mask,
        )
        self.ode = FixedStepODEBlock(vector_field, ode_method, ode_substeps)
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_dim),
        )
        self.register_buffer("candidate_mask", candidate_mask.bool().clone())

    def _step_sizes(
        self, states: torch.Tensor, times: torch.Tensor | None
    ) -> torch.Tensor:
        predictions = states.shape[2] - self.history_steps
        if not self.use_observed_dt:
            return states.new_full((states.shape[0], predictions), self.integration_dt)
        if times is None or tuple(times.shape) != (states.shape[0], states.shape[2]):
            actual = None if times is None else tuple(times.shape)
            raise ValueError(
                f"times has shape {actual}; expected {(states.shape[0], states.shape[2])}."
            )
        deltas = times[:, self.history_steps :] - times[:, self.history_steps - 1 : -1]
        if bool((deltas <= 0).any()):
            raise ValueError("times must be strictly increasing.")
        return deltas.to(states.dtype) / self.time_scale

    def forward(
        self,
        states: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, nodes, time, features].")
        if states.shape[1] != self.node_count or states.shape[-1] != self.state_dim:
            raise ValueError("states do not match the configured node/state dimensions.")
        if states.shape[2] <= self.history_steps:
            raise ValueError("states must contain at least one step after the history.")
        latent, lag_attention = self.history_encoder(
            states[:, :, : self.history_steps]
        )
        step_sizes = self._step_sizes(states, times)
        predictions = []
        attentions = []
        latents = []
        self.ode.field.nfe = 0
        for step_index in range(step_sizes.shape[1]):
            latent, attention = self.ode(latent, step_sizes[:, step_index])
            predictions.append(self.readout(latent))
            attentions.append(attention)
            latents.append(latent)
        return {
            "prediction": torch.stack(predictions, dim=2),
            "attention": torch.stack(attentions, dim=1),
            "edge_attention": off_diagonal_values(torch.stack(attentions, dim=1)),
            "latent": torch.stack(latents, dim=2),
            "lag_attention": lag_attention,
            "nfe": torch.tensor(self.ode.field.nfe, device=states.device),
        }


def ritini_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    attention: torch.Tensor,
    candidate_mask: torch.Tensor,
    prior_adjacency: torch.Tensor | None,
    lambda_velocity: float,
    lambda_prior: float,
    lambda_entropy: float,
    lambda_graph_smooth: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trajectory reconstruction plus RiTINI graph regularizers."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ.")
    position_dims = min(3, prediction.shape[-1])
    position_mse = F.mse_loss(
        prediction[..., :position_dims], target[..., :position_dims]
    )
    if prediction.shape[-1] > position_dims:
        velocity_mse = F.mse_loss(
            prediction[..., position_dims:], target[..., position_dims:]
        )
    else:
        velocity_mse = prediction.new_zeros(())

    valid_attention = attention[..., candidate_mask]
    eps = torch.finfo(attention.dtype).eps
    entropy = -(valid_attention * torch.log(valid_attention.clamp_min(eps))).mean()
    if attention.shape[1] > 1:
        graph_smooth = (attention[:, 1:] - attention[:, :-1]).square().mean()
    else:
        graph_smooth = attention.new_zeros(())

    if prior_adjacency is None:
        prior_loss = attention.new_zeros(())
    else:
        if tuple(prior_adjacency.shape) != tuple(candidate_mask.shape):
            raise ValueError("prior_adjacency shape does not match the graph.")
        prior = prior_adjacency[candidate_mask].to(attention.dtype)
        prior = prior.view(*([1] * (valid_attention.ndim - 1)), -1)
        prior = prior.expand_as(valid_attention)
        prior_loss = F.binary_cross_entropy(
            valid_attention.clamp(min=eps, max=1.0 - eps), prior
        )

    total = (
        position_mse
        + lambda_velocity * velocity_mse
        + lambda_prior * prior_loss
        + lambda_entropy * entropy
        + lambda_graph_smooth * graph_smooth
    )
    return total, {
        "position_mse": position_mse,
        "velocity_mse": velocity_mse,
        "prior_loss": prior_loss,
        "attention_entropy": entropy,
        "graph_smooth": graph_smooth,
    }


__all__ = [
    "RiTINISwarm",
    "complete_directed_mask",
    "off_diagonal_values",
    "ritini_loss",
]
