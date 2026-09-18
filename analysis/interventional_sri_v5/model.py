"""Full SRI encoder-decoder augmented with paired physical interventions.

The implementation follows the mechanisms described in the SRI article:

* per-time node and directed-edge GNN features;
* multi-head attention between candidate relations of the same receiver;
* causal forward LSTMs for learned priors ``p_G`` and ``p_A``;
* reverse LSTMs for training posteriors ``q_G`` and ``q_A``;
* categorical edge types and categorical node states sampled with
  Gumbel-Softmax;
* one message MLP per active edge type and autoregressive multi-step decoding.

The project-specific addition is a pair of baseline/intervention rollouts.  A
current intervention is never visible to the encoder.  Both branches share the
same posterior graph, node state, decoder memory and random numbers at the fork
time; only their decoder control input differs.  Future priors are causal and
may consequently diverge after the intervention changes predicted states.

Tensor convention: states are ``[B,N,T,D]`` and directed edges use the
receiver-major ordering of ``relation_matrices`` (matrix row=receiver,
column=sender).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _check_history_shape(
    name: str,
    values: torch.Tensor,
    batch: int,
    nodes: int,
    steps: int,
) -> None:
    if values.ndim != 4 or tuple(values.shape[:3]) != (batch, nodes, steps):
        raise ValueError(
            f"{name} must have shape [batch, nodes, {steps}, features]; "
            f"got {tuple(values.shape)}."
        )


def _gumbel_noise_like(values: torch.Tensor) -> torch.Tensor:
    uniform = torch.rand_like(values).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(uniform))


def _sample_categorical(
    logits: torch.Tensor,
    *,
    temperature: float,
    stochastic: bool,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard straight-through categorical sample with optional shared noise."""

    if not stochastic:
        return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)
    if noise is None:
        noise = _gumbel_noise_like(logits)
    if tuple(noise.shape) != tuple(logits.shape):
        raise ValueError("Gumbel noise must have the same shape as logits.")
    soft = F.softmax((logits + noise) / temperature, dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


def _binary_from_types(values: torch.Tensor) -> torch.Tensor:
    no_edge = values[..., :1]
    active = values[..., 1:].sum(dim=-1, keepdim=True)
    return torch.cat([no_edge, active], dim=-1)


class SRITemporalEncoder(nn.Module):
    """SRI forward-prior/backward-posterior encoder for edges and nodes."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        intervention_dim: int,
        hidden: int,
        attention_heads: int,
        num_edge_types: int,
        num_node_states: int,
        dropout: float,
        strength_floor: float,
        initial_edge_probability: float,
    ):
        super().__init__()
        if hidden % attention_heads:
            raise ValueError("hidden must be divisible by attention_heads.")
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.hidden = int(hidden)
        self.num_edge_types = int(num_edge_types)
        self.num_node_states = int(num_node_states)
        self.strength_floor = float(strength_floor)

        self.node_input = nn.Sequential(
            nn.Linear(state_dim + context_dim + intervention_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.relation_attention = nn.MultiheadAttention(
            hidden,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.relation_norm = nn.LayerNorm(hidden)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

        # Equations 16--23 of SRI: independent recurrent distributions for
        # graph relations (G) and node states (A).
        self.edge_prior_cell = nn.LSTMCell(hidden + num_edge_types, hidden)
        self.edge_prior_head = nn.Linear(hidden, num_edge_types)
        self.edge_backward_cell = nn.LSTMCell(hidden, hidden)
        self.edge_posterior_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, num_edge_types),
        )
        self.node_prior_cell = nn.LSTMCell(hidden + num_node_states, hidden)
        self.node_prior_head = nn.Linear(hidden, num_node_states)
        self.node_backward_cell = nn.LSTMCell(hidden, hidden)
        self.node_posterior_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, num_node_states),
        )

        active_each = initial_edge_probability / (num_edge_types - 1)
        edge_prior = torch.tensor(
            [1.0 - initial_edge_probability]
            + [active_each] * (num_edge_types - 1),
            dtype=torch.float32,
        )
        self.register_buffer("initial_edge_prior", edge_prior)
        self.register_buffer(
            "initial_node_prior",
            torch.full((num_node_states,), 1.0 / num_node_states),
        )
        # These are only initialization biases.  Unlike v3/v4, the KL target
        # thereafter is the learned, trajectory-dependent forward prior.
        with torch.no_grad():
            self.edge_prior_head.bias.copy_(edge_prior.log())
            self.node_prior_head.bias.copy_(self.initial_node_prior.log())
        nn.init.normal_(self.edge_prior_head.weight, std=1e-3)
        nn.init.normal_(self.node_prior_head.weight, std=1e-3)

    @staticmethod
    def _incoming_attention_mask(rel_rec: torch.Tensor) -> torch.Tensor:
        receivers = rel_rec.argmax(dim=-1)
        return receivers[:, None] != receivers[None, :]

    def _node_features(
        self,
        states: torch.Tensor,
        context: torch.Tensor,
        interventions: torch.Tensor,
    ) -> torch.Tensor:
        return self.node_input(torch.cat([states, context, interventions], dim=-1))

    def _relation_features(
        self,
        node_features: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return strength-aware edge features for a sequence or one step."""

        one_step = node_features.ndim == 3
        if one_step:
            node_features = node_features.unsqueeze(2)
        batch, nodes, steps, hidden = node_features.shape
        time_major = node_features.permute(0, 2, 1, 3).reshape(
            batch * steps, nodes, hidden
        )
        senders = torch.matmul(rel_send, time_major)
        receivers = torch.matmul(rel_rec, time_major)
        raw = self.edge_embedding(torch.cat([senders, receivers], dim=-1))
        attended, weights = self.relation_attention(
            raw,
            raw,
            raw,
            attn_mask=self._incoming_attention_mask(rel_rec),
            need_weights=True,
            average_attn_weights=True,
        )
        strength_features = self.relation_norm(raw + attended)
        strength = torch.sigmoid(self.strength_head(strength_features)).squeeze(-1)
        strength = self.strength_floor + (1.0 - self.strength_floor) * strength
        edges = strength_features.reshape(batch, steps, -1, hidden).permute(0, 2, 1, 3)
        strength = strength.reshape(batch, steps, -1).permute(0, 2, 1)
        weights = weights.reshape(batch, steps, weights.shape[-2], weights.shape[-1])
        if one_step:
            return edges[:, :, 0], strength[:, :, 0], weights[:, 0]
        return edges, strength, weights

    @staticmethod
    def _run_forward_prior(
        features: torch.Tensor,
        cell: nn.LSTMCell,
        head: nn.Linear,
        initial_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, entities, steps, hidden = features.shape
        flat = batch * entities
        recurrent_h = features.new_zeros(flat, hidden)
        recurrent_c = features.new_zeros(flat, hidden)
        previous = initial_probability.to(features).view(1, -1).expand(flat, -1)
        logits = []
        hidden_states = []
        for time_index in range(steps):
            current = features[:, :, time_index].reshape(flat, hidden)
            recurrent_h, recurrent_c = cell(
                torch.cat([current, previous], dim=-1),
                (recurrent_h, recurrent_c),
            )
            current_logits = head(recurrent_h)
            previous = F.softmax(current_logits, dim=-1)
            logits.append(current_logits.reshape(batch, entities, -1))
            hidden_states.append(recurrent_h.reshape(batch, entities, hidden))
        return (
            torch.stack(logits, dim=2),
            torch.stack(hidden_states, dim=2),
            (
                recurrent_h.reshape(batch, entities, hidden),
                recurrent_c.reshape(batch, entities, hidden),
            ),
        )

    @staticmethod
    def _run_backward(
        features: torch.Tensor,
        cell: nn.LSTMCell,
    ) -> torch.Tensor:
        batch, entities, steps, hidden = features.shape
        flat = batch * entities
        recurrent_h = features.new_zeros(flat, hidden)
        recurrent_c = features.new_zeros(flat, hidden)
        hidden_states: list[torch.Tensor | None] = [None] * steps
        for time_index in range(steps - 1, -1, -1):
            current = features[:, :, time_index].reshape(flat, hidden)
            recurrent_h, recurrent_c = cell(current, (recurrent_h, recurrent_c))
            hidden_states[time_index] = recurrent_h.reshape(batch, entities, hidden)
        return torch.stack([value for value in hidden_states if value is not None], dim=2)

    def forward(
        self,
        history_states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        history_context: torch.Tensor | None = None,
        history_interventions: torch.Tensor | None = None,
        *,
        temperature: float,
        stochastic: bool,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        if history_states.ndim != 4:
            raise ValueError("history_states must have shape [B,N,T,D].")
        batch, nodes, steps, features = history_states.shape
        if features != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {features}.")
        if tuple(rel_rec.shape) != tuple(rel_send.shape) or rel_rec.shape[1] != nodes:
            raise ValueError("Relation matrices do not match the node count.")
        if history_context is None:
            history_context = history_states.new_zeros(
                batch, nodes, steps, self.context_dim
            )
        if history_interventions is None:
            history_interventions = history_states.new_zeros(
                batch, nodes, steps, self.intervention_dim
            )
        _check_history_shape("history_context", history_context, batch, nodes, steps)
        _check_history_shape(
            "history_interventions", history_interventions, batch, nodes, steps
        )
        if history_context.shape[-1] != self.context_dim:
            raise ValueError("history_context has the wrong feature dimension.")
        if history_interventions.shape[-1] != self.intervention_dim:
            raise ValueError("history_interventions has the wrong feature dimension.")

        node_features = self._node_features(
            history_states, history_context, history_interventions
        )
        edge_features, strength, attention = self._relation_features(
            node_features, rel_rec, rel_send
        )
        edge_prior_logits, edge_prior_hidden, edge_prior_state = self._run_forward_prior(
            edge_features,
            self.edge_prior_cell,
            self.edge_prior_head,
            self.initial_edge_prior,
        )
        node_prior_logits, node_prior_hidden, node_prior_state = self._run_forward_prior(
            node_features,
            self.node_prior_cell,
            self.node_prior_head,
            self.initial_node_prior,
        )
        edge_backward = self._run_backward(edge_features, self.edge_backward_cell)
        node_backward = self._run_backward(node_features, self.node_backward_cell)
        edge_posterior_logits = self.edge_posterior_head(
            torch.cat([edge_backward, edge_prior_hidden], dim=-1)
        )
        node_posterior_logits = self.node_posterior_head(
            torch.cat([node_backward, node_prior_hidden], dim=-1)
        )
        edge_posterior_probability = F.softmax(edge_posterior_logits, dim=-1)
        node_posterior_probability = F.softmax(node_posterior_logits, dim=-1)
        edge_posterior_sample = _sample_categorical(
            edge_posterior_logits,
            temperature=temperature,
            stochastic=stochastic,
        )
        node_posterior_sample = _sample_categorical(
            node_posterior_logits,
            temperature=temperature,
            stochastic=stochastic,
        )
        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "relation_attention_history": attention,
            "strength_history": strength,
            "edge_prior_logits_history": edge_prior_logits,
            "edge_prior_probability_history": F.softmax(edge_prior_logits, dim=-1),
            "edge_posterior_logits_history": edge_posterior_logits,
            "edge_posterior_probability_history": edge_posterior_probability,
            "edge_posterior_sample_history": edge_posterior_sample,
            "node_prior_logits_history": node_prior_logits,
            "node_prior_probability_history": F.softmax(node_prior_logits, dim=-1),
            "node_posterior_logits_history": node_posterior_logits,
            "node_posterior_probability_history": node_posterior_probability,
            "node_posterior_sample_history": node_posterior_sample,
            "edge_prior_state": edge_prior_state,
            "node_prior_state": node_prior_state,
        }

    def future_features(
        self,
        states: torch.Tensor,
        context: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The current intervention is deliberately absent.  It can influence
        # future graph priors only through the state produced by the decoder.
        zero_intervention = states.new_zeros(
            (*states.shape[:-1], self.intervention_dim)
        )
        node_features = self._node_features(states, context, zero_intervention)
        edge_features, strength, _ = self._relation_features(
            node_features, rel_rec, rel_send
        )
        return node_features, edge_features, strength

    def future_prior_step(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_state: tuple[torch.Tensor, torch.Tensor],
        edge_state: tuple[torch.Tensor, torch.Tensor],
        previous_node_sample: torch.Tensor,
        previous_edge_sample: torch.Tensor,
        *,
        temperature: float,
        stochastic: bool,
        node_noise: torch.Tensor | None,
        edge_noise: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        batch, nodes, hidden = node_features.shape
        edges = edge_features.shape[1]
        node_h, node_c = node_state
        edge_h, edge_c = edge_state
        node_h_flat, node_c_flat = self.node_prior_cell(
            torch.cat(
                [
                    node_features.reshape(batch * nodes, hidden),
                    previous_node_sample.reshape(batch * nodes, -1),
                ],
                dim=-1,
            ),
            (
                node_h.reshape(batch * nodes, hidden),
                node_c.reshape(batch * nodes, hidden),
            ),
        )
        edge_h_flat, edge_c_flat = self.edge_prior_cell(
            torch.cat(
                [
                    edge_features.reshape(batch * edges, hidden),
                    previous_edge_sample.reshape(batch * edges, -1),
                ],
                dim=-1,
            ),
            (
                edge_h.reshape(batch * edges, hidden),
                edge_c.reshape(batch * edges, hidden),
            ),
        )
        node_logits = self.node_prior_head(node_h_flat).reshape(
            batch, nodes, self.num_node_states
        )
        edge_logits = self.edge_prior_head(edge_h_flat).reshape(
            batch, edges, self.num_edge_types
        )
        node_sample = _sample_categorical(
            node_logits,
            temperature=temperature,
            stochastic=stochastic,
            noise=node_noise,
        )
        edge_sample = _sample_categorical(
            edge_logits,
            temperature=temperature,
            stochastic=stochastic,
            noise=edge_noise,
        )
        return {
            "node_logits": node_logits,
            "node_probability": F.softmax(node_logits, dim=-1),
            "node_sample": node_sample,
            "node_state": (
                node_h_flat.reshape(batch, nodes, hidden),
                node_c_flat.reshape(batch, nodes, hidden),
            ),
            "edge_logits": edge_logits,
            "edge_probability": F.softmax(edge_logits, dim=-1),
            "edge_sample": edge_sample,
            "edge_state": (
                edge_h_flat.reshape(batch, edges, hidden),
                edge_c_flat.reshape(batch, edges, hidden),
            ),
        }


class TypedInterventionalDecoder(nn.Module):
    """SRI type-specific messages plus local state and intervention dynamics."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        intervention_dim: int,
        num_edge_types: int,
        num_node_states: int,
        hidden: int,
        message_hidden: int,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.num_edge_types = int(num_edge_types)
        self.num_node_states = int(num_node_states)
        self.hidden = int(hidden)
        self.hidden_initialization = nn.Linear(state_dim + context_dim, hidden)
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden, message_hidden),
                    nn.ELU(),
                    nn.Linear(message_hidden, message_hidden),
                    nn.LayerNorm(message_hidden),
                    nn.Tanh(),
                )
                for _ in range(num_edge_types - 1)
            ]
        )
        for network in self.messages:
            nn.init.zeros_(network[2].weight)
            nn.init.zeros_(network[2].bias)
        self.node_state_embedding = nn.Parameter(
            torch.empty(num_node_states, hidden)
        )
        nn.init.normal_(self.node_state_embedding, std=1.0 / math.sqrt(hidden))
        self.intervention_embedding = nn.Sequential(
            nn.Linear(intervention_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.update = nn.GRUCell(
            state_dim + message_hidden + hidden + hidden + context_dim,
            hidden,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, state_dim),
        )

    def initialize_hidden(
        self, states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(
            self.hidden_initialization(torch.cat([states, context], dim=-1))
        )

    def step(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor,
        edge_types: torch.Tensor,
        edge_strength: torch.Tensor,
        node_states: torch.Tensor,
        interventions: torch.Tensor,
        context: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        senders = torch.matmul(rel_send, hidden)
        receivers = torch.matmul(rel_rec, hidden)
        pair = torch.cat([senders, receivers], dim=-1)
        typed_messages = torch.stack(
            [network(pair) for network in self.messages], dim=-2
        )
        active_types = edge_types[..., 1:].unsqueeze(-1)
        messages = (typed_messages * active_types).sum(dim=-2)
        messages = messages * edge_strength.unsqueeze(-1)
        incoming = torch.matmul(rel_rec.transpose(0, 1), messages)
        intervention_hidden = self.intervention_embedding(interventions)
        node_latent = torch.matmul(node_states, self.node_state_embedding)
        update_input = torch.cat(
            [states, incoming, node_latent, intervention_hidden, context], dim=-1
        )
        next_hidden = self.update(
            update_input.reshape(-1, update_input.shape[-1]),
            hidden.reshape(-1, hidden.shape[-1]),
        ).reshape_as(hidden)
        prediction = states + self.readout(next_hidden)
        return prediction, next_hidden, incoming

    def reconstruct_history(
        self,
        history_states: torch.Tensor,
        history_context: torch.Tensor,
        history_interventions: torch.Tensor,
        edge_types: torch.Tensor,
        edge_strength: torch.Tensor,
        node_states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = history_states[:, :, 0]
        hidden = self.initialize_hidden(current, history_context[:, :, 0])
        predictions = []
        for time_index in range(history_states.shape[2] - 1):
            prediction, hidden, _ = self.step(
                current,
                hidden,
                edge_types[:, :, time_index],
                edge_strength[:, :, time_index],
                node_states[:, :, time_index],
                history_interventions[:, :, time_index],
                history_context[:, :, time_index],
                rel_rec,
                rel_send,
            )
            predictions.append(prediction)
            # Teacher forcing trains every historical posterior rather than
            # allowing only the final graph to receive a reconstruction signal.
            current = history_states[:, :, time_index + 1]
        return torch.stack(predictions, dim=2), hidden


class InterventionalSRIV5(nn.Module):
    """Full dynamic SRI backbone with paired causal rollouts."""

    graph_is_dynamic = True

    def __init__(
        self,
        state_dim: int,
        context_dim: int = 0,
        intervention_dim: int = 4,
        num_edge_types: int = 3,
        num_node_states: int = 3,
        encoder_hidden: int = 128,
        attention_heads: int = 4,
        decoder_hidden: int = 128,
        message_hidden: int = 128,
        dropout: float = 0.1,
        strength_floor: float = 0.1,
        temperature: float = 1.0,
        hard_edge_threshold: float = 0.5,
        initial_edge_probability: float = 0.25,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.num_edge_types = int(num_edge_types)
        self.num_node_states = int(num_node_states)
        self.temperature = float(temperature)
        self.hard_edge_threshold = float(hard_edge_threshold)
        self.encoder = SRITemporalEncoder(
            state_dim,
            context_dim,
            intervention_dim,
            encoder_hidden,
            attention_heads,
            num_edge_types,
            num_node_states,
            dropout,
            strength_floor,
            initial_edge_probability,
        )
        self.decoder = TypedInterventionalDecoder(
            state_dim,
            context_dim,
            intervention_dim,
            num_edge_types,
            num_node_states,
            decoder_hidden,
            message_hidden,
        )
        if encoder_hidden != decoder_hidden:
            raise ValueError(
                "v5 currently requires encoder_hidden == decoder_hidden so the "
                "history decoder and future prior share a compact state size."
            )

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def set_edge_decision_threshold(self, threshold: float) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("edge threshold must lie in (0, 1).")
        self.hard_edge_threshold = float(threshold)

    def _select_edge_types(
        self,
        probability: torch.Tensor,
        sample: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "scheduled":
            return sample
        if mode == "soft":
            return probability
        if mode == "zero":
            result = torch.zeros_like(probability)
            result[..., 0] = 1.0
            return result
        if mode in {"hard", "shuffled"}:
            existence = 1.0 - probability[..., 0]
            active_index = probability[..., 1:].argmax(dim=-1) + 1
            selected = torch.where(
                existence >= self.hard_edge_threshold,
                active_index,
                torch.zeros_like(active_index),
            )
            result = F.one_hot(selected, self.num_edge_types).to(probability.dtype)
            if mode == "shuffled":
                edge_dimension = -3 if result.ndim == 4 else -2
                result = torch.roll(result, shifts=1, dims=edge_dimension)
            return result
        raise ValueError(
            "decoder_graph_mode must be scheduled, soft, hard, zero or shuffled."
        )

    @staticmethod
    def _select_node_states(
        probability: torch.Tensor,
        sample: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode in {"scheduled"}:
            return sample
        if mode == "soft":
            return probability
        return F.one_hot(probability.argmax(dim=-1), probability.shape[-1]).to(
            probability.dtype
        )

    @staticmethod
    def _effective_edge(edge_types: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        return edge_types[..., 1:].sum(dim=-1) * strength

    def _rollout_branch(
        self,
        initial_state: torch.Tensor,
        initial_decoder_hidden: torch.Tensor,
        initial_edge_probability: torch.Tensor,
        initial_edge_sample: torch.Tensor,
        initial_strength: torch.Tensor,
        initial_node_probability: torch.Tensor,
        initial_node_sample: torch.Tensor,
        initial_edge_prior_state: tuple[torch.Tensor, torch.Tensor],
        initial_node_prior_state: tuple[torch.Tensor, torch.Tensor],
        interventions: torch.Tensor,
        contexts: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        mode: str,
        edge_noises: list[torch.Tensor],
        node_noises: list[torch.Tensor],
        diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        current = initial_state
        decoder_hidden = initial_decoder_hidden
        edge_probability = initial_edge_probability
        edge_sample = initial_edge_sample
        edge_strength = initial_strength
        node_probability = initial_node_probability
        node_sample = initial_node_sample
        edge_prior_state = tuple(value.clone() for value in initial_edge_prior_state)
        node_prior_state = tuple(value.clone() for value in initial_node_prior_state)

        predictions = []
        incoming_messages = []
        edge_probabilities = []
        edge_samples = []
        strengths = []
        effective_samples = []
        effective_means = []
        decoder_edges = []
        edge_type_probabilities = []
        node_state_probabilities = []
        for step_index in range(interventions.shape[2]):
            selected_edge = self._select_edge_types(
                edge_probability, edge_sample, mode
            )
            selected_node = self._select_node_states(
                node_probability, node_sample, mode
            )
            selected_strength = edge_strength
            if mode == "shuffled":
                selected_strength = torch.roll(selected_strength, shifts=1, dims=-1)
            current, decoder_hidden, incoming = self.decoder.step(
                current,
                decoder_hidden,
                selected_edge,
                selected_strength,
                selected_node,
                interventions[:, :, step_index],
                contexts[:, :, step_index],
                rel_rec,
                rel_send,
            )
            predictions.append(current)
            if diagnostics:
                incoming_messages.append(incoming)
                edge_type_probabilities.append(edge_probability)
                node_state_probabilities.append(node_probability)
                edge_probabilities.append(_binary_from_types(edge_probability))
                edge_samples.append(_binary_from_types(selected_edge))
                strengths.append(selected_strength)
                effective_samples.append(
                    self._effective_edge(selected_edge, selected_strength)
                )
                effective_means.append(
                    (1.0 - edge_probability[..., 0]) * edge_strength
                )
                decoder_edges.append(
                    self._effective_edge(selected_edge, selected_strength)
                )

            if step_index + 1 == interventions.shape[2]:
                continue
            node_features, edge_features, edge_strength = self.encoder.future_features(
                current,
                contexts[:, :, step_index],
                rel_rec,
                rel_send,
            )
            prior = self.encoder.future_prior_step(
                node_features,
                edge_features,
                node_prior_state,
                edge_prior_state,
                selected_node,
                selected_edge,
                temperature=self.temperature,
                stochastic=self.training,
                node_noise=node_noises[step_index] if self.training else None,
                edge_noise=edge_noises[step_index] if self.training else None,
            )
            node_probability = prior["node_probability"]
            node_sample = prior["node_sample"]
            node_prior_state = prior["node_state"]
            edge_probability = prior["edge_probability"]
            edge_sample = prior["edge_sample"]
            edge_prior_state = prior["edge_state"]

        result = {"prediction": torch.stack(predictions, dim=2)}
        if diagnostics:
            result.update(
                {
                    "incoming_message": torch.stack(incoming_messages, dim=2),
                    "graph_probability": torch.stack(edge_probabilities, dim=1),
                    "graph_sample": torch.stack(edge_samples, dim=1),
                    "graph_strength": torch.stack(strengths, dim=1),
                    "graph_effective_sample": torch.stack(effective_samples, dim=1),
                    "graph_effective_mean": torch.stack(effective_means, dim=1),
                    "graph_decoder_edge": torch.stack(decoder_edges, dim=1),
                    "edge_type_probability": torch.stack(
                        edge_type_probabilities, dim=1
                    ),
                    "node_state_probability": torch.stack(
                        node_state_probabilities, dim=1
                    ),
                }
            )
        return result

    def _rollout_pair(
        self,
        first_interventions: torch.Tensor,
        second_interventions: torch.Tensor,
        branch_arguments: dict,
        *,
        diagnostics: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run two independent branches in one larger batch.

        The operation is mathematically identical to two calls to
        ``_rollout_branch``.  Shared Gumbel noise is duplicated explicitly so
        that the baseline/intervention comparison keeps the same random draws.
        """

        batch = first_interventions.shape[0]

        def duplicate(values: torch.Tensor) -> torch.Tensor:
            return torch.cat([values, values], dim=0)

        joint_arguments = {
            **branch_arguments,
            "initial_state": duplicate(branch_arguments["initial_state"]),
            "initial_decoder_hidden": duplicate(
                branch_arguments["initial_decoder_hidden"]
            ),
            "initial_edge_probability": duplicate(
                branch_arguments["initial_edge_probability"]
            ),
            "initial_edge_sample": duplicate(
                branch_arguments["initial_edge_sample"]
            ),
            "initial_strength": duplicate(branch_arguments["initial_strength"]),
            "initial_node_probability": duplicate(
                branch_arguments["initial_node_probability"]
            ),
            "initial_node_sample": duplicate(
                branch_arguments["initial_node_sample"]
            ),
            "initial_edge_prior_state": tuple(
                duplicate(value)
                for value in branch_arguments["initial_edge_prior_state"]
            ),
            "initial_node_prior_state": tuple(
                duplicate(value)
                for value in branch_arguments["initial_node_prior_state"]
            ),
            "contexts": duplicate(branch_arguments["contexts"]),
            "edge_noises": [
                duplicate(value) for value in branch_arguments["edge_noises"]
            ],
            "node_noises": [
                duplicate(value) for value in branch_arguments["node_noises"]
            ],
        }
        joint = self._rollout_branch(
            interventions=torch.cat(
                [first_interventions, second_interventions], dim=0
            ),
            diagnostics=diagnostics,
            **joint_arguments,
        )
        first = {key: value[:batch] for key, value in joint.items()}
        second = {key: value[batch:] for key, value in joint.items()}
        return first, second

    def forward(
        self,
        history_states: torch.Tensor,
        intervention_future: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        history_context: torch.Tensor | None = None,
        future_context: torch.Tensor | None = None,
        history_interventions: torch.Tensor | None = None,
        baseline_interventions: torch.Tensor | None = None,
        decoder_graph_mode: str = "scheduled",
        include_graph_ablations: bool = False,
        rollout_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        if intervention_future.ndim != 4:
            raise ValueError("intervention_future must have shape [B,N,H,U].")
        batch, nodes, horizon, intervention_features = intervention_future.shape
        if intervention_features != self.intervention_dim:
            raise ValueError("intervention_future has the wrong feature dimension.")
        if tuple(history_states.shape[:2]) != (batch, nodes):
            raise ValueError("History and rollout batch/node dimensions differ.")
        history_steps = history_states.shape[2]
        if history_context is None:
            history_context = history_states.new_zeros(
                batch, nodes, history_steps, self.context_dim
            )
        if future_context is None:
            future_context = history_states.new_zeros(
                batch, nodes, horizon, self.context_dim
            )
        if history_interventions is None:
            history_interventions = history_states.new_zeros(
                batch, nodes, history_steps, self.intervention_dim
            )
        if baseline_interventions is None:
            baseline_interventions = torch.zeros_like(intervention_future)
        if tuple(future_context.shape) != (batch, nodes, horizon, self.context_dim):
            raise ValueError("future_context has the wrong shape.")
        if tuple(baseline_interventions.shape) != tuple(intervention_future.shape):
            raise ValueError("baseline_interventions must match intervention_future.")

        encoded = self.encoder(
            history_states,
            rel_rec,
            rel_send,
            history_context,
            history_interventions,
            temperature=self.temperature,
            stochastic=self.training,
        )
        edge_q = encoded["edge_posterior_probability_history"]
        edge_sample_history = encoded["edge_posterior_sample_history"]
        node_q = encoded["node_posterior_probability_history"]
        node_sample_history = encoded["node_posterior_sample_history"]
        strength_history = encoded["strength_history"]
        history_edge = self._select_edge_types(
            edge_q,
            edge_sample_history,
            decoder_graph_mode,
        )
        history_node = self._select_node_states(
            node_q,
            node_sample_history,
            decoder_graph_mode,
        )
        history_strength = strength_history
        if decoder_graph_mode == "shuffled":
            history_strength = torch.roll(history_strength, shifts=1, dims=1)
        history_prediction, decoder_hidden = self.decoder.reconstruct_history(
            history_states,
            history_context,
            history_interventions,
            history_edge,
            history_strength,
            history_node,
            rel_rec,
            rel_send,
        )

        initial_edge_probability = edge_q[:, :, -1]
        initial_edge_sample = edge_sample_history[:, :, -1]
        initial_strength = strength_history[:, :, -1]
        initial_node_probability = node_q[:, :, -1]
        initial_node_sample = node_sample_history[:, :, -1]
        edge_count = initial_edge_probability.shape[1]
        edge_noises = [
            _gumbel_noise_like(initial_edge_probability)
            for _ in range(max(horizon - 1, 0))
        ]
        node_noises = [
            _gumbel_noise_like(initial_node_probability)
            for _ in range(max(horizon - 1, 0))
        ]
        branch_arguments = dict(
            initial_state=history_states[:, :, -1],
            initial_decoder_hidden=decoder_hidden,
            initial_edge_probability=initial_edge_probability,
            initial_edge_sample=initial_edge_sample,
            initial_strength=initial_strength,
            initial_node_probability=initial_node_probability,
            initial_node_sample=initial_node_sample,
            initial_edge_prior_state=encoded["edge_prior_state"],
            initial_node_prior_state=encoded["node_prior_state"],
            contexts=future_context,
            rel_rec=rel_rec,
            rel_send=rel_send,
            mode=decoder_graph_mode,
            edge_noises=edge_noises,
            node_noises=node_noises,
        )
        baseline, intervention = self._rollout_pair(
            baseline_interventions,
            intervention_future,
            branch_arguments,
            diagnostics=rollout_diagnostics,
        )
        existence_probability = _binary_from_types(initial_edge_probability)
        existence_sample = _binary_from_types(initial_edge_sample)
        result = {
            **{key: value for key, value in encoded.items() if torch.is_tensor(value)},
            "history_prediction": history_prediction,
            "edge_type_probability": initial_edge_probability,
            "node_state_probability": initial_node_probability,
            "existence_probability": existence_probability,
            "existence_sample": existence_sample,
            "strength": initial_strength,
            "effective_edge_sample": existence_sample[..., 1] * initial_strength,
            "effective_edge_mean": existence_probability[..., 1] * initial_strength,
            "baseline_prediction": baseline["prediction"],
            "intervention_prediction": intervention["prediction"],
            "effect_prediction": intervention["prediction"] - baseline["prediction"],
            "hard_edge_threshold": history_states.new_tensor(self.hard_edge_threshold),
            "decoder_graph_blend": history_states.new_tensor(1.0),
            "intervention_target_node": intervention_future.abs().sum(dim=(2, 3))
            > 1e-8,
        }
        if rollout_diagnostics:
            for name in (
                "incoming_message",
                "graph_probability",
                "graph_sample",
                "graph_strength",
                "graph_effective_sample",
                "graph_effective_mean",
                "graph_decoder_edge",
                "edge_type_probability",
                "node_state_probability",
            ):
                result[f"baseline_{name}"] = baseline[name]
                result[f"intervention_{name}"] = intervention[name]
        if include_graph_ablations:
            zero_arguments = {**branch_arguments, "mode": "zero"}
            zero_baseline, zero_intervention = self._rollout_pair(
                baseline_interventions,
                intervention_future,
                zero_arguments,
                diagnostics=False,
            )
            shuffled_arguments = {**branch_arguments, "mode": "shuffled"}
            shuffled_baseline, shuffled_intervention = self._rollout_pair(
                baseline_interventions,
                intervention_future,
                shuffled_arguments,
                diagnostics=False,
            )
            result.update(
                {
                    "no_graph_effect_prediction": (
                        zero_intervention["prediction"] - zero_baseline["prediction"]
                    ),
                    "shuffled_effect_prediction": (
                        shuffled_intervention["prediction"]
                        - shuffled_baseline["prediction"]
                    ),
                }
            )
        if edge_count != rel_rec.shape[0]:
            raise RuntimeError("Encoded edge count differs from relation matrices.")
        return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum()
    numerator = (values * expanded).sum()
    return numerator / denominator.clamp_min(1.0)


@dataclass(frozen=True)
class SRIV5LossWeights:
    lambda_history: float = 0.5
    lambda_intervention: float = 1.0
    lambda_effect: float = 1.0
    lambda_non_target_effect: float = 1.0
    lambda_graph_necessity: float = 0.5
    lambda_graph_contrast: float = 0.5
    beta_edge_kl: float = 0.1
    beta_node_kl: float = 0.1
    graph_margin: float = 0.05
    non_target_min_effect: float = 0.01


def _categorical_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    q = q.clamp_min(1e-9)
    p = p.clamp_min(1e-9)
    return (q * (q.log() - p.log())).sum(dim=-1).mean()


def interventional_sri_v5_loss(
    output: dict[str, torch.Tensor],
    history_states: torch.Tensor,
    baseline_target: torch.Tensor,
    intervention_target: torch.Tensor,
    paired_mask: torch.Tensor,
    *,
    state_sigma: float,
    effect_scale: torch.Tensor,
    weights: SRIV5LossWeights,
    kl_scale: float = 1.0,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SRI ELBO terms plus paired interventional and graph-necessity losses."""

    history_error = (
        output["history_prediction"] - history_states[:, :, 1:]
    ).square()
    baseline_error = (output["baseline_prediction"] - baseline_target).square()
    intervention_error = (
        output["intervention_prediction"] - intervention_target
    ).square()
    true_effect = intervention_target - baseline_target
    effect_error = (output["effect_prediction"] - true_effect).square()
    if effect_scale.ndim != 1 or effect_scale.shape[0] != baseline_target.shape[-1]:
        raise ValueError("effect_scale must contain one value per state feature.")
    standardized_effect = effect_error / effect_scale.to(effect_error).square().view(
        1, 1, 1, -1
    )
    horizon_scale = None
    if horizon_weights is not None:
        if horizon_weights.ndim != 1 or horizon_weights.shape[0] != baseline_target.shape[2]:
            raise ValueError("horizon_weights must contain one value per rollout step.")
        horizon_scale = horizon_weights.to(baseline_target).view(1, 1, -1, 1)
        baseline_error = baseline_error * horizon_scale
        intervention_error = intervention_error * horizon_scale
        effect_error = effect_error * horizon_scale
        standardized_effect = standardized_effect * horizon_scale

    history_nll = history_error.mean() / (2.0 * state_sigma**2)
    baseline_nll = baseline_error.mean() / (2.0 * state_sigma**2)
    intervention_nll = _masked_mean(intervention_error, paired_mask) / (
        2.0 * state_sigma**2
    )
    effect_mse = _masked_mean(effect_error, paired_mask)
    effect_standardized_mse = _masked_mean(standardized_effect, paired_mask)

    target_nodes = output["intervention_target_node"].bool()
    responsive = (
        true_effect.square().mean(dim=(2, 3)).sqrt()
        >= weights.non_target_min_effect
    )
    propagation_mask = paired_mask[:, None] & ~target_nodes & responsive
    propagation_nodes = propagation_mask.float().sum()
    has_propagation = (propagation_nodes > 0).to(effect_error.dtype)
    target_energy = true_effect.square()
    if horizon_scale is not None:
        target_energy = target_energy * horizon_scale
    energy = _masked_mean(target_energy, propagation_mask).clamp_min(1e-8)
    full_relative = _masked_mean(effect_error, propagation_mask) / energy
    non_target_relative = full_relative * has_propagation
    graph_necessity = effect_error.new_zeros(())
    graph_contrast = effect_error.new_zeros(())
    if "no_graph_effect_prediction" in output:
        no_graph_error = (
            output["no_graph_effect_prediction"] - true_effect
        ).square()
        if horizon_scale is not None:
            no_graph_error = no_graph_error * horizon_scale
        no_graph_relative = _masked_mean(no_graph_error, propagation_mask) / energy
        graph_necessity = (
            F.relu(full_relative - no_graph_relative + weights.graph_margin)
            * has_propagation
        )
    if "shuffled_effect_prediction" in output:
        shuffled_error = (
            output["shuffled_effect_prediction"] - true_effect
        ).square()
        if horizon_scale is not None:
            shuffled_error = shuffled_error * horizon_scale
        shuffled_relative = _masked_mean(shuffled_error, propagation_mask) / energy
        graph_contrast = (
            F.relu(full_relative - shuffled_relative + weights.graph_margin)
            * has_propagation
        )

    edge_kl = _categorical_kl(
        output["edge_posterior_probability_history"],
        output["edge_prior_probability_history"],
    )
    node_kl = _categorical_kl(
        output["node_posterior_probability_history"],
        output["node_prior_probability_history"],
    )
    loss = (
        weights.lambda_history * history_nll
        + baseline_nll
        + weights.lambda_intervention * intervention_nll
        + weights.lambda_effect * effect_standardized_mse
        + weights.lambda_non_target_effect * non_target_relative
        + weights.lambda_graph_necessity * graph_necessity
        + weights.lambda_graph_contrast * graph_contrast
        + float(kl_scale)
        * (weights.beta_edge_kl * edge_kl + weights.beta_node_kl * node_kl)
    )
    return loss, {
        "history_nll": history_nll,
        "baseline_nll": baseline_nll,
        "intervention_nll": intervention_nll,
        "effect_mse": effect_mse,
        "effect_standardized_mse": effect_standardized_mse,
        "non_target_effect_relative": non_target_relative,
        "graph_necessity": graph_necessity,
        "graph_contrast": graph_contrast,
        "edge_kl": edge_kl,
        "node_kl": node_kl,
        "kl_scale": loss.new_tensor(float(kl_scale)),
        "propagation_nodes": propagation_nodes,
        "paired_fraction": paired_mask.float().mean(),
    }


__all__ = [
    "InterventionalSRIV5",
    "SRITemporalEncoder",
    "SRIV5LossWeights",
    "TypedInterventionalDecoder",
    "interventional_sri_v5_loss",
]
