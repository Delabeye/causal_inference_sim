"""Interventional SRI with paired factual/counterfactual rollouts.

The model is the executable version of the theoretical architecture used in
this project.  It keeps SRI's relation interaction attention, but uses the
simpler graph ontology required by the simulator:

* ``Z[receiver, sender]`` is a binary edge-existence variable;
* ``S[receiver, sender]`` is a continuous strength in ``[0, 1]``;
* the effective graph is ``A = Z * S``;
* the current intervention is visible to the decoder, never to the encoder;
* baseline and intervention rollouts share the same posterior sample, initial
  state, initial decoder memory, and parameters.

Tensor convention
-----------------
States use ``[batch, nodes, time, features]``.  Directed edge vectors are in
receiver-major order, matching ``relation_matrices`` and the simulator's
``A[receiver, sender]`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _check_history_shape(name: str, values: torch.Tensor, batch: int, nodes: int, steps: int) -> None:
    if values.ndim != 4 or tuple(values.shape[:3]) != (batch, nodes, steps):
        raise ValueError(
            f"{name} must have shape [batch, nodes, {steps}, features]; "
            f"got {tuple(values.shape)}."
        )


class InterventionalSRIEncoder(nn.Module):
    """Infer edge existence and strength from pre-intervention history only.

    The temporal GRU first summarizes each node history.  A node-to-edge MLP
    constructs one candidate representation per ordered pair.  Multi-head
    attention then lets candidate senders of the same receiver interact, as in
    SRI's relationship-strength block.  Two heads parameterize the posterior:

    ``q(Z|H)``
        Categorical ``{no-edge, edge}``, sampled with Gumbel-Softmax.
    ``S(H)``
        Deterministic conditional edge strength in ``[0, 1]``.

    The descriptor of the *current* intervention is deliberately absent from
    this API.  Only interventions already present in the history may be passed
    through ``history_interventions``.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        intervention_dim: int,
        hidden: int,
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        if hidden % attention_heads:
            raise ValueError("hidden must be divisible by attention_heads.")
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.hidden = int(hidden)
        node_input_dim = state_dim + context_dim + intervention_dim
        self.node_input = nn.Sequential(
            nn.Linear(node_input_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
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
        self.existence_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2)
        )
        self.strength_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )

    @staticmethod
    def _incoming_attention_mask(rel_rec: torch.Tensor) -> torch.Tensor:
        """Block attention between edges having different receivers."""

        receiver = rel_rec.argmax(dim=-1)
        return receiver[:, None] != receiver[None, :]

    def forward(
        self,
        history_states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        history_context: torch.Tensor | None,
        history_interventions: torch.Tensor | None,
        temperature: float,
        hard: bool,
        stochastic: bool,
    ) -> dict[str, torch.Tensor]:
        if history_states.ndim != 4:
            raise ValueError(
                "history_states must have shape [batch, nodes, time, features]."
            )
        batch, nodes, steps, features = history_states.shape
        if features != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {features}."
            )
        if tuple(rel_rec.shape) != tuple(rel_send.shape) or rel_rec.shape[1] != nodes:
            raise ValueError("Relation matrices do not match the number of nodes.")

        if history_context is None:
            history_context = history_states.new_zeros(batch, nodes, steps, self.context_dim)
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

        node_inputs = torch.cat(
            [history_states, history_context, history_interventions], dim=-1
        )
        projected = self.node_input(node_inputs).reshape(batch * nodes, steps, self.hidden)
        _, node_hidden = self.temporal(projected)
        node_hidden = node_hidden[-1].reshape(batch, nodes, self.hidden)

        senders = torch.matmul(rel_send, node_hidden)
        receivers = torch.matmul(rel_rec, node_hidden)
        edges = self.edge_embedding(torch.cat([senders, receivers], dim=-1))
        attended, attention = self.relation_attention(
            edges,
            edges,
            edges,
            attn_mask=self._incoming_attention_mask(rel_rec),
            need_weights=True,
            average_attn_weights=True,
        )
        edge_hidden = self.relation_norm(edges + attended)
        existence_logits = self.existence_head(edge_hidden)
        existence_probability = F.softmax(existence_logits, dim=-1)
        if stochastic:
            existence_sample = F.gumbel_softmax(
                existence_logits, tau=temperature, hard=hard, dim=-1
            )
        else:
            existence_sample = F.one_hot(
                existence_probability.argmax(dim=-1), num_classes=2
            ).to(existence_probability.dtype)
        strength = torch.sigmoid(self.strength_head(edge_hidden)).squeeze(-1)
        effective_sample = existence_sample[..., 1] * strength
        effective_mean = existence_probability[..., 1] * strength
        return {
            "node_embedding": node_hidden,
            "edge_embedding": edge_hidden,
            "relation_attention": attention,
            "existence_logits": existence_logits,
            "existence_probability": existence_probability,
            "existence_sample": existence_sample,
            "strength": strength,
            "effective_edge_sample": effective_sample,
            "effective_edge_mean": effective_mean,
        }


class DynamicGraphPrior(nn.Module):
    """Causal recurrent prior for graph evolution during a predicted rollout.

    At transition ``t -> t+1`` the prior observes only the newly predicted
    node state, its previous edge memory and the previous graph.  It therefore
    never consumes a future ground-truth state.  Existence and strength are
    residual updates, which makes the initial posterior the graph at the
    snapshot and lets the topology evolve smoothly afterwards.
    """

    def __init__(
        self,
        state_dim: int,
        encoder_hidden: int,
        hidden: int,
        delta_scale: float,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.delta_scale = float(delta_scale)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU()
        )
        self.hidden_initialization = nn.Linear(encoder_hidden, hidden)
        self.recurrent = nn.GRUCell(2 * hidden + 3, hidden)
        self.existence_delta = nn.Linear(hidden, 2)
        self.strength_delta = nn.Linear(hidden, 1)
        nn.init.normal_(self.existence_delta.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.existence_delta.bias)
        nn.init.normal_(self.strength_delta.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.strength_delta.bias)

    def initialize_hidden(self, edge_embedding: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden_initialization(edge_embedding))

    @staticmethod
    def _sample_existence(
        logits: torch.Tensor,
        gumbel_noise: torch.Tensor | None,
        temperature: float,
        hard: bool,
        stochastic: bool,
    ) -> torch.Tensor:
        if not stochastic:
            return F.one_hot(logits.argmax(dim=-1), num_classes=2).to(logits.dtype)
        if gumbel_noise is None or tuple(gumbel_noise.shape) != tuple(logits.shape):
            raise ValueError("Shared Gumbel noise must match dynamic edge logits.")
        soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        if not hard:
            return soft
        discrete = F.one_hot(soft.argmax(dim=-1), num_classes=2).to(soft.dtype)
        return discrete - soft.detach() + soft

    def step(
        self,
        states: torch.Tensor,
        previous_probability: torch.Tensor,
        previous_strength: torch.Tensor,
        hidden: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        *,
        gumbel_noise: torch.Tensor | None,
        temperature: float,
        hard: bool,
        stochastic: bool,
    ) -> dict[str, torch.Tensor]:
        node = self.state_embedding(states)
        senders = torch.matmul(rel_send, node)
        receivers = torch.matmul(rel_rec, node)
        recurrent_input = torch.cat(
            [senders, receivers, previous_probability, previous_strength.unsqueeze(-1)],
            dim=-1,
        )
        next_hidden = self.recurrent(
            recurrent_input.reshape(-1, recurrent_input.shape[-1]),
            hidden.reshape(-1, hidden.shape[-1]),
        ).reshape_as(hidden)
        previous_logits = previous_probability.clamp_min(1e-7).log()
        logits = previous_logits + self.delta_scale * self.existence_delta(next_hidden)
        probability = F.softmax(logits, dim=-1)
        previous_strength_logit = torch.logit(
            previous_strength.clamp(min=1e-5, max=1.0 - 1e-5)
        )
        strength = torch.sigmoid(
            previous_strength_logit
            + self.delta_scale * self.strength_delta(next_hidden).squeeze(-1)
        )
        sample = self._sample_existence(
            logits,
            gumbel_noise,
            temperature,
            hard,
            stochastic,
        )
        return {
            "hidden": next_hidden,
            "existence_logits": logits,
            "existence_probability": probability,
            "existence_sample": sample,
            "strength": strength,
            "effective_edge_sample": sample[..., 1] * strength,
            "effective_edge_mean": probability[..., 1] * strength,
        }


class InterventionalMessageDecoder(nn.Module):
    """Shared recurrent decoder conditioned on graph and intervention.

    The graph has a direct operational meaning: ``effective_edge`` multiplies
    every sender-to-receiver message before aggregation.  The intervention
    descriptor is encoded separately and injected into the receiving node's
    update.  Consequently ``U=0`` recovers the nominal SRI-style decoder.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        intervention_dim: int,
        hidden: int,
        message_hidden: int,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.hidden = int(hidden)
        self.message_hidden = int(message_hidden)
        self.hidden_initialization = nn.Linear(state_dim + context_dim, hidden)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden, message_hidden),
            nn.ELU(),
            nn.Linear(message_hidden, message_hidden),
            # A bounded message prevents the message MLP from compensating an
            # arbitrarily small edge strength with arbitrarily large values.
            # The continuous strength therefore remains an operational gain.
            nn.LayerNorm(message_hidden),
            nn.Tanh(),
        )
        self.intervention_embedding = nn.Sequential(
            nn.Linear(intervention_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        recurrent_input = state_dim + message_hidden + hidden + context_dim
        self.update = nn.GRUCell(recurrent_input, hidden)
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
        effective_edge: torch.Tensor,
        interventions: torch.Tensor,
        context: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        senders = torch.matmul(rel_send, hidden)
        receivers = torch.matmul(rel_rec, hidden)
        messages = self.message(torch.cat([senders, receivers], dim=-1))
        messages = messages * effective_edge.unsqueeze(-1)
        incoming = torch.matmul(rel_rec.transpose(0, 1), messages)
        intervention_hidden = self.intervention_embedding(interventions)
        update_input = torch.cat(
            [states, incoming, intervention_hidden, context], dim=-1
        )
        next_hidden = self.update(
            update_input.reshape(-1, update_input.shape[-1]),
            hidden.reshape(-1, hidden.shape[-1]),
        ).reshape_as(hidden)
        prediction = states + self.readout(next_hidden)
        return prediction, next_hidden, incoming

    def rollout(
        self,
        initial_state: torch.Tensor,
        initial_hidden: torch.Tensor,
        effective_edge: torch.Tensor,
        interventions: torch.Tensor,
        contexts: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        current = initial_state
        hidden = initial_hidden
        predictions = []
        incoming_messages = []
        for step_index in range(interventions.shape[2]):
            current, hidden, incoming = self.step(
                current,
                hidden,
                effective_edge,
                interventions[:, :, step_index],
                contexts[:, :, step_index],
                rel_rec,
                rel_send,
            )
            predictions.append(current)
            incoming_messages.append(incoming)
        return {
            "prediction": torch.stack(predictions, dim=2),
            "incoming_message": torch.stack(incoming_messages, dim=2),
            "final_hidden": hidden,
        }


class InterventionalSRIModel(nn.Module):
    """Binary-strength SRI backbone plus paired dynamic-graph rollouts."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int = 0,
        intervention_dim: int = 4,
        encoder_hidden: int = 128,
        attention_heads: int = 4,
        decoder_hidden: int = 128,
        message_hidden: int = 128,
        dynamic_graph_hidden: int = 128,
        dynamic_graph_delta_scale: float = 0.1,
        dropout: float = 0.1,
        temperature: float = 0.5,
        hard_gumbel: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.intervention_dim = int(intervention_dim)
        self.temperature = float(temperature)
        self.hard_gumbel = bool(hard_gumbel)
        # During the first epochs the decoder consumes E[Z] * S.  Training can
        # then progressively blend towards the straight-through discrete graph.
        self.decoder_graph_blend = 0.0
        self.hard_edge_threshold = 0.5
        self.encoder = InterventionalSRIEncoder(
            state_dim,
            context_dim,
            intervention_dim,
            encoder_hidden,
            attention_heads,
            dropout,
        )
        self.graph_prior = DynamicGraphPrior(
            state_dim,
            encoder_hidden,
            dynamic_graph_hidden,
            dynamic_graph_delta_scale,
        )
        # The decoder intentionally contains no dropout: two calls with U=0
        # must be identical, including during training.
        self.decoder = InterventionalMessageDecoder(
            state_dim,
            context_dim,
            intervention_dim,
            decoder_hidden,
            message_hidden,
        )

    def set_graph_schedule(self, *, discrete_blend: float, temperature: float) -> None:
        """Set the soft-to-discrete curriculum used by subsequent forwards."""

        if not 0.0 <= discrete_blend <= 1.0:
            raise ValueError("discrete_blend must lie in [0, 1].")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.decoder_graph_blend = float(discrete_blend)
        self.temperature = float(temperature)

    def set_edge_decision_threshold(self, threshold: float) -> None:
        """Set the validation-calibrated threshold used by ``hard`` decoding."""

        if not 0.0 < threshold < 1.0:
            raise ValueError("edge decision threshold must lie in (0, 1).")
        self.hard_edge_threshold = float(threshold)

    def _decoder_edge(
        self, graph: dict[str, torch.Tensor], mode: str
    ) -> torch.Tensor:
        """Select the edge tensor consumed by the message decoder.

        ``soft`` is the posterior expectation ``P(edge) * strength``. ``hard``
        uses the validation-calibrated binary decision. ``scheduled`` blends
        soft edges with straight-through Gumbel samples, while
        ``zero``/``shuffled`` are diagnostic ablations.
        """

        soft = graph["effective_edge_mean"]
        # Explicit hard evaluation uses the validation-calibrated threshold.
        # The scheduled training path below still uses the straight-through
        # Gumbel sample so gradients reach the existence logits.
        hard = (
            graph["existence_probability"][..., 1]
            >= self.hard_edge_threshold
        ).to(soft.dtype) * graph["strength"]
        straight_through = graph["effective_edge_sample"]
        if mode == "soft":
            return soft
        if mode == "hard":
            return hard
        if mode == "scheduled":
            return torch.lerp(soft, straight_through, self.decoder_graph_blend)
        if mode == "zero":
            return torch.zeros_like(soft)
        if mode == "shuffled":
            return torch.roll(soft, shifts=1, dims=-1)
        raise ValueError(
            "decoder_graph_mode must be scheduled, soft, hard, zero or shuffled."
        )

    @staticmethod
    def _gumbel_noise(reference: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        uniform = torch.rand(shape, dtype=reference.dtype, device=reference.device)
        return -torch.log(-torch.log(uniform.clamp(min=1e-7, max=1.0 - 1e-7)))

    def _dynamic_rollout(
        self,
        initial_state: torch.Tensor,
        initial_hidden: torch.Tensor,
        initial_graph: dict[str, torch.Tensor],
        initial_graph_hidden: torch.Tensor,
        interventions: torch.Tensor,
        contexts: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        shared_gumbel_noise: torch.Tensor | None,
        decoder_graph_mode: str,
    ) -> dict[str, torch.Tensor]:
        current = initial_state
        decoder_hidden = initial_hidden
        graph_hidden = initial_graph_hidden
        graph = initial_graph
        predictions = []
        incoming_messages = []
        graph_probabilities = []
        graph_samples = []
        graph_strengths = []
        graph_effective_samples = []
        graph_effective_means = []
        graph_decoder_edges = []
        horizon = interventions.shape[2]
        for step_index in range(horizon):
            graph_probabilities.append(graph["existence_probability"])
            graph_samples.append(graph["existence_sample"])
            graph_strengths.append(graph["strength"])
            graph_effective_samples.append(graph["effective_edge_sample"])
            graph_effective_means.append(graph["effective_edge_mean"])
            decoder_edge = self._decoder_edge(graph, decoder_graph_mode)
            graph_decoder_edges.append(decoder_edge)
            current, decoder_hidden, incoming = self.decoder.step(
                current,
                decoder_hidden,
                decoder_edge,
                interventions[:, :, step_index],
                contexts[:, :, step_index],
                rel_rec,
                rel_send,
            )
            predictions.append(current)
            incoming_messages.append(incoming)
            if step_index + 1 < horizon:
                noise = (
                    None
                    if shared_gumbel_noise is None
                    else shared_gumbel_noise[:, step_index]
                )
                graph = self.graph_prior.step(
                    current,
                    graph["existence_probability"],
                    graph["strength"],
                    graph_hidden,
                    rel_rec,
                    rel_send,
                    gumbel_noise=noise,
                    temperature=self.temperature,
                    hard=self.hard_gumbel,
                    stochastic=self.training,
                )
                graph_hidden = graph["hidden"]
        return {
            "prediction": torch.stack(predictions, dim=2),
            "incoming_message": torch.stack(incoming_messages, dim=2),
            "final_hidden": decoder_hidden,
            "graph_probability": torch.stack(graph_probabilities, dim=1),
            "graph_sample": torch.stack(graph_samples, dim=1),
            "graph_strength": torch.stack(graph_strengths, dim=1),
            "graph_effective_sample": torch.stack(graph_effective_samples, dim=1),
            "graph_effective_mean": torch.stack(graph_effective_means, dim=1),
            "graph_decoder_edge": torch.stack(graph_decoder_edges, dim=1),
        }

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
    ) -> dict[str, torch.Tensor]:
        """Encode once, then decode nominal and intervened futures.

        ``intervention_future`` has shape ``[B, N, H, U]``.  The current
        intervention never enters the encoder.  If ``baseline_interventions``
        is omitted, the control branch receives an all-zero intervention.
        """

        if intervention_future.ndim != 4:
            raise ValueError(
                "intervention_future must have shape [batch, nodes, horizon, features]."
            )
        batch, nodes, horizon, intervention_features = intervention_future.shape
        if intervention_features != self.intervention_dim:
            raise ValueError("intervention_future has the wrong feature dimension.")
        if tuple(history_states.shape[:2]) != (batch, nodes):
            raise ValueError("History and rollout batch/node dimensions differ.")
        if future_context is None:
            future_context = history_states.new_zeros(
                batch, nodes, horizon, self.context_dim
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
            hard=self.hard_gumbel,
            stochastic=self.training,
        )
        if history_context is None:
            last_context = history_states.new_zeros(batch, nodes, self.context_dim)
        else:
            last_context = history_context[:, :, -1]
        initial_state = history_states[:, :, -1]
        initial_hidden = self.decoder.initialize_hidden(initial_state, last_context)
        initial_graph_hidden = self.graph_prior.initialize_hidden(
            encoded["edge_embedding"]
        )
        initial_graph = {
            key: encoded[key]
            for key in (
                "existence_logits",
                "existence_probability",
                "existence_sample",
                "strength",
                "effective_edge_sample",
                "effective_edge_mean",
            )
        }
        shared_gumbel_noise = None
        if self.training and horizon > 1:
            edges = rel_rec.shape[0]
            shared_gumbel_noise = self._gumbel_noise(
                history_states,
                (batch, horizon - 1, edges, 2),
            )
        baseline = self._dynamic_rollout(
            initial_state,
            initial_hidden,
            initial_graph,
            initial_graph_hidden,
            baseline_interventions,
            future_context,
            rel_rec,
            rel_send,
            shared_gumbel_noise,
            decoder_graph_mode,
        )
        intervention = self._dynamic_rollout(
            initial_state,
            initial_hidden,
            initial_graph,
            initial_graph_hidden,
            intervention_future,
            future_context,
            rel_rec,
            rel_send,
            shared_gumbel_noise,
            decoder_graph_mode,
        )
        result = {
            **encoded,
            "baseline_prediction": baseline["prediction"],
            "intervention_prediction": intervention["prediction"],
            "effect_prediction": (
                intervention["prediction"] - baseline["prediction"]
            ),
            "baseline_incoming_message": baseline["incoming_message"],
            "intervention_incoming_message": intervention["incoming_message"],
            "baseline_graph_probability": baseline["graph_probability"],
            "intervention_graph_probability": intervention["graph_probability"],
            "baseline_graph_sample": baseline["graph_sample"],
            "intervention_graph_sample": intervention["graph_sample"],
            "baseline_graph_strength": baseline["graph_strength"],
            "intervention_graph_strength": intervention["graph_strength"],
            "baseline_graph_effective_sample": baseline["graph_effective_sample"],
            "intervention_graph_effective_sample": intervention["graph_effective_sample"],
            "baseline_graph_effective_mean": baseline["graph_effective_mean"],
            "intervention_graph_effective_mean": intervention["graph_effective_mean"],
            "baseline_graph_decoder_edge": baseline["graph_decoder_edge"],
            "intervention_graph_decoder_edge": intervention["graph_decoder_edge"],
            "decoder_graph_mode": decoder_graph_mode,
            "decoder_graph_blend": history_states.new_tensor(
                self.decoder_graph_blend
            ),
            "hard_edge_threshold": history_states.new_tensor(
                self.hard_edge_threshold
            ),
            "intervention_target_node": intervention_future.abs().sum(
                dim=(2, 3)
            )
            > 1e-8,
        }
        if include_graph_ablations:
            zero_baseline = self._dynamic_rollout(
                initial_state,
                initial_hidden,
                initial_graph,
                initial_graph_hidden,
                baseline_interventions,
                future_context,
                rel_rec,
                rel_send,
                shared_gumbel_noise,
                "zero",
            )
            zero_intervention = self._dynamic_rollout(
                initial_state,
                initial_hidden,
                initial_graph,
                initial_graph_hidden,
                intervention_future,
                future_context,
                rel_rec,
                rel_send,
                shared_gumbel_noise,
                "zero",
            )
            result.update(
                {
                    "no_graph_baseline_prediction": zero_baseline["prediction"],
                    "no_graph_intervention_prediction": zero_intervention[
                        "prediction"
                    ],
                    "no_graph_effect_prediction": (
                        zero_intervention["prediction"]
                        - zero_baseline["prediction"]
                    ),
                }
            )
        return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum()
    if float(denominator.detach()) == 0.0:
        return values.new_zeros(())
    return (values * expanded).sum() / denominator


@dataclass(frozen=True)
class InterventionalLossWeights:
    """Weights of the predictive, causal, and graph objective terms."""

    lambda_intervention: float = 1.0
    lambda_effect: float = 2.0
    beta_kl: float = 0.1
    # Kept as an explicit opt-in diagnostic. It must stay zero for the main
    # model because penalizing P(edge) * S directly collapses the intensity.
    lambda_strength: float = 0.0
    lambda_graph_smoothness: float = 1e-3
    lambda_graph_necessity: float = 1.0
    graph_necessity_margin: float = 0.05
    graph_necessity_min_effect: float = 1e-2


def interventional_sri_loss(
    output: dict[str, torch.Tensor],
    baseline_target: torch.Tensor,
    intervention_target: torch.Tensor,
    paired_mask: torch.Tensor,
    state_sigma: float,
    edge_prior_probability: float,
    weights: InterventionalLossWeights,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss for mixed nominal and physically paired fork samples.

    Every sample contributes to the nominal trajectory loss.  Only samples
    whose ``paired_mask`` is true contribute to the intervention and causal
    effect losses.  The causal target is the paired physical difference
    ``X_intervention - X_baseline``.
    """

    baseline_error = (output["baseline_prediction"] - baseline_target).square()
    intervention_error = (
        output["intervention_prediction"] - intervention_target
    ).square()
    true_effect = intervention_target - baseline_target
    effect_error = (output["effect_prediction"] - true_effect).square()
    if horizon_weights is not None:
        if horizon_weights.ndim != 1 or horizon_weights.shape[0] != baseline_target.shape[2]:
            raise ValueError("horizon_weights must contain one value per rollout step.")
        scale = horizon_weights.to(baseline_target).view(1, 1, -1, 1)
        baseline_error = baseline_error * scale
        intervention_error = intervention_error * scale
        effect_error = effect_error * scale

    baseline_nll = baseline_error.mean() / (2.0 * state_sigma**2)
    intervention_nll = _masked_mean(
        intervention_error, paired_mask
    ) / (2.0 * state_sigma**2)
    effect_mse = _masked_mean(effect_error, paired_mask)

    # An intervention is injected only into its target node.  Any predicted
    # effect on another responsive node must therefore travel through graph
    # messages.  The hinge asks the full graph to improve on the exact no-graph
    # effect (zero on non-target nodes) by a small margin.
    target_nodes = output.get("intervention_target_node")
    if target_nodes is None:
        graph_necessity = effect_error.new_zeros(())
        graph_necessity_full_error = effect_error.new_zeros(())
        graph_necessity_no_graph_error = effect_error.new_zeros(())
        graph_necessity_nodes = effect_error.new_zeros(())
    else:
        responsive = (
            true_effect.square().mean(dim=(2, 3)).sqrt()
            >= weights.graph_necessity_min_effect
        )
        propagation_mask = (
            paired_mask[:, None] & ~target_nodes.bool() & responsive
        )
        graph_necessity_nodes = propagation_mask.float().sum()
        if bool(propagation_mask.any()):
            graph_necessity_full_error = _masked_mean(
                effect_error, propagation_mask
            )
            no_graph_error = true_effect.square()
            if horizon_weights is not None:
                no_graph_error = no_graph_error * scale
            graph_necessity_no_graph_error = _masked_mean(
                no_graph_error, propagation_mask
            )
            graph_necessity = F.relu(
                effect_error.new_tensor(weights.graph_necessity_margin)
                + graph_necessity_full_error
                / graph_necessity_no_graph_error.clamp_min(1e-8)
                - 1.0
            )
        else:
            graph_necessity = effect_error.new_zeros(())
            graph_necessity_full_error = effect_error.new_zeros(())
            graph_necessity_no_graph_error = effect_error.new_zeros(())

    q = output["existence_probability"].clamp_min(1e-9)
    prior = q.new_tensor(
        [1.0 - edge_prior_probability, edge_prior_probability]
    ).clamp_min(1e-9)
    existence_kl = (q * (q.log() - prior.log())).sum(dim=-1).mean()
    baseline_effective = output["baseline_graph_effective_mean"]
    intervention_effective = output["intervention_graph_effective_mean"]
    if bool(paired_mask.any()):
        effective_strength = 0.5 * (
            baseline_effective.mean()
            + _masked_mean(intervention_effective, paired_mask)
        )
    else:
        effective_strength = baseline_effective.mean()
    if baseline_effective.shape[1] > 1:
        baseline_smoothness = (
            baseline_effective[:, 1:] - baseline_effective[:, :-1]
        ).abs().mean()
        intervention_smoothness = _masked_mean(
            (
                intervention_effective[:, 1:]
                - intervention_effective[:, :-1]
            ).abs(),
            paired_mask,
        )
        graph_smoothness = (
            0.5 * (baseline_smoothness + intervention_smoothness)
            if bool(paired_mask.any())
            else baseline_smoothness
        )
    else:
        graph_smoothness = baseline_effective.new_zeros(())
    loss = (
        baseline_nll
        + weights.lambda_intervention * intervention_nll
        + weights.lambda_effect * effect_mse
        + weights.beta_kl * existence_kl
        + weights.lambda_strength * effective_strength
        + weights.lambda_graph_smoothness * graph_smoothness
        + weights.lambda_graph_necessity * graph_necessity
    )
    return loss, {
        "baseline_nll": baseline_nll,
        "intervention_nll": intervention_nll,
        "effect_mse": effect_mse,
        "existence_kl": existence_kl,
        "effective_strength_l1": effective_strength,
        "graph_smoothness": graph_smoothness,
        "graph_necessity": graph_necessity,
        "graph_necessity_full_error": graph_necessity_full_error,
        "graph_necessity_no_graph_error": graph_necessity_no_graph_error,
        "graph_necessity_nodes": graph_necessity_nodes,
        "paired_fraction": paired_mask.float().mean(),
    }


__all__ = [
    "InterventionalLossWeights",
    "InterventionalMessageDecoder",
    "InterventionalSRIEncoder",
    "InterventionalSRIModel",
    "DynamicGraphPrior",
    "interventional_sri_loss",
]
