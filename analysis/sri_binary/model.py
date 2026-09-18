"""SRI-inspired dynamic graph model with binary no-edge/edge relations.

The model retains the main SRI ideas that are useful for this simulator:

* a dynamic relation is inferred at every transition;
* edge features are coupled through multi-head attention;
* forward/reverse LSTMs represent temporal prior/posterior relations;
* a scalar relation strength is inferred separately from the binary type;
* type 0 contributes exactly no inter-drone message.

It intentionally does not reproduce SRI's repulsion/alignment ontology.  The
only categorical meanings are 0=no-edge and 1=edge.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from analysis.nri_dynamic_categorical.model import (
    RelationGNN,
    RecurrentCategoricalMessageDecoder,
)


class SRIBinaryEncoder(nn.Module):
    """Dynamic binary prior/posterior and conditional relation strength."""

    def __init__(
        self,
        state_dim: int,
        hidden: int,
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.gnn = RelationGNN(state_dim, hidden, dropout)
        self.edge_attention = nn.MultiheadAttention(
            hidden,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden)
        self.prior_lstm = nn.LSTMCell(hidden, hidden)
        self.posterior_reverse_lstm = nn.LSTMCell(hidden, hidden)
        self.prior_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2)
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Linear(hidden, 2)
        )
        self.prior_strength_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )
        self.posterior_strength_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )

    @staticmethod
    def _initial_lstm_state(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(values)
        return zeros, zeros.clone()

    def _attend_incoming_edges(
        self,
        embeddings: torch.Tensor,
        rel_rec: torch.Tensor,
    ) -> torch.Tensor:
        """Attend only among candidate senders of the same receiver."""

        receiver_ids = rel_rec.argmax(dim=1)
        attention_mask = receiver_ids[:, None] != receiver_ids[None, :]
        attended, _ = self.edge_attention(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.attention_norm(embeddings + attended)

    def _one_step_embedding(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> torch.Tensor:
        return self._attend_incoming_edges(
            self.gnn(states, rel_rec, rel_send), rel_rec
        )

    def prior_step(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        temporal_state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedding = self._one_step_embedding(states, rel_rec, rel_send)
        flat = embedding.reshape(-1, self.hidden)
        if temporal_state is None:
            temporal_state = self._initial_lstm_state(flat)
        temporal_state = self.prior_lstm(flat, temporal_state)
        hidden = temporal_state[0]
        logits = self.prior_head(hidden).reshape(states.shape[0], -1, 2)
        strength = torch.sigmoid(self.prior_strength_head(hidden)).reshape(
            states.shape[0], -1
        )
        return logits, strength, temporal_state

    def forward(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        temperature: float,
        hard: bool,
        stochastic: bool,
    ) -> dict[str, torch.Tensor]:
        embeddings = torch.stack(
            [
                self._one_step_embedding(states[:, :, t], rel_rec, rel_send)
                for t in range(states.shape[2] - 1)
            ],
            dim=1,
        )
        batch, transitions, edges, _ = embeddings.shape

        prior_temporal = None
        prior_hidden = []
        prior_logits = []
        prior_strength = []
        for t in range(transitions):
            flat = embeddings[:, t].reshape(-1, self.hidden)
            if prior_temporal is None:
                prior_temporal = self._initial_lstm_state(flat)
            prior_temporal = self.prior_lstm(flat, prior_temporal)
            hidden = prior_temporal[0]
            prior_hidden.append(hidden.reshape(batch, edges, self.hidden))
            prior_logits.append(self.prior_head(hidden).reshape(batch, edges, 2))
            prior_strength.append(
                torch.sigmoid(self.prior_strength_head(hidden)).reshape(batch, edges)
            )

        reverse_temporal = None
        reverse_hidden: list[torch.Tensor | None] = [None] * transitions
        for t in range(transitions - 1, -1, -1):
            flat = embeddings[:, t].reshape(-1, self.hidden)
            if reverse_temporal is None:
                reverse_temporal = self._initial_lstm_state(flat)
            reverse_temporal = self.posterior_reverse_lstm(flat, reverse_temporal)
            reverse_hidden[t] = reverse_temporal[0].reshape(
                batch, edges, self.hidden
            )

        coupled = [
            torch.cat([prior_hidden[t], reverse_hidden[t]], dim=-1)
            for t in range(transitions)
        ]
        prior_logits_tensor = torch.stack(prior_logits, dim=1)
        posterior_logits = torch.stack(
            [self.posterior_head(value) for value in coupled], dim=1
        )
        prior_probabilities = F.softmax(prior_logits_tensor, dim=-1)
        posterior_probabilities = F.softmax(posterior_logits, dim=-1)
        posterior_strength = torch.stack(
            [
                torch.sigmoid(self.posterior_strength_head(value)).squeeze(-1)
                for value in coupled
            ],
            dim=1,
        )
        prior_strength_tensor = torch.stack(prior_strength, dim=1)
        if stochastic:
            edge_sample = F.gumbel_softmax(
                posterior_logits, tau=temperature, hard=hard, dim=-1
            )
        else:
            edge_sample = F.one_hot(
                posterior_probabilities.argmax(dim=-1), num_classes=2
            ).to(posterior_probabilities.dtype)

        effective_sample = edge_sample[..., 1] * posterior_strength
        soft_effective = posterior_probabilities[..., 1] * posterior_strength
        return {
            "prior_logits": prior_logits_tensor,
            "posterior_logits": posterior_logits,
            "prior_probabilities": prior_probabilities,
            "posterior_probabilities": posterior_probabilities,
            "prior_strength": prior_strength_tensor,
            "posterior_strength": posterior_strength,
            "edge_sample": edge_sample,
            "effective_edge_sample": effective_sample,
            "soft_effective_edge": soft_effective,
        }


class SRIBinaryModel(nn.Module):
    """SRI temporal/strength architecture with binary edge semantics."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        encoder_hidden: int,
        attention_heads: int,
        message_hidden: int,
        decoder_hidden: int,
        context_hidden: int,
        dropout: float,
        temperature: float,
        hard_gumbel: bool,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.temperature = float(temperature)
        self.hard_gumbel = bool(hard_gumbel)
        self.encoder = SRIBinaryEncoder(
            state_dim, encoder_hidden, attention_heads, dropout
        )
        self.decoder = RecurrentCategoricalMessageDecoder(
            state_dim,
            2,
            message_hidden,
            decoder_hidden,
            context_dim,
            context_hidden,
            dropout,
        )

    def _contexts(
        self, states: torch.Tensor, contexts: torch.Tensor | None
    ) -> torch.Tensor:
        if contexts is None:
            if self.context_dim:
                raise ValueError("This model requires context tensors.")
            return states.new_empty(*states.shape[:3], 0)
        return contexts

    @staticmethod
    def _weighted_binary_sample(
        edge_sample: torch.Tensor, strength: torch.Tensor
    ) -> torch.Tensor:
        # Decoder type 0 is skipped exactly. Type 1 is modulated by strength.
        return torch.stack(
            [edge_sample[..., 0], edge_sample[..., 1] * strength], dim=-1
        )

    def forward(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        contexts: torch.Tensor | None = None,
        reconstruction_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        contexts = self._contexts(states, contexts)
        transitions = states.shape[2] - 1
        if not 1 <= reconstruction_steps <= transitions:
            raise ValueError("reconstruction_steps is outside the sequence.")
        encoded = self.encoder(
            states,
            rel_rec,
            rel_send,
            temperature=self.temperature,
            hard=self.hard_gumbel,
            stochastic=self.training,
        )
        weighted = self._weighted_binary_sample(
            encoded["edge_sample"], encoded["posterior_strength"]
        )
        hidden = self.decoder.initialize_hidden(states[:, :, 0], contexts[:, :, 0])
        predictions = []
        context_predictions = []
        autoregressive_start = transitions - reconstruction_steps
        current_state = states[:, :, 0]
        current_context = contexts[:, :, 0]
        for t in range(transitions):
            if t <= autoregressive_start:
                decoder_state = states[:, :, t]
                decoder_context = contexts[:, :, t]
            else:
                decoder_state = current_state
                decoder_context = current_context
            current_state, current_context, hidden = self.decoder(
                decoder_state,
                weighted[:, t],
                rel_rec,
                rel_send,
                decoder_context,
                hidden,
            )
            predictions.append(current_state)
            context_predictions.append(current_context)
        return {
            **encoded,
            "prediction": torch.stack(predictions, dim=2),
            "context_prediction": torch.stack(context_predictions, dim=2),
        }

    def forecast_after_context(
        self,
        states: torch.Tensor,
        contexts: torch.Tensor | None,
        observed_steps: int,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contexts = self._contexts(states, contexts)
        temporal_state = None
        hidden = self.decoder.initialize_hidden(states[:, :, 0], contexts[:, :, 0])
        for t in range(observed_steps - 1):
            logits, strength, temporal_state = self.encoder.prior_step(
                states[:, :, t], rel_rec, rel_send, temporal_state
            )
            probability = F.softmax(logits, dim=-1)
            sample = F.one_hot(probability.argmax(-1), num_classes=2).to(
                probability.dtype
            )
            weighted = self._weighted_binary_sample(sample, strength)
            _, _, hidden = self.decoder(
                states[:, :, t],
                weighted,
                rel_rec,
                rel_send,
                contexts[:, :, t],
                hidden,
            )

        current = states[:, :, observed_steps - 1]
        current_context = contexts[:, :, observed_steps - 1]
        predictions = []
        context_predictions = []
        probabilities_out = []
        strengths_out = []
        samples_out = []
        for _ in range(observed_steps - 1, states.shape[2] - 1):
            logits, strength, temporal_state = self.encoder.prior_step(
                current, rel_rec, rel_send, temporal_state
            )
            probability = F.softmax(logits, dim=-1)
            sample = F.one_hot(probability.argmax(-1), num_classes=2).to(
                probability.dtype
            )
            weighted = self._weighted_binary_sample(sample, strength)
            current, current_context, hidden = self.decoder(
                current,
                weighted,
                rel_rec,
                rel_send,
                current_context,
                hidden,
            )
            predictions.append(current)
            context_predictions.append(current_context)
            probabilities_out.append(probability)
            strengths_out.append(strength)
            samples_out.append(sample)
        return {
            "prediction": torch.stack(predictions, dim=2),
            "context_prediction": torch.stack(context_predictions, dim=2),
            "prior_probabilities": torch.stack(probabilities_out, dim=1),
            "prior_strength": torch.stack(strengths_out, dim=1),
            "edge_sample": torch.stack(samples_out, dim=1),
        }


def sri_binary_loss(
    prediction: torch.Tensor,
    state_target: torch.Tensor,
    context_prediction: torch.Tensor,
    context_target: torch.Tensor,
    posterior_probabilities: torch.Tensor,
    prior_probabilities: torch.Tensor,
    posterior_strength: torch.Tensor,
    edge_prior: tuple[float, float],
    state_sigma: float,
    beta_kl: float,
    beta_sparse_prior: float,
    lambda_graph_smooth: float,
    lambda_strength_l1: float,
    lambda_context: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unsupervised trajectory ELBO plus explicit graph regularization."""

    reconstruction_nll = F.mse_loss(prediction, state_target) / (
        2.0 * state_sigma**2
    )
    context_mse = (
        F.mse_loss(context_prediction, context_target)
        if context_prediction.numel()
        else prediction.new_zeros(())
    )
    q = posterior_probabilities.clamp_min(1e-9)
    p = prior_probabilities.clamp_min(1e-9)
    temporal_kl = (q * (q.log() - p.log())).sum(-1).mean()
    fixed_prior = torch.as_tensor(
        edge_prior, dtype=q.dtype, device=q.device
    ).clamp_min(1e-9)
    sparse_prior_kl = (q * (q.log() - fixed_prior.log())).sum(-1).mean()
    effective = q[..., 1] * posterior_strength
    graph_smooth = (
        (effective[:, 1:] - effective[:, :-1]).square().mean()
        if effective.shape[1] > 1
        else effective.new_zeros(())
    )
    strength_l1 = effective.mean()
    negative_elbo = reconstruction_nll + beta_kl * temporal_kl
    loss = (
        negative_elbo
        + beta_sparse_prior * sparse_prior_kl
        + lambda_graph_smooth * graph_smooth
        + lambda_strength_l1 * strength_l1
        + lambda_context * context_mse
    )
    return loss, {
        "negative_elbo": negative_elbo,
        "reconstruction_nll": reconstruction_nll,
        "temporal_kl": temporal_kl,
        "sparse_prior_kl": sparse_prior_kl,
        "graph_smooth": graph_smooth,
        "strength_l1": strength_l1,
        "context_mse": context_mse,
    }


__all__ = ["SRIBinaryEncoder", "SRIBinaryModel", "sri_binary_loss"]
