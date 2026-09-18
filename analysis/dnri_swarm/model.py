"""Paper-faithful dynamic Neural Relational Inference modules.

For every transition t -> t+1, the encoder produces

* a causal learned prior p(z^t | x^{1:t}) from a forward LSTM, and
* a smoothing posterior q(z^t | x^{1:T}) from forward/backward LSTMs.

The recurrent decoder uses the sampled time-varying edge type in its message
passing operation and predicts a residual state update. Tensor conventions in
this package are [batch, nodes, time, features] and [batch, edges, time, types].
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from analysis.nri_original_swarm.model import MLP


GraphSource = Literal["posterior", "prior"]
LSTMState = tuple[torch.Tensor, torch.Tensor]


def _classification_head(
    n_in: int,
    n_hidden: int,
    n_out: int,
    layers: int,
) -> nn.Module:
    """MLP head used for both q and the learned prior in dNRI experiments."""
    if layers == 1:
        return nn.Linear(n_in, n_out)
    modules: list[nn.Module] = [nn.Linear(n_in, n_hidden), nn.ELU()]
    for _ in range(layers - 2):
        modules.extend((nn.Linear(n_hidden, n_hidden), nn.ELU()))
    modules.append(nn.Linear(n_hidden, n_out))
    return nn.Sequential(*modules)


def _node_to_edge(
    nodes: torch.Tensor,
    rel_rec: torch.Tensor,
    rel_send: torch.Tensor,
) -> torch.Tensor:
    """Map [B,N,T,H] node features to [B,E,T,2H] directed pairs."""
    receivers = torch.einsum("en,bnth->beth", rel_rec, nodes)
    senders = torch.einsum("en,bnth->beth", rel_send, nodes)
    return torch.cat([senders, receivers], dim=-1)


def _edge_to_node(edges: torch.Tensor, rel_rec: torch.Tensor) -> torch.Tensor:
    """Aggregate incoming [B,E,T,H] messages into [B,N,T,H]."""
    incoming = torch.einsum("en,beth->bnth", rel_rec, edges)
    return incoming / max(rel_rec.shape[1] - 1, 1)


class DynamicRelationalEncoder(nn.Module):
    """Spatial factor GNN plus temporal forward/backward LSTMs."""

    def __init__(
        self,
        node_dims: int,
        hidden: int,
        rnn_hidden: int,
        edge_types: int,
        dropout: float = 0.0,
        rnn_layers: int = 1,
        encoder_head_hidden: int = 128,
        encoder_head_layers: int = 3,
        prior_head_hidden: int = 128,
        prior_head_layers: int = 3,
    ) -> None:
        super().__init__()
        self.edge_types = int(edge_types)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = int(rnn_layers)

        self.node_mlp = MLP(node_dims, hidden, hidden, dropout)
        self.edge_mlp = MLP(2 * hidden, hidden, hidden, dropout)
        self.node_factor_mlp = MLP(hidden, hidden, hidden, dropout)
        self.edge_factor_mlp = MLP(3 * hidden, hidden, hidden, dropout)

        recurrent_dropout = dropout if rnn_layers > 1 else 0.0
        self.forward_rnn = nn.LSTM(
            hidden,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.backward_rnn = nn.LSTM(
            hidden,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.prior_head = _classification_head(
            rnn_hidden, prior_head_hidden, edge_types, prior_head_layers
        )
        self.posterior_head = _classification_head(
            2 * rnn_hidden, encoder_head_hidden, edge_types, encoder_head_layers
        )
        self._reset_output_heads()

    def _reset_output_heads(self) -> None:
        for module in (self.prior_head, self.posterior_head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0.1)

    def spatial_embedding(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> torch.Tensor:
        """Build one factor-graph embedding for every edge and time step."""
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, nodes, time, features].")
        nodes = self.node_mlp(states)
        edges = self.edge_mlp(_node_to_edge(nodes, rel_rec, rel_send))
        skip = edges
        nodes = self.node_factor_mlp(_edge_to_node(edges, rel_rec))
        factor_edges = _node_to_edge(nodes, rel_rec, rel_send)
        return self.edge_factor_mlp(torch.cat([factor_edges, skip], dim=-1))

    def forward(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> dict[str, torch.Tensor | LSTMState]:
        """Infer prior/posterior logits for all supplied transition inputs."""
        embedded = self.spatial_embedding(states, rel_rec, rel_send)
        batch, edges, timesteps, hidden = embedded.shape
        sequence = embedded.reshape(batch * edges, timesteps, hidden)

        forward_hidden, prior_state = self.forward_rnn(sequence)
        reverse_input = torch.flip(sequence, dims=(1,))
        backward_hidden, _ = self.backward_rnn(reverse_input)
        backward_hidden = torch.flip(backward_hidden, dims=(1,))

        prior_logits = self.prior_head(forward_hidden)
        posterior_logits = self.posterior_head(
            torch.cat([forward_hidden, backward_hidden], dim=-1)
        )
        return {
            "prior_logits": prior_logits.reshape(
                batch, edges, timesteps, self.edge_types
            ),
            "posterior_logits": posterior_logits.reshape(
                batch, edges, timesteps, self.edge_types
            ),
            "prior_state": prior_state,
        }

    def prior_step(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        prior_state: LSTMState | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        """Advance the causal prior by one state during an open-loop rollout."""
        if states.ndim != 3:
            raise ValueError("states must have shape [batch, nodes, features].")
        embedded = self.spatial_embedding(
            states.unsqueeze(2), rel_rec, rel_send
        ).squeeze(2)
        batch, edges, hidden = embedded.shape
        sequence = embedded.reshape(batch * edges, 1, hidden)
        forward_hidden, new_state = self.forward_rnn(sequence, prior_state)
        logits = self.prior_head(forward_hidden[:, 0]).reshape(
            batch, edges, self.edge_types
        )
        return logits, new_state


class DynamicRNNDecoder(nn.Module):
    """Recurrent NRI decoder receiving a potentially new graph at every step."""

    def __init__(
        self,
        node_dims: int,
        edge_types: int,
        hidden: int,
        dropout: float = 0.0,
        skip_first_edge_type: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dims = int(hidden)
        self.edge_types = int(edge_types)
        self.dropout = float(dropout)
        self.skip_first = bool(skip_first_edge_type)

        self.message_fc1 = nn.ModuleList(
            nn.Linear(2 * hidden, hidden) for _ in range(edge_types)
        )
        self.message_fc2 = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(edge_types)
        )

        self.input_reset = nn.Linear(node_dims, hidden)
        self.input_update = nn.Linear(node_dims, hidden)
        self.input_candidate = nn.Linear(node_dims, hidden)
        self.message_reset = nn.Linear(hidden, hidden, bias=False)
        self.message_update = nn.Linear(hidden, hidden, bias=False)
        self.message_candidate = nn.Linear(hidden, hidden, bias=False)

        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, hidden)
        self.out_fc3 = nn.Linear(hidden, node_dims)

    def initial_hidden(self, states: torch.Tensor) -> torch.Tensor:
        return states.new_zeros(states.shape[0], states.shape[1], self.hidden_dims)

    def single_step(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor,
        edge_sample: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform dynamic message passing and one residual state update."""
        receivers = torch.einsum("en,bnh->beh", rel_rec, hidden)
        senders = torch.einsum("en,bnh->beh", rel_send, hidden)
        # Match the recurrent interaction decoder in the reference dNRI code.
        pair_hidden = torch.cat([receivers, senders], dim=-1)
        messages = hidden.new_zeros(hidden.shape[0], rel_rec.shape[0], self.hidden_dims)

        first_type = 1 if self.skip_first else 0
        active_types = max(self.edge_types - first_type, 1)
        for edge_type in range(first_type, self.edge_types):
            message = torch.tanh(self.message_fc1[edge_type](pair_hidden))
            message = F.dropout(message, self.dropout, training=self.training)
            message = torch.tanh(self.message_fc2[edge_type](message))
            messages = messages + (
                message * edge_sample[..., edge_type : edge_type + 1] / active_types
            )

        aggregated = torch.einsum("en,beh->bnh", rel_rec, messages)
        aggregated = aggregated / max(states.shape[1] - 1, 1)

        reset = torch.sigmoid(
            self.input_reset(states) + self.message_reset(aggregated)
        )
        update = torch.sigmoid(
            self.input_update(states) + self.message_update(aggregated)
        )
        candidate = torch.tanh(
            self.input_candidate(states)
            + reset * self.message_candidate(aggregated)
        )
        hidden = (1.0 - update) * candidate + update * hidden

        output = F.dropout(F.relu(self.out_fc1(hidden)), self.dropout, self.training)
        output = F.dropout(F.relu(self.out_fc2(output)), self.dropout, self.training)
        delta = self.out_fc3(output)
        return states + delta, hidden


class DNRI(nn.Module):
    """Complete dNRI ELBO model with reconstruction and causal forecasting APIs."""

    def __init__(
        self,
        node_dims: int,
        edge_types: int = 2,
        encoder_hidden: int = 256,
        encoder_rnn_hidden: int = 64,
        decoder_hidden: int = 256,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        encoder_rnn_layers: int = 1,
        encoder_head_hidden: int = 128,
        encoder_head_layers: int = 3,
        prior_head_hidden: int = 128,
        prior_head_layers: int = 3,
        skip_first_edge_type: bool = True,
        gumbel_temperature: float = 0.5,
        hard_gumbel_train: bool = False,
    ) -> None:
        super().__init__()
        self.edge_types = int(edge_types)
        self.gumbel_temperature = float(gumbel_temperature)
        self.hard_gumbel_train = bool(hard_gumbel_train)
        self.encoder = DynamicRelationalEncoder(
            node_dims,
            encoder_hidden,
            encoder_rnn_hidden,
            edge_types,
            encoder_dropout,
            encoder_rnn_layers,
            encoder_head_hidden,
            encoder_head_layers,
            prior_head_hidden,
            prior_head_layers,
        )
        self.decoder = DynamicRNNDecoder(
            node_dims,
            edge_types,
            decoder_hidden,
            decoder_dropout,
            skip_first_edge_type,
        )

    def sample_edges(
        self,
        logits: torch.Tensor,
        *,
        hard: bool | None = None,
        stochastic: bool | None = None,
    ) -> torch.Tensor:
        """Sample differentiably in training and use argmax deterministically in eval."""
        if hard is None:
            hard = self.hard_gumbel_train if self.training else True
        if stochastic is None:
            stochastic = self.training
        if stochastic:
            return F.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=hard,
                dim=-1,
            )
        if hard:
            types = logits.argmax(dim=-1)
            return F.one_hot(types, num_classes=self.edge_types).to(logits.dtype)
        return torch.softmax(logits / self.gumbel_temperature, dim=-1)

    def encode(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> dict[str, torch.Tensor | LSTMState]:
        if states.shape[2] < 2:
            raise ValueError("At least two states are required to infer transitions.")
        # x_t determines the edge used to predict x_{t+1}; the final state has
        # no target transition inside this sequence.
        return self.encoder(states[:, :, :-1], rel_rec, rel_send)

    def _decode(
        self,
        states: torch.Tensor,
        logits: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        *,
        teacher_forcing: bool,
        hard: bool | None,
        stochastic: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.decoder.initial_hidden(states[:, :, 0])
        predictions = []
        samples = []
        current = states[:, :, 0]
        for step in range(states.shape[2] - 1):
            if teacher_forcing:
                current = states[:, :, step]
            edge_sample = self.sample_edges(
                logits[:, :, step], hard=hard, stochastic=stochastic
            )
            current, hidden = self.decoder.single_step(
                current, hidden, edge_sample, rel_rec, rel_send
            )
            predictions.append(current)
            samples.append(edge_sample)
        return torch.stack(predictions, dim=2), torch.stack(samples, dim=2)

    def forward(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        *,
        graph_source: GraphSource = "posterior",
        teacher_forcing: bool = True,
        hard: bool | None = None,
        stochastic: bool | None = None,
    ) -> dict[str, torch.Tensor | LSTMState]:
        encoded = self.encode(states, rel_rec, rel_send)
        if graph_source not in {"posterior", "prior"}:
            raise ValueError("graph_source must be 'posterior' or 'prior'.")
        logits = encoded[f"{graph_source}_logits"]
        assert isinstance(logits, torch.Tensor)
        prediction, edge_sample = self._decode(
            states,
            logits,
            rel_rec,
            rel_send,
            teacher_forcing=teacher_forcing,
            hard=hard,
            stochastic=stochastic,
        )
        return {
            **encoded,
            "prediction": prediction,
            "edge_sample": edge_sample,
            "graph_source": graph_source,
        }

    def predict_future(
        self,
        history: torch.Tensor,
        horizon: int,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        *,
        hard: bool = True,
        stochastic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forecast without future observations, using the learned prior only."""
        if horizon < 1:
            raise ValueError("horizon must be positive.")
        encoded = self.encode(history, rel_rec, rel_send)
        prior_logits = encoded["prior_logits"]
        prior_state = encoded["prior_state"]
        assert isinstance(prior_logits, torch.Tensor)
        assert isinstance(prior_state, tuple)

        # Burn the decoder memory on observed transitions. Its outputs are
        # ignored because the true next observation is available in history.
        hidden = self.decoder.initial_hidden(history[:, :, 0])
        for step in range(history.shape[2] - 1):
            edge_sample = self.sample_edges(
                prior_logits[:, :, step], hard=hard, stochastic=stochastic
            )
            _, hidden = self.decoder.single_step(
                history[:, :, step], hidden, edge_sample, rel_rec, rel_send
            )

        current = history[:, :, -1]
        predictions = []
        future_logits = []
        future_edges = []
        for _ in range(horizon):
            logits, prior_state = self.encoder.prior_step(
                current, rel_rec, rel_send, prior_state
            )
            edge_sample = self.sample_edges(logits, hard=hard, stochastic=stochastic)
            current, hidden = self.decoder.single_step(
                current, hidden, edge_sample, rel_rec, rel_send
            )
            predictions.append(current)
            future_logits.append(logits)
            future_edges.append(edge_sample)
        return {
            "prediction": torch.stack(predictions, dim=2),
            "prior_logits": torch.stack(future_logits, dim=2),
            "edge_sample": torch.stack(future_edges, dim=2),
        }


def gaussian_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variance: float,
) -> torch.Tensor:
    """Fixed-variance Gaussian NLL used by the normalized dNRI experiments."""
    if variance <= 0:
        raise ValueError("variance must be positive.")
    return ((prediction - target) ** 2 / (2.0 * variance)).sum(dim=-1).mean()


def learned_prior_kl(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """KL(q(z^t|x^{1:T}) || p(z^t|x^{1:t})) across all edges/times."""
    posterior = torch.softmax(posterior_logits, dim=-1)
    log_posterior = torch.log_softmax(posterior_logits, dim=-1)
    log_prior = torch.log_softmax(prior_logits, dim=-1)
    del num_nodes  # Retained in the public API for parity with NRI loss helpers.
    return (posterior * (log_posterior - log_prior)).sum(dim=-1).mean()


def uniform_prior_kl(posterior_logits: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Optional regularizer toward a uniform categorical graph prior."""
    posterior = torch.softmax(posterior_logits, dim=-1)
    log_posterior = torch.log_softmax(posterior_logits, dim=-1)
    log_uniform = -torch.log(
        posterior_logits.new_tensor(float(posterior_logits.shape[-1]))
    )
    del num_nodes
    return (posterior * (log_posterior - log_uniform)).sum(dim=-1).mean()


def dnri_loss(
    output: dict[str, torch.Tensor | LSTMState],
    target: torch.Tensor,
    *,
    variance: float,
    num_nodes: int,
    kl_weight: float = 1.0,
    uniform_prior_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the negative dNRI evidence lower bound."""
    prediction = output["prediction"]
    posterior_logits = output["posterior_logits"]
    prior_logits = output["prior_logits"]
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(posterior_logits, torch.Tensor)
    assert isinstance(prior_logits, torch.Tensor)
    nll = gaussian_nll(prediction, target, variance)
    kl = learned_prior_kl(posterior_logits, prior_logits, num_nodes)
    uniform_kl = uniform_prior_kl(posterior_logits, num_nodes)
    loss = nll + kl_weight * kl + uniform_prior_weight * uniform_kl
    return {"loss": loss, "nll": nll, "kl": kl, "uniform_kl": uniform_kl}
