"""Modern PyTorch implementation of the original factor-graph NRI modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Two-layer ELU MLP with the batch normalization used by original NRI."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.1)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape
        output = F.elu(self.fc1(inputs))
        output = F.dropout(output, self.dropout, training=self.training)
        output = F.elu(self.fc2(output))
        output = self.bn(output.reshape(-1, output.shape[-1]))
        return output.reshape(*shape[:-1], -1)


def relation_matrices(num_nodes: int, device: torch.device | None = None):
    """Enumerate every directed off-diagonal pair as receiver/sender one-hot rows."""
    receivers = []
    senders = []
    for receiver in range(num_nodes):
        for sender in range(num_nodes):
            if receiver == sender:
                continue
            receivers.append(receiver)
            senders.append(sender)
    rel_rec = F.one_hot(torch.tensor(receivers), num_classes=num_nodes).float()
    rel_send = F.one_hot(torch.tensor(senders), num_classes=num_nodes).float()
    if device is not None:
        rel_rec = rel_rec.to(device)
        rel_send = rel_send.to(device)
    return rel_rec, rel_send


class MLPEncoder(nn.Module):
    """Infer one categorical edge type for each ordered node pair."""

    def __init__(
        self,
        timesteps: int,
        node_dims: int,
        hidden: int,
        edge_types: int,
        dropout: float = 0.0,
        factor_graph: bool = True,
        context_dims: int = 0,
    ):
        super().__init__()
        self.factor_graph = bool(factor_graph)
        self.context_dims = int(context_dims)
        self.mlp1 = MLP(
            timesteps * (node_dims + self.context_dims), hidden, hidden, dropout
        )
        self.mlp2 = MLP(2 * hidden, hidden, hidden, dropout)
        self.mlp3 = MLP(hidden, hidden, hidden, dropout)
        mlp4_in = 3 * hidden if factor_graph else 2 * hidden
        self.mlp4 = MLP(mlp4_in, hidden, hidden, dropout)
        self.fc_out = nn.Linear(hidden, edge_types)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.1)

    @staticmethod
    def node_to_edge(nodes: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor):
        receivers = torch.matmul(rel_rec, nodes)
        senders = torch.matmul(rel_send, nodes)
        return torch.cat([senders, receivers], dim=-1)

    @staticmethod
    def edge_to_node(edges: torch.Tensor, rel_rec: torch.Tensor):
        incoming = torch.matmul(rel_rec.transpose(0, 1), edges)
        return incoming / incoming.shape[1]

    def forward(
        self,
        states: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        contexts: torch.Tensor | None = None,
    ):
        # states: [batch, nodes, timesteps, features]
        if self.context_dims:
            expected = (*states.shape[:3], self.context_dims)
            if contexts is None or tuple(contexts.shape) != expected:
                actual = None if contexts is None else tuple(contexts.shape)
                raise ValueError(f"contexts has shape {actual}, expected {expected}")
            node_inputs = torch.cat([states, contexts], dim=-1)
        else:
            node_inputs = states
        nodes = node_inputs.reshape(node_inputs.shape[0], node_inputs.shape[1], -1)
        nodes = self.mlp1(nodes)
        edges = self.mlp2(self.node_to_edge(nodes, rel_rec, rel_send))
        skip = edges
        if self.factor_graph:
            nodes = self.mlp3(self.edge_to_node(edges, rel_rec))
            edges = self.node_to_edge(nodes, rel_rec, rel_send)
        else:
            edges = self.mlp3(edges)
        edges = self.mlp4(torch.cat([edges, skip], dim=-1))
        return self.fc_out(edges)


class RNNDecoder(nn.Module):
    """Original NRI recurrent decoder with one fixed graph per sequence."""

    def __init__(
        self,
        node_dims: int,
        edge_types: int,
        hidden: int,
        dropout: float = 0.0,
        skip_first_edge_type: bool = True,
        context_dims: int = 0,
    ):
        super().__init__()
        self.message_fc1 = nn.ModuleList(
            nn.Linear(2 * hidden, hidden) for _ in range(edge_types)
        )
        self.message_fc2 = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(edge_types)
        )

        # GRU-style update from Eq. (14) of the NRI paper. The relational
        # messages play the role of the recurrent input to each node memory.
        self.hidden_reset = nn.Linear(hidden, hidden, bias=False)
        self.hidden_update = nn.Linear(hidden, hidden, bias=False)
        self.hidden_candidate = nn.Linear(hidden, hidden, bias=False)
        self.input_reset = nn.Linear(node_dims, hidden)
        self.input_update = nn.Linear(node_dims, hidden)
        self.input_candidate = nn.Linear(node_dims, hidden)
        self.context_dims = int(context_dims)
        if self.context_dims:
            self.context_reset = nn.Linear(self.context_dims, hidden, bias=False)
            self.context_update = nn.Linear(self.context_dims, hidden, bias=False)
            self.context_candidate = nn.Linear(self.context_dims, hidden, bias=False)
        else:
            self.context_reset = None
            self.context_update = None
            self.context_candidate = None

        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, hidden)
        self.out_fc3 = nn.Linear(hidden, node_dims)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.skip_first = bool(skip_first_edge_type)

    def single_step(
        self,
        states: torch.Tensor,
        contexts: torch.Tensor,
        hidden: torch.Tensor,
        edge_sample: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Messages depend on the recurrent representations of sender/receiver.
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pair_hidden = torch.cat([senders, receivers], dim=-1)
        messages = torch.zeros(
            *pair_hidden.shape[:-1], self.hidden,
            device=states.device,
            dtype=states.dtype,
        )
        start_type = 1 if self.skip_first else 0
        active_types = max(len(self.message_fc1) - start_type, 1)
        for edge_type in range(start_type, len(self.message_fc1)):
            message = torch.tanh(self.message_fc1[edge_type](pair_hidden))
            message = F.dropout(message, self.dropout, training=self.training)
            message = torch.tanh(self.message_fc2[edge_type](message))
            messages = messages + (
                message * edge_sample[..., edge_type : edge_type + 1] / active_types
            )

        aggregated = torch.matmul(
            messages.transpose(-2, -1), rel_rec
        ).transpose(-2, -1)
        aggregated = aggregated / states.shape[-2]

        reset_logits = self.input_reset(states) + self.hidden_reset(aggregated)
        update_logits = self.input_update(states) + self.hidden_update(aggregated)
        candidate_input = self.input_candidate(states)
        if self.context_dims:
            reset_logits = reset_logits + self.context_reset(contexts)
            update_logits = update_logits + self.context_update(contexts)
            candidate_input = candidate_input + self.context_candidate(contexts)
        reset = torch.sigmoid(reset_logits)
        update = torch.sigmoid(update_logits)
        candidate = torch.tanh(
            candidate_input + reset * self.hidden_candidate(aggregated)
        )
        hidden = (1.0 - update) * candidate + update * hidden

        output = F.dropout(F.relu(self.out_fc1(hidden)), self.dropout, training=self.training)
        output = F.dropout(F.relu(self.out_fc2(output)), self.dropout, training=self.training)
        delta = self.out_fc3(output)
        return states + delta, hidden

    def forward(
        self,
        states: torch.Tensor,
        edge_sample: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        prediction_steps: int,
        contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ground truth is fed every prediction_steps transitions. Between those
        # points, predictions are fed back autoregressively; memory is never reset.
        states_t = states.transpose(1, 2).contiguous()
        if self.context_dims:
            expected = (*states.shape[:3], self.context_dims)
            if contexts is None or tuple(contexts.shape) != expected:
                actual = None if contexts is None else tuple(contexts.shape)
                raise ValueError(f"contexts has shape {actual}, expected {expected}")
            contexts_t = contexts.transpose(1, 2).contiguous()
        else:
            contexts_t = states_t.new_empty(*states_t.shape[:3], 0)
        hidden = torch.zeros(
            states.shape[0],
            states.shape[1],
            self.hidden,
            device=states.device,
            dtype=states.dtype,
        )
        predictions = []
        previous_prediction = None
        for step in range(states_t.shape[1] - 1):
            if step % prediction_steps == 0:
                current = states_t[:, step]
            else:
                current = previous_prediction
            previous_prediction, hidden = self.single_step(
                current, contexts_t[:, step], hidden, edge_sample, rel_rec, rel_send
            )
            predictions.append(previous_prediction)

        return torch.stack(predictions, dim=1).transpose(1, 2).contiguous()


def gaussian_nll(prediction: torch.Tensor, target: torch.Tensor, variance: float):
    """Negative Gaussian log likelihood, normalized exactly as original NRI."""
    return ((prediction - target) ** 2 / (2.0 * variance)).sum() / (
        target.shape[0] * target.shape[1]
    )


def categorical_kl(probabilities: torch.Tensor, num_nodes: int, prior=None):
    """KL term up to the constant omitted by the original implementation."""
    eps = 1e-16
    if prior is None:
        terms = probabilities * torch.log(probabilities + eps)
    else:
        log_prior = torch.log(
            torch.as_tensor(prior, dtype=probabilities.dtype, device=probabilities.device)
        )
        terms = probabilities * (torch.log(probabilities + eps) - log_prior)
    return terms.sum() / (num_nodes * probabilities.shape[0])
