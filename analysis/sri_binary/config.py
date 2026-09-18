"""Editable configuration for the SRI-inspired binary graph model."""

from dataclasses import dataclass
from pathlib import Path

from analysis.nri_dynamic_categorical.config import Config as DynamicConfig
from analysis.nri_ground_truth_continuous.config import PROJECT_ROOT


@dataclass
class Config(DynamicConfig):
    # Dataset used for the leader-switch experiment.
    log_dir: Path = PROJECT_ROOT / "datasets/leader_switch_triangle_100runs_60s"
    output_dir: Path = PROJECT_ROOT / "causal_out/sri_binary_leader_switch"
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
        "drone_4",
    )

    # SRI is given trajectories only. In particular, follower targets are not
    # exposed to the model because they already contain the leader state.
    context_feature_columns: tuple[str, ...] = ()
    downsample: int = 50
    seq_len: int = 49
    window_stride: int = 12
    max_windows_per_run: int | None = None
    multi_horizon_steps: tuple[int, ...] = (3, 6, 12, 21, 30, 36)

    # Binary ontology: 0=no-edge, 1=edge. Strength is a separate conditional
    # scalar and never becomes an additional edge type.
    edge_types: int = 2
    attention_heads: int = 4
    edge_prior: tuple[float, float] = (0.85, 0.15)

    # Unsupervised objective. Simulator matrices remain evaluation-only.
    state_sigma: float = 0.1
    beta_kl: float = 1.0
    beta_sparse_prior: float = 1.0
    lambda_graph_smooth: float = 0.05
    lambda_strength_l1: float = 0.01
    lambda_context: float = 0.0
    evaluation_target: str = "structural"  # structural or active

    # Architecture/training.
    encoder_hidden: int = 128
    message_hidden: int = 128
    decoder_hidden: int = 128
    context_hidden: int = 64
    dropout: float = 0.1
    reconstruction_steps: int = 10
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 5e-4
    lr_decay: int = 100
    lr_gamma: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.edge_types != 2:
            raise ValueError("SRI binary requires exactly two types: no-edge and edge.")
        if self.encoder_hidden % self.attention_heads:
            raise ValueError("encoder_hidden must be divisible by attention_heads.")
        if len(self.edge_prior) != 2:
            raise ValueError("edge_prior must contain (p_no_edge, p_edge).")
        if min(self.edge_prior) <= 0 or abs(sum(self.edge_prior) - 1.0) > 1e-6:
            raise ValueError("edge_prior entries must be positive and sum to one.")
        if min(
            self.beta_sparse_prior,
            self.lambda_graph_smooth,
            self.lambda_strength_l1,
        ) < 0:
            raise ValueError("SRI graph regularization weights must be non-negative.")
        if self.evaluation_target not in {"structural", "active"}:
            raise ValueError("evaluation_target must be 'structural' or 'active'.")


CONFIG = Config()
