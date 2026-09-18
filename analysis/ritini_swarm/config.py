"""Editable configuration for the RiTINI swarm baseline."""

from dataclasses import dataclass
from pathlib import Path

from analysis.nri_ground_truth_continuous.config import (
    PROJECT_ROOT,
    Config as DataConfig,
)


@dataclass
class Config(DataConfig):
    """RiTINI settings plus the shared run-level data configuration."""

    # This dataset contains paired nominal/controlled-push trajectories and
    # aligned interaction matrices for held-out evaluation.
    log_dir: Path = PROJECT_ROOT / "logs_100runs_paired"
    output_dir: Path = PROJECT_ROOT / "causal_out/ritini_swarm"
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
    )
    context_feature_columns: tuple[str, ...] = ()
    downsample: int = 30
    seq_len: int = 60
    window_stride: int = 60
    require_all_planned_runs: bool = False

    # RiTINI architecture. The official implementation uses an LSTM history
    # encoder, a multi-head GAT vector field and a Neural ODE integrator.
    history_steps: int = 8
    latent_dim: int = 64
    attention_heads: int = 4
    field_hidden: int = 128
    dropout: float = 0.1
    negative_slope: float = 0.2
    ode_method: str = "rk4"  # euler or rk4
    ode_substeps: int = 1
    use_observed_dt: bool = False
    integration_dt: float = 0.1
    time_scale: float = 1.0

    # The complete directed graph is the permissive graph. A prior can be
    # uniform/complete or estimated from training-only simulator matrices.
    # By default lambda_prior=0, so ground truth is evaluation-only.
    prior_mode: str = "complete"  # complete, structural_mean, active_mean
    prior_threshold: float = 0.5
    lambda_prior: float = 0.0
    lambda_entropy: float = 1e-3
    lambda_graph_smooth: float = 1e-2
    lambda_velocity: float = 0.2
    evaluation_target: str = "structural"  # structural, active, continuous
    attention_threshold: float = 0.2

    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 5e-4
    lr_decay: int = 100
    lr_gamma: float = 0.5

    # Fields inherited only for compatibility with the shared data loader.
    graph_supervision: bool = False
    detach_graph_for_decoder: bool = False
    observed_steps: int = 8
    multi_horizon_steps: tuple[int, ...] = (1,)

    def validate(self) -> None:
        super().validate()
        if not 2 <= self.history_steps < self.seq_len:
            raise ValueError("history_steps must lie in [2, seq_len).")
        if self.observed_steps != self.history_steps:
            raise ValueError("observed_steps must equal history_steps for this baseline.")
        if self.latent_dim < 1 or self.field_hidden < 1:
            raise ValueError("latent dimensions must be positive.")
        if self.attention_heads < 1 or self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads.")
        if self.ode_method not in {"euler", "rk4"}:
            raise ValueError("ode_method must be 'euler' or 'rk4'.")
        if self.ode_substeps < 1:
            raise ValueError("ode_substeps must be positive.")
        if min(self.integration_dt, self.time_scale) <= 0:
            raise ValueError("integration_dt and time_scale must be positive.")
        if self.prior_mode not in {"complete", "structural_mean", "active_mean"}:
            raise ValueError("Unsupported prior_mode.")
        if not 0 <= self.prior_threshold <= 1:
            raise ValueError("prior_threshold must lie in [0, 1].")
        if min(
            self.lambda_prior,
            self.lambda_entropy,
            self.lambda_graph_smooth,
            self.lambda_velocity,
        ) < 0:
            raise ValueError("RiTINI loss weights must be non-negative.")
        if self.evaluation_target not in {"structural", "active", "continuous"}:
            raise ValueError("Unsupported evaluation_target.")
        if not 0 <= self.attention_threshold <= 1:
            raise ValueError("attention_threshold must lie in [0, 1].")


CONFIG = Config()
