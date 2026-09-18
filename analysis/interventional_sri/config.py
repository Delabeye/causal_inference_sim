"""Editable configuration for the interventional SRI model.

Generate the physical fork dataset first, then run::

    python -m analysis.interventional_sri.train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Dataset --------------------------------------------------------------
    log_dir: Path = PROJECT_ROOT / "datasets/all_formations_forks_60s"
    # Keep v3 checkpoints separate from fixed/discrete decoder experiments.
    output_dir: Path = PROJECT_ROOT / "causal_out/interventional_sri_dynamic_v3"
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
    )
    state_dim: int = 6  # x, y, z, vx, vy, vz
    intervention_dim: int = 4  # force_x, force_y, force_z, active
    context_dim: int = 0

    # The synchronized simulator learning trace is logged at 80 Hz. Keeping one
    # sample every eight gives a 10 Hz learning grid: 20 history steps = 2 s
    # and 80 rollout steps = 8 s.
    # The current fork dataset is logged at ~60 Hz; stride 6 gives ~10 Hz.
    downsample: int = 6
    history_steps: int = 20
    rollout_steps: int = 80
    baseline_window_stride: int = 80
    max_baseline_windows_per_run: int | None = 6
    intervention_force_scale: float = 0.02
    time_tolerance_s: float = 0.02
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    # Model ----------------------------------------------------------------
    encoder_hidden: int = 128
    attention_heads: int = 4
    decoder_hidden: int = 128
    message_hidden: int = 128
    dynamic_graph_hidden: int = 128
    dynamic_graph_delta_scale: float = 0.1
    dropout: float = 0.1
    # The decoder starts with the posterior mean P(edge)*S, then gradually
    # approaches a straight-through categorical graph.
    gumbel_temperature: float = 1.0
    final_gumbel_temperature: float = 0.3
    soft_graph_warmup_epochs: int = 30
    graph_discretization_epochs: int = 120
    hard_gumbel: bool = True

    # Objective ------------------------------------------------------------
    state_sigma: float = 0.1
    edge_prior_probability: float = 0.15
    lambda_intervention: float = 1.0
    lambda_effect: float = 2.0
    beta_kl: float = 0.1
    # Do not sparsify P(edge)*S: doing so directly collapses S. Sparsity is
    # already carried by the categorical existence prior/KL.
    lambda_strength: float = 0.0
    lambda_graph_smoothness: float = 1e-3
    lambda_graph_necessity: float = 1.0
    # Require at least a 5% relative improvement over the exact zero-effect
    # prediction on responsive, non-intervened nodes.
    graph_necessity_margin: float = 0.05
    graph_necessity_min_effect: float = 1e-2
    horizon_decay: float = 0.99

    # Optimization ---------------------------------------------------------
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 5e-4
    lr_decay: int = 100
    lr_gamma: float = 0.5
    gradient_clip: float = 5.0
    seed: int = 42
    torch_threads: int = 1
    num_workers: int = 0
    device: str = "auto"  # auto, cpu, mps, cuda

    def validate(self) -> None:
        if len(self.drone_names) < 2:
            raise ValueError("At least two drones are required.")
        if min(
            self.state_dim,
            self.intervention_dim,
            self.downsample,
            self.history_steps,
            self.rollout_steps,
            self.baseline_window_stride,
            self.encoder_hidden,
            self.attention_heads,
            self.decoder_hidden,
            self.message_hidden,
            self.dynamic_graph_hidden,
            self.graph_discretization_epochs,
            self.batch_size,
            self.epochs,
        ) < 1:
            raise ValueError("Dimensions, sampling intervals and counts must be positive.")
        if self.context_dim < 0:
            raise ValueError("context_dim must be non-negative.")
        if self.encoder_hidden % self.attention_heads:
            raise ValueError("encoder_hidden must be divisible by attention_heads.")
        if not 0.0 < self.gumbel_temperature:
            raise ValueError("gumbel_temperature must be positive.")
        if not 0.0 < self.final_gumbel_temperature:
            raise ValueError("final_gumbel_temperature must be positive.")
        if self.final_gumbel_temperature > self.gumbel_temperature:
            raise ValueError(
                "final_gumbel_temperature must not exceed gumbel_temperature."
            )
        if self.soft_graph_warmup_epochs < 0:
            raise ValueError("soft_graph_warmup_epochs must be non-negative.")
        if not 0.0 < self.dynamic_graph_delta_scale:
            raise ValueError("dynamic_graph_delta_scale must be positive.")
        if not 0.0 < self.state_sigma:
            raise ValueError("state_sigma must be positive.")
        if not 0.0 < self.edge_prior_probability < 1.0:
            raise ValueError("edge_prior_probability must lie in (0, 1).")
        if self.intervention_force_scale <= 0 or self.time_tolerance_s <= 0:
            raise ValueError("Force scale and time tolerance must be positive.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1).")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in (0, 1).")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be below one.")
        if min(
            self.lambda_intervention,
            self.lambda_effect,
            self.beta_kl,
            self.lambda_strength,
            self.lambda_graph_smoothness,
            self.lambda_graph_necessity,
            self.graph_necessity_margin,
            self.graph_necessity_min_effect,
        ) < 0:
            raise ValueError("Loss weights must be non-negative.")
        if not 0.0 < self.horizon_decay <= 1.0:
            raise ValueError("horizon_decay must lie in (0, 1].")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps or cuda.")


CONFIG = Config()


__all__ = ["CONFIG", "Config", "PROJECT_ROOT"]
