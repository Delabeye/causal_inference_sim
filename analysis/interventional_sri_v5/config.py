"""Configuration of the full SRI + intervention experiment (v5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Dataset --------------------------------------------------------------
    log_dir: Path = PROJECT_ROOT / "datasets/all_formations_forks_60s"
    output_dir: Path = PROJECT_ROOT / "causal_out/interventional_sri_v5"
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
    )
    state_dim: int = 6
    intervention_dim: int = 4
    context_dim: int = 0
    # The existing learning traces contain one row every ~16.64 ms (~60 Hz).
    # Keeping one row out of six gives the intended ~10 Hz learning grid.
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
    # Type 0 is always no-interaction.  Types 1 and 2 are two unsupervised
    # interaction mechanisms, mirroring SRI's typed physical relations while
    # existence evaluation marginalizes both into one structural edge.
    num_edge_types: int = 3
    num_node_states: int = 3
    encoder_hidden: int = 128
    attention_heads: int = 4
    decoder_hidden: int = 128
    message_hidden: int = 128
    dropout: float = 0.1
    strength_floor: float = 0.10
    initial_edge_probability: float = 0.25
    gumbel_temperature: float = 1.0
    final_gumbel_temperature: float = 0.30
    temperature_anneal_epochs: int = 150
    hard_edge_threshold: float = 0.5

    # Objective ------------------------------------------------------------
    state_sigma: float = 0.1
    lambda_history: float = 0.5
    lambda_intervention: float = 1.0
    lambda_effect: float = 1.0
    lambda_non_target_effect: float = 1.0
    lambda_graph_necessity: float = 0.5
    lambda_graph_contrast: float = 0.5
    beta_edge_kl: float = 0.1
    beta_node_kl: float = 0.1
    kl_warmup_epochs: int = 50
    graph_margin: float = 0.05
    effect_scale_floor: float = 0.02
    non_target_min_effect: float = 0.01
    horizon_decay: float = 0.99

    # Optimization ---------------------------------------------------------
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 5e-4
    lr_decay: int = 100
    lr_gamma: float = 0.5
    gradient_clip: float = 5.0
    minimum_checkpoint_epoch: int = 1
    checkpoint_graph_gap_weight: float = 1.0
    seed: int = 42
    torch_threads: int = 1
    num_workers: int = 0
    device: str = "auto"

    def validate(self) -> None:
        positive = (
            self.state_dim,
            self.intervention_dim,
            self.downsample,
            self.history_steps,
            self.rollout_steps,
            self.baseline_window_stride,
            self.num_edge_types,
            self.num_node_states,
            self.encoder_hidden,
            self.attention_heads,
            self.decoder_hidden,
            self.message_hidden,
            self.temperature_anneal_epochs,
            self.kl_warmup_epochs,
            self.batch_size,
            self.epochs,
        )
        if min(positive) < 1:
            raise ValueError("Dimensions, horizons and training counts must be positive.")
        if self.num_edge_types < 2:
            raise ValueError("num_edge_types must include no-edge and at least one active type.")
        if self.num_node_states < 2:
            raise ValueError("num_node_states must be at least two.")
        if len(self.drone_names) < 2:
            raise ValueError("At least two drones are required.")
        if self.context_dim < 0:
            raise ValueError("context_dim must be non-negative.")
        if self.encoder_hidden % self.attention_heads:
            raise ValueError("encoder_hidden must be divisible by attention_heads.")
        if self.batch_size % 2:
            raise ValueError("batch_size must be even for balanced paired batches.")
        if not 0.0 <= self.strength_floor < 1.0:
            raise ValueError("strength_floor must lie in [0, 1).")
        for name in (
            "initial_edge_probability",
            "hard_edge_threshold",
            "train_fraction",
            "val_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1).")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be below one.")
        if self.gumbel_temperature <= 0 or self.final_gumbel_temperature <= 0:
            raise ValueError("Gumbel temperatures must be positive.")
        if self.state_sigma <= 0 or self.effect_scale_floor <= 0:
            raise ValueError("Noise and effect scales must be positive.")
        if self.intervention_force_scale <= 0 or self.time_tolerance_s <= 0:
            raise ValueError("Force scale and time tolerance must be positive.")
        if not 0.0 < self.horizon_decay <= 1.0:
            raise ValueError("horizon_decay must lie in (0, 1].")
        loss_weights = (
            self.lambda_history,
            self.lambda_intervention,
            self.lambda_effect,
            self.lambda_non_target_effect,
            self.lambda_graph_necessity,
            self.lambda_graph_contrast,
            self.beta_edge_kl,
            self.beta_node_kl,
            self.graph_margin,
            self.non_target_min_effect,
            self.checkpoint_graph_gap_weight,
        )
        if min(loss_weights) < 0:
            raise ValueError("Loss and selection weights must be non-negative.")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps or cuda.")


CONFIG = Config()


__all__ = ["CONFIG", "Config", "PROJECT_ROOT"]
