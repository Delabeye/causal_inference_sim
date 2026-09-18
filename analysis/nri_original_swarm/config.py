"""All user-editable parameters for the NRI swarm baseline.

The training command intentionally has no large argparse surface. Edit this file,
then run ``python -m analysis.nri_original_swarm.train`` from the project root.
"""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Data -----------------------------------------------------------------
    log_dir: Path = PROJECT_ROOT / "datasets/leader_switch_triangle_100runs_60s"
    # Keep this sparse-prior experiment separate from the uniform-prior
    # baseline so that the two checkpoints can be compared directly.
    output_dir: Path = PROJECT_ROOT / "causal_out/nri_original_swarm_leader_swap"
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3"
    )
    feature_columns: tuple[str, ...] = (
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_vx",
        "gt_vy",
        "gt_vz",
    )
    # Local exogenous context. A follower's logged target is generated from the
    # leader state, so only leaders/independent drones receive target_rel_*;
    # followers receive their desired offset instead.
    context_feature_columns: tuple[str, ...] = (
        "navigation_target_rel_x",
        "navigation_target_rel_y",
        "navigation_target_rel_z",
        "desired_offset_x",
        "desired_offset_y",
        "desired_offset_z",
        "wind_x",
        "wind_y",
        "wind_z",
        "nearest_obstacle_dist",
        "nearest_obstacle_dir_x",
        "nearest_obstacle_dir_y",
        "nearest_obstacle_dir_z",
        "external_force_x",
        "external_force_y",
        "external_force_z",
    )
    obstacle_distance_clip: float = 20.0
    downsample: int = 50
    seq_len: int = 49
    window_stride: int = 49
    max_windows_per_run: int | None = None

    # Explicit run lists override plan.csv and automatic run-level splitting.
    train_run_ids: list[int] = field(default_factory=list)
    val_run_ids: list[int] = field(default_factory=list)
    test_run_ids: list[int] = field(default_factory=list)
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    # "structural": fixed leader->follower graph, closest to original NRI.
    # "active": interaction_active occupancy from run_<id>_interactions.csv.
    ground_truth_mode: str = "structural"
    active_fraction_threshold: float = 0.05
    require_ground_truth: bool = True

    # Original NRI model ---------------------------------------------------
    edge_types: int = 2
    encoder_hidden: int = 256
    decoder_hidden: int = 256
    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0
    factor_graph: bool = True
    skip_first_edge_type: bool = True
    prediction_steps: int = 10
    gumbel_temperature: float = 0.5
    hard_gumbel_train: bool = False
    output_variance: float = 5e-5
    # There are 3 structural leader->follower edges among the 20 possible
    # directed off-diagonal edges (5 drones), hence an expected density of 0.15.
    prior: tuple[float, ...] | None = (0.75, 0.25)
    # The Gaussian NLL is summed over sequence steps and state dimensions, and
    # is consequently much larger than the raw categorical KL.  This weight
    # makes the sparse prior material without adding adjacency supervision.
    kl_weight: float = 1000.0

    # Optimization ---------------------------------------------------------
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 5e-4
    lr_decay: int = 200
    lr_gamma: float = 0.5
    seed: int = 42
    torch_threads: int = 1
    device: str = "auto"  # auto, cpu, mps, cuda
    num_workers: int = 0
    max_saved_run_heatmaps: int = 20
    max_saved_trajectory_runs: int = 20
    trajectory_windows_per_run: int = 1

    def validate(self) -> None:
        if self.edge_types != 2:
            raise ValueError("This UAV baseline currently expects two edge types.")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if self.downsample < 1 or self.window_stride < 1:
            raise ValueError("downsample and window_stride must be positive.")
        if self.obstacle_distance_clip <= 0:
            raise ValueError("obstacle_distance_clip must be positive.")
        if not 1 <= self.prediction_steps <= self.seq_len:
            raise ValueError("prediction_steps must lie in [1, seq_len].")
        if self.max_saved_trajectory_runs < 0 or self.trajectory_windows_per_run < 0:
            raise ValueError("Trajectory export limits must be non-negative.")
        if self.ground_truth_mode not in {"structural", "active"}:
            raise ValueError("ground_truth_mode must be 'structural' or 'active'.")
        if not 0.0 <= self.active_fraction_threshold <= 1.0:
            raise ValueError("active_fraction_threshold must lie in [0, 1].")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative.")
        if self.prior is not None:
            if len(self.prior) != self.edge_types:
                raise ValueError("prior must contain one probability per edge type.")
            if abs(sum(self.prior) - 1.0) > 1e-6 or min(self.prior) <= 0:
                raise ValueError("prior probabilities must be positive and sum to one.")


CONFIG = Config()
