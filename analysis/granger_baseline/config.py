"""Fixed, user-editable settings for the conditional Granger baseline."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Data -----------------------------------------------------------------
    log_dir: Path = (
        PROJECT_ROOT
        / "datasets/formation_30each_100s_periodic_perturb_border_isolated"
    )
    output_dir: Path = PROJECT_ROOT / "causal_out/granger_nri_original_swarm_confirm"
    comparison_nri_dir: Path | None = (
        PROJECT_ROOT / "causal_out/nri_original_swarm_confirm"
    )
    drone_names: tuple[str, ...] = (
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
    )
    state_columns: tuple[str, ...] = (
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_vx",
        "gt_vy",
        "gt_vz",
    )
    control_columns: tuple[str, ...] = (
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
    downsample: int = 6
    require_all_planned_runs: bool = True
    train_run_ids: list[int] = field(default_factory=list)
    val_run_ids: list[int] = field(default_factory=list)
    test_run_ids: list[int] = field(default_factory=list)
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    # Conditional VAR used internally by Granger --------------------------
    granger_lags: int = 10
    granger_ridge: float = 1.0
    use_controls: bool = True

    # Saved diagnostics ----------------------------------------------------
    max_saved_test_runs: int = 20
    seed: int = 42

    def validate(self) -> None:
        if self.downsample < 1:
            raise ValueError("downsample must be positive.")
        if self.granger_lags < 1:
            raise ValueError("granger_lags must be positive.")
        if self.granger_ridge < 0:
            raise ValueError("granger_ridge must be non-negative.")
        if self.obstacle_distance_clip <= 0:
            raise ValueError("obstacle_distance_clip must be positive.")
        if len(self.state_columns) != 6:
            raise ValueError("This baseline expects 3D position and 3D velocity.")


CONFIG = Config()
