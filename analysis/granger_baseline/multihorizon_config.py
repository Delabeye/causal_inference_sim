"""Configuration for the multi-horizon velocity-change Granger model."""

from dataclasses import dataclass
from pathlib import Path

from .config import Config as BaseConfig
from .config import PROJECT_ROOT


@dataclass
class Config(BaseConfig):
    output_dir: Path = (
        PROJECT_ROOT / "causal_out/granger_multihorizon_velocity_delta_nri_confirm"
    )
    horizon_steps: tuple[int, ...] = (5, 10, 20)
    horizon_aggregation: str = "max"

    def validate(self) -> None:
        super().validate()
        if not self.horizon_steps or min(self.horizon_steps) < 1:
            raise ValueError("horizon_steps must contain positive integers.")
        if len(set(self.horizon_steps)) != len(self.horizon_steps):
            raise ValueError("horizon_steps must not contain duplicates.")
        if self.horizon_aggregation != "max":
            raise ValueError("Only max horizon aggregation is currently supported.")


CONFIG = Config()
