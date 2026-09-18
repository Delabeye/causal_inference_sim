"""SRI-inspired causal graph inference with explicit physical interventions."""

import numpy as _numpy  # noqa: F401 -- load the shared macOS OpenMP runtime first

from .config import CONFIG, Config
from .model import (
    DynamicGraphPrior,
    InterventionalLossWeights,
    InterventionalSRIEncoder,
    InterventionalSRIModel,
    interventional_sri_loss,
)

__all__ = [
    "CONFIG",
    "Config",
    "DynamicGraphPrior",
    "InterventionalLossWeights",
    "InterventionalSRIEncoder",
    "InterventionalSRIModel",
    "interventional_sri_loss",
]
