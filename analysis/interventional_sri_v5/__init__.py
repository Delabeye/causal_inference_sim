"""Full SRI-style dynamic relational inference with physical interventions."""

# On macOS, NumPy and PyTorch can otherwise load their OpenMP runtimes in the
# opposite order through package imports and trigger ``OMP Error #15`` before
# the CLI starts.  Import NumPy first, as done by the repository test suite.
import numpy as _np  # noqa: F401

from .config import CONFIG, Config
from .model import (
    InterventionalSRIV5,
    SRIV5LossWeights,
    interventional_sri_v5_loss,
)

__all__ = [
    "CONFIG",
    "Config",
    "InterventionalSRIV5",
    "SRIV5LossWeights",
    "interventional_sri_v5_loss",
]
