"""SRI-inspired dynamic relational inference with binary edge semantics."""

from .config import CONFIG, Config
from .model import SRIBinaryModel, sri_binary_loss

__all__ = ["CONFIG", "Config", "SRIBinaryModel", "sri_binary_loss"]
