"""Configuration for the dNRI swarm baseline.

The model follows the learned-prior dNRI objective from Graber & Schwing
(CVPR 2020).  Relational labels are used only for evaluation, never in the
training loss.
"""

from dataclasses import dataclass
from pathlib import Path

from analysis.nri_original_swarm.config import Config as NRIConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config(NRIConfig):
    # Use the same window loader and state normalization as the NRI baseline.
    log_dir: Path = PROJECT_ROOT / "datasets/all_formations_forks_60s"
    output_dir: Path = PROJECT_ROOT / "causal_out/dnri_swarm"

    # The paper's core model receives trajectories only. Keeping this empty
    # prevents privileged simulator context from making the baseline stronger.
    context_feature_columns: tuple[str, ...] = ()
    downsample: int = 6
    seq_len: int = 49
    window_stride: int = 24

    # Per-time-step edge encoder: spatial GNN followed by forward/backward LSTMs.
    encoder_hidden: int = 256
    encoder_rnn_hidden: int = 64
    encoder_rnn_layers: int = 1
    encoder_head_hidden: int = 128
    encoder_head_layers: int = 3
    prior_head_hidden: int = 128
    prior_head_layers: int = 3
    encoder_dropout: float = 0.0

    # Recurrent relational decoder from dNRI/NRI.
    decoder_hidden: int = 256
    decoder_dropout: float = 0.0
    skip_first_edge_type: bool = True
    edge_types: int = 2

    # ELBO: Gaussian reconstruction + KL(q(z^t|x) || p(z^t|x^{1:t})).
    output_variance: float = 5e-5
    kl_weight: float = 1.0
    prior: tuple[float, ...] | None = None
    uniform_prior_weight: float = 0.0
    gumbel_temperature: float = 0.5
    hard_gumbel_train: bool = False
    teacher_forcing: bool = True

    # Forecast evaluation: observe burn_in_steps, predict the rest using only
    # the causal learned prior and the recurrent decoder.
    burn_in_steps: int = 16

    # Optimization defaults.
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 5e-4
    lr_decay: int = 100
    lr_gamma: float = 0.5

    # Kept for compatibility with the shared loader/config validation. dNRI
    # selects teacher forcing explicitly instead of NRI's prediction_steps.
    prediction_steps: int = 1

    def validate(self) -> None:
        super().validate()
        if self.encoder_rnn_hidden < 1 or self.encoder_rnn_layers < 1:
            raise ValueError("encoder_rnn_hidden and encoder_rnn_layers must be positive.")
        if min(
            self.encoder_head_hidden,
            self.encoder_head_layers,
            self.prior_head_hidden,
            self.prior_head_layers,
        ) < 1:
            raise ValueError("Encoder/prior head sizes and layer counts must be positive.")
        if not 1 <= self.burn_in_steps < self.seq_len:
            raise ValueError("burn_in_steps must lie in [1, seq_len - 1].")
        if self.uniform_prior_weight < 0.0:
            raise ValueError("uniform_prior_weight must be non-negative.")
        if self.context_feature_columns:
            raise ValueError(
                "The paper-faithful dNRI baseline uses states only; "
                "context_feature_columns must be empty."
            )


CONFIG = Config()
