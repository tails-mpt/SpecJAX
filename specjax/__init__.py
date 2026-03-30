"""SpecJAX — EAGLE3 Speculative Decoding Training in JAX/TPU."""

__version__ = "0.1.0"

from specjax.models.target import get_target
from specjax.models.draft.eagle3 import Eagle3Config, eagle3_forward, load_eagle3_params
