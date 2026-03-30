"""EAGLE3 draft model."""

from specjax.models.draft.eagle3 import (
    Eagle3Config,
    eagle3_forward,
    load_eagle3_params,
    save_eagle3_checkpoint,
    compute_loss,
    compute_ttt_loss,
)
