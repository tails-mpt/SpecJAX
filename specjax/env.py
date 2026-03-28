"""TPU environment configuration — must be imported before JAX."""

import os


def configure_tpu_env() -> None:
    """Set environment variables required for JAX on TPU.

    Must be called before any JAX import. Handles both single-host and
    multi-host (JAX_NUM_PROCESSES > 1) configurations.
    """
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault(
        "XLA_PERSISTENT_CACHE_PATH",
        os.path.join(os.environ.get("HOME", "/tmp"), "xla_cache_jax"),
    )

    # Single-host bounds: only set when NOT in multi-host mode.
    # When JAX_NUM_PROCESSES > 1 the JAX coordinator handles topology discovery;
    # forcing TPU_PROCESS_BOUNDS=1,1,1 here would override it and collapse every
    # worker to process 0/1 (single-host mode), breaking distributed training.
    if int(os.environ.get("JAX_NUM_PROCESSES", "1")) == 1:
        os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
        os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
        os.environ.setdefault("CLOUD_TPU_TASK_ID", "0")
