"""Target model registry for SpecJAX."""

import importlib


TARGET_MODELS = {
    "glm_flash": ("specjax.models.target.glm_flash", "load_params", "glm_forward"),
    "glm5_fp8": ("specjax.models.target.glm5_fp8", "load_params", "glm5_forward"),
    "glm47_fp8": ("specjax.models.target.glm47_fp8", "load_params", "glm47_forward"),
    "qwen3": ("specjax.models.target.qwen3", "load_params", "qwen3_forward"),
    "qwen3_next": ("specjax.models.target.qwen3_next", "load_params", "qwen3_next_forward"),
    "qwen2": ("specjax.models.target.qwen2", "load_params", "qwen2_forward"),
    "qwen25": ("specjax.models.target.qwen25", "load_params", "qwen25_forward"),
    "llama": ("specjax.models.target.llama", "load_params", "llama_forward"),
    "minimax_m2": ("specjax.models.target.minimax_m2", "load_params", "minimax_m2_forward"),
}


def get_target(name: str) -> tuple:
    """Return (load_params_fn, forward_fn) for the given target model name."""
    if name not in TARGET_MODELS:
        raise ValueError(
            f"Unknown target model '{name}'. "
            f"Available: {list(TARGET_MODELS.keys())}"
        )
    module_path, load_fn_name, forward_fn_name = TARGET_MODELS[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, load_fn_name), getattr(mod, forward_fn_name)
