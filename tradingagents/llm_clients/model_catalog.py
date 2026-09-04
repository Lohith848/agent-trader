"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

ModelOption = tuple[str, str]
ProviderModeOptions = dict[str, dict[str, list[ModelOption]]]

# Providers that serve many / frequently-changing models: offer only "Custom
# model ID" rather than a list that goes stale.
_CUSTOM_ONLY: dict[str, list[ModelOption]] = {
    "quick": [("Custom model ID", "custom")],
    "deep": [("Custom model ID", "custom")],
}


MODEL_OPTIONS: ProviderModeOptions = {
    "google": {
        "quick": [
            ("Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
            ("Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash"),
        ],
    },
    "ollama": {
        "quick": [
            ("Llama 3.2 (3B)", "llama3.2:latest"),
            ("Llama 3 (8B)", "llama3:latest"),
            ("Qwen 2.5 Coder (7B)", "qwen2.5-coder:7b"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Llama 3 (8B)", "llama3:latest"),
            ("Qwen 2.5 Coder (7B)", "qwen2.5-coder:7b"),
            ("Llama 3.2 (3B)", "llama3.2:latest"),
            ("Custom model ID", "custom"),
        ],
    },
    "groq": _CUSTOM_ONLY,
    "nvidia": _CUSTOM_ONLY,
    "openrouter": _CUSTOM_ONLY,
}


def get_model_options(provider: str, mode: str) -> list[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return MODEL_OPTIONS[provider.lower()][mode]


def get_known_models() -> dict[str, list[str]]:
    """Build known model names from the shared CLI catalog."""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }
