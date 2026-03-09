from src.core.config import config


class ModelManager:
    """Map Claude model names to provider-specific deployment names."""

    _PASSTHROUGH_PREFIXES = (
        "gpt-",
        "o1-",
        "o3-",
        "o4-",
        "ep-",
        "doubao-",
        "deepseek-",
    )

    def __init__(self, config):
        self.config = config

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern"""
        # If it's already a provider-specific model/deployment name, return as-is
        if claude_model.startswith(self._PASSTHROUGH_PREFIXES):
            return claude_model

        # Map based on model naming patterns
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.config.small_model
        elif "sonnet" in model_lower:
            return self.config.middle_model
        elif "opus" in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model


model_manager = ModelManager(config)
