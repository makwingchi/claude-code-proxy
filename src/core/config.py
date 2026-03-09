import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Runtime configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
        self.enable_test_connection = _parse_bool_env("ENABLE_TEST_CONNECTION", default=False)

    def validate_api_key(self) -> bool:
        """Return whether an upstream API key is configured."""
        return bool(self.openai_api_key)

    def validate_client_api_key(self, client_api_key: str) -> bool:
        """Validate the client-provided Anthropic API key."""
        if not self.anthropic_api_key:
            return True
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self) -> Dict[str, str]:
        """Build custom upstream headers from CUSTOM_HEADER_* variables."""
        custom_headers: Dict[str, str] = {}
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("CUSTOM_HEADER_"):
                continue
            header_name = env_key[14:]
            if not header_name:
                continue
            custom_headers[header_name.replace("_", "-")] = env_value
        return custom_headers


config = Config()
if not config.anthropic_api_key:
    logger.warning(
        "ANTHROPIC_API_KEY not set; client API key validation is disabled."
    )
logger.info("Configuration loaded: base_url=%s", config.openai_base_url)
