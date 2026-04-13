
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AgentConfig:
    """
    Configuration management for Data Insights Assistant for Non-Technical Users.
    Handles environment variable loading, API key management, LLM config, domain settings,
    validation, error handling, and default values.
    """

    # Required environment variables for external services
    REQUIRED_ENV_VARS = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # LLM configuration defaults
    LLM_CONFIG_DEFAULTS = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a Data Insights Assistant for non-technical users. Your role is to answer user questions about data by providing clear, concise, and accurate insights based strictly on the retrieved knowledge base content. Do not use technical jargon or assume prior data knowledge. If the answer cannot be found in the provided documents, politely inform the user and suggest rephrasing their question. Always ensure your responses are easy to understand and directly address the user's query."
        ),
        "user_prompt_template": (
            "Please enter your question about the data. I will provide an easy-to-understand answer based on the available information."
        ),
        "few_shot_examples": [
            "Q: What are some key facts about Mars? A: Here are some key facts about Mars based on the available data: [summarized facts from retrieved content].",
            "Q: Tell me something interesting about Jupiter. A: According to the data, Jupiter is known for [relevant fact from knowledge base]."
        ]
    }

    # Domain-specific settings
    DOMAIN_SETTINGS = {
        "domain": "Data Insights / Planetary Information",
        "rag": {
            "enabled": True,
            "retrieval_service": "azure_ai_search",
            "embedding_model": "text-embedding-ada-002",
            "top_k": 5,
            "search_type": "vector_semantic"
        },
        "fallback_response": (
            "I'm sorry, I couldn't find an answer to your question in the available data. Please try rephrasing your question or ask about a different topic."
        ),
        "required_config_keys": [
            "azure_ai_search_endpoint",
            "knowledge_base_index"
        ]
    }

    def __init__(self):
        self.env: Dict[str, Optional[str]] = {}
        self.llm_config: Dict[str, Any] = {}
        self.domain_settings: Dict[str, Any] = {}
        self._load_env()
        self._load_llm_config()
        self._load_domain_settings()
        self._validate()

    def _load_env(self):
        """Load required environment variables with error handling."""
        for key in self.REQUIRED_ENV_VARS:
            value = os.getenv(key)
            if not value:
                raise ConfigError(f"Missing required environment variable: {key}")
            self.env[key] = value

    def _load_llm_config(self):
        """Load LLM configuration, allowing for environment overrides."""
        self.llm_config = self.LLM_CONFIG_DEFAULTS.copy()
        # Allow override via environment variables if needed
        self.llm_config["model"] = os.getenv("LLM_MODEL", self.llm_config["model"])
        self.llm_config["temperature"] = float(os.getenv("LLM_TEMPERATURE", self.llm_config["temperature"]))
        self.llm_config["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", self.llm_config["max_tokens"]))
        self.llm_config["system_prompt"] = os.getenv("LLM_SYSTEM_PROMPT", self.llm_config["system_prompt"])
        self.llm_config["user_prompt_template"] = os.getenv("LLM_USER_PROMPT_TEMPLATE", self.llm_config["user_prompt_template"])

    def _load_domain_settings(self):
        """Load domain-specific settings."""
        self.domain_settings = self.DOMAIN_SETTINGS.copy()

    def _validate(self):
        """Validate all required configuration is present."""
        missing = [k for k in self.REQUIRED_ENV_VARS if not self.env.get(k)]
        if missing:
            raise ConfigError(f"Missing required environment variables: {missing}")

    def get_env(self, key: str) -> Optional[str]:
        """Get an environment variable value."""
        return self.env.get(key)

    def get_llm_config(self) -> Dict[str, Any]:
        """Get the LLM configuration."""
        return self.llm_config

    def get_domain_settings(self) -> Dict[str, Any]:
        """Get domain-specific settings."""
        return self.domain_settings

    def get_fallback_response(self) -> str:
        """Get the fallback response for unanswered questions."""
        return self.domain_settings.get("fallback_response", self.DOMAIN_SETTINGS["fallback_response"])

# Example usage:
# try:
#     config = AgentConfig()
#     llm_cfg = config.get_llm_config()
#     azure_search_endpoint = config.get_env("AZURE_SEARCH_ENDPOINT")
# except ConfigError as e:
#     print(f"Configuration error: {e}")
