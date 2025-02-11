# config.py
"""
Configuration settings for the LangGraph multi-agent AI project.

This file reads critical configuration values from environment variables,
provides default settings, initializes logging, sets up a persistent checkpoint
instance using MemorySaver, and exposes configuration and factory functions for LangGraph.

Supported providers:
  - "anthropic": Uses ChatAnthropic with the key from ANTHROPIC_API_KEY.
  - "openai": Uses ChatOpenAI with the key from OPENAI_API_KEY.
  - "grok" (or "xai"): Uses ChatOpenAI as an interface to Grok with the key from XAI_API_KEY.

Note: The LLM instances returned by get_llm() support structured output via the 
.with_structured_output() method. This is essential for our supervisor routing
logic and agent structured responses.
"""

import os
import logging
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ValidationError

# Import provider libraries.
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class ConfigSchema(TypedDict):
    """Schema for LangGraph runtime configuration."""
    model: Optional[str]
    system_message: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]


class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    XAI_API_KEY = os.environ.get("XAI_API_KEY")

    # Default model settings
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0))
    DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()

    # Model configurations
    MODEL_CONFIGS = {
        "anthropic": {
            "model_name": "claude-3-5-sonnet-20240620",
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 4000,
        },
        "openai": {
            "model_name": "gpt-4o",  # Preferred openai model
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 4000,
        },
        "grok": {
            "model_name": "grok-2-1212",  # Fallback to latest Grok model - this model name is accurate
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 4000,
        }
    }

    # Logging configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Persistent checkpointer instance
    PERSISTENT_CHECKPOINTER = MemorySaver()

    @classmethod
    def init_logging(cls):
        """Initialize logging with configured settings."""
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        logging.info("Logging initialized at level: %s", cls.LOG_LEVEL)

    @classmethod
    def get_api_key(cls, provider: str) -> str:
        """Get the API key for the specified provider."""
        key_map = {
            "openai": ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            "anthropic": ("ANTHROPIC_API_KEY", cls.ANTHROPIC_API_KEY),
            "grok": ("XAI_API_KEY", cls.XAI_API_KEY),
        }

        env_var, key = key_map.get(provider, (None, None))
        if not key:
            raise ValueError(f"{env_var} environment variable not set")
        return key


# Initialize logging immediately
Config.init_logging()


class ModelConfig(BaseModel):
    """Validation schema for LLM configurations"""
    provider: Literal['openai', 'anthropic', 'grok', 'azure_openai']
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    structured_output_method: Optional[str] = None


def get_model_instance(provider: str, **kwargs):
    # Validate provider first
    if not provider or provider not in Config.MODEL_CONFIGS:
        available = list(Config.MODEL_CONFIGS.keys())
        raise ValueError(f"Invalid LLM provider: {provider}. Available: {available}")

    try:
        # Validate configuration against schema
        config_data = {**Config.MODEL_CONFIGS[provider], **kwargs}
        config_obj = ModelConfig(
            provider=provider,
            **config_data
        )
    except ValidationError as e:
        error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        raise ValueError(
            "Invalid model configuration:" + "\n" + "\n".join(error_messages)
        )

    # Exclude structured_output_method from model params
    model_params = config_obj.model_dump(exclude={'provider', 'structured_output_method'})

    logging.info("Creating model instance for provider: %s with params: %s",
                 provider, model_params)

    if provider == "anthropic":
        return ChatAnthropic(
            api_key=Config.get_api_key("anthropic"),
            **model_params
        )
    elif provider in ["openai", "grok"]:
        return ChatOpenAI(
            api_key=Config.get_api_key(provider),
            **model_params
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_llm(configurable: dict = {}):
    """
    Factory function to create a language model instance based on configuration.

    Args:
        configurable: Optional configuration dictionary that can include:
               - provider: Provider name ("anthropic", "openai", "grok", etc.)
               - system_message: Optional system message to prepend
               - temperature: Temperature parameter for generation
               - top_p: Top-p parameter for generation
               - max_tokens: Maximum tokens to generate
               - model_kwargs: Additional keyword arguments for the model (e.g., structured_output_method)

    Returns:
        An instance of BaseChatModel configured according to the specified parameters.
        Note: The returned LLM instance supports structured output via .with_structured_output().
    """
    provider = configurable.get('provider', 'openai')  # Set default provider
    model_kwargs = configurable.get('model_kwargs', {})
    return get_model_instance(provider, **model_kwargs)


def create_model_config(
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    **kwargs
) -> RunnableConfig:
    """
    Create a RunnableConfig for LangGraph workflow configuration.

    Args:
        model: Optional model provider to use.
        system_message: Optional system message to prepend.
        **kwargs: Additional configuration parameters.

    Returns:
        RunnableConfig with the specified configuration.
    """
    config = {"model": model} if model else {}
    if system_message:
        config["system_message"] = system_message
    config.update(kwargs)
    return {"configurable": config}
