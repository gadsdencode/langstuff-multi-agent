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
import json
from functools import lru_cache, wraps
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional, Dict, Any, Literal, Callable
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models import (
    chat_models as base_chat_models
)
from langstuff_multi_agent.utils.memory import (
    LangGraphMemoryCheckpointer,
    MemoryManager
)


class ConfigSchema(TypedDict):
    """Enhanced configuration schema for assistant nodes"""
    model: Optional[str]
    system_message: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    provider: Literal['openai', 'anthropic', 'grok']  # Required provider field


class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    XAI_API_KEY = os.environ.get("XAI_API_KEY")

    # Default model settings
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0.4))
    DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()

    # Cache settings
    LLM_CACHE_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "8"))

    # Track API key states for cache invalidation
    _api_key_states = {
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
        "grok": XAI_API_KEY
    }

    @classmethod
    def api_keys_changed(cls) -> bool:
        """Check if any API keys have changed since last check."""
        current_states = {
            "anthropic": cls.ANTHROPIC_API_KEY,
            "openai": cls.OPENAI_API_KEY,
            "grok": cls.XAI_API_KEY
        }
        changed = current_states != cls._api_key_states
        cls._api_key_states = current_states.copy()
        return changed

    # Model configurations
    MODEL_CONFIGS = {
        "anthropic": {
            "model_name": "claude-3-5-sonnet-20240620",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 4000,
        },
        "openai": {
            "model_name": "gpt-4o-mini",  # Preferred openai model
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 4000,
        },
        "grok": {
            "model_name": "grok-2-1212",  # Fallback to latest Grok model - this model name is accurate
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 4000,
        }
    }

    # Logging configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Initialize memory system
    MEMORY_MANAGER = MemoryManager(persist_path="memory_store")
    PERSISTENT_CHECKPOINTER: Optional[LangGraphMemoryCheckpointer] = None

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

    @classmethod
    def init_checkpointer(cls) -> None:
        """Initialize the persistent checkpointer with memory manager."""
        if cls.PERSISTENT_CHECKPOINTER is None:
            cls.PERSISTENT_CHECKPOINTER = LangGraphMemoryCheckpointer(
                cls.MEMORY_MANAGER
            )
        logging.info("Memory checkpointer initialized")


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


class LLMCacheStats:
    """Tracks statistics for the LLM cache."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def __str__(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return (
            f"Cache Stats - Hits: {self.hits}, Misses: {self.misses}, "
            f"Hit Rate: {hit_rate:.1f}%, Invalidations: {self.invalidations}"
        )


# Global cache statistics
llm_cache_stats = LLMCacheStats()


def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string, handling edge cases."""
    try:
        return json.dumps(obj, sort_keys=True)
    except (TypeError, ValueError, OverflowError) as e:
        logging.warning(f"JSON serialization failed: {e}. Using str representation.")
        return str(obj)


@lru_cache(maxsize=None)  # Size controlled by wrapper
def _get_cached_llm_inner(provider: str, model_kwargs_json: str):
    """Internal cached function for LLM instantiation."""
    model_kwargs = json.loads(model_kwargs_json)
    return get_model_instance(provider, **model_kwargs)


def _cache_wrapper(func: Callable) -> Callable:
    """Wrapper to add cache statistics and invalidation."""
    cache = lru_cache(maxsize=Config.LLM_CACHE_SIZE)(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for API key changes
        if Config.api_keys_changed():
            logging.info("API keys changed - invalidating LLM cache")
            cache.cache_clear()
            llm_cache_stats.invalidations += 1

        # Track cache statistics
        result = cache(*args, **kwargs)
        if hasattr(cache, 'cache_info'):
            info = cache.cache_info()
            llm_cache_stats.hits = info.hits
            llm_cache_stats.misses = info.misses

        return result

    # Expose cache clear method
    wrapper.cache_clear = cache.cache_clear
    return wrapper


@_cache_wrapper
def _get_cached_llm(provider: str, model_kwargs_json: str):
    """
    Helper function that creates and caches LLM instances based on configuration.
    Uses JSON string of model_kwargs as a hashable cache key.

    Args:
        provider: The LLM provider name
        model_kwargs_json: JSON string of model configuration parameters

    Returns:
        Cached LLM instance
    """
    try:
        return _get_cached_llm_inner(provider, model_kwargs_json)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode model_kwargs_json: {e}")
        raise ValueError(f"Invalid model configuration JSON: {e}")


def get_llm(configurable: dict = {}) -> base_chat_models.BaseChatModel:
    """
    Factory function to create a language model instance based on configuration.
    Uses caching to avoid re-instantiating the LLM if the configuration hasn't changed.

    The cache size can be configured via the LLM_CACHE_SIZE environment variable.
    Cache statistics are available via the llm_cache_stats global variable.

    Args:
        configurable: Optional configuration dictionary that can include:
               - provider: Provider name ("anthropic", "openai", "grok", etc.)
               - model_kwargs: Additional keyword arguments for the model

    Returns:
        A cached instance of BaseChatModel configured according to the specified parameters.

    Raises:
        ValueError: If the configuration is invalid or JSON serialization fails
    """
    provider = configurable.get('provider', 'openai')
    model_kwargs = configurable.get('model_kwargs', {})

    try:
        # Convert model_kwargs to a JSON string for hashing
        model_kwargs_json = safe_json_dumps(model_kwargs)
        llm = _get_cached_llm(provider, model_kwargs_json)

        # Add memory context to all LLM calls
        return llm.with_config(
            {"checkpointer": Config.PERSISTENT_CHECKPOINTER}
        )
    except Exception as e:
        logging.error(f"Failed to create LLM instance: {e}")
        raise ValueError(f"Failed to create LLM instance: {e}")


def create_model_config(
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    **kwargs
) -> RunnableConfig:
    """
    Updated config creator with validation

    :param model: Model identifier (e.g. "gpt-4o")
    :param system_message: Role definition for the assistant
    :param **kwargs: Additional config parameters
    """
    validated = ConfigSchema(
        model=model or Config.DEFAULT_MODEL,
        system_message=system_message,
        temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
        provider=kwargs.get('provider', Config.DEFAULT_PROVIDER),
        top_p=kwargs.get('top_p', 0.1),
        max_tokens=kwargs.get('max_tokens', 4000)
    )
    return {"configurable": validated}
