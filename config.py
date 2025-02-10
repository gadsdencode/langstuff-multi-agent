# config.py
"""
Configuration settings for the LangGraph multi-agent AI project.

This file reads critical configuration values from environment variables,
provides default settings, initializes logging, sets up a persistent checkpoint
instance using MemorySaver, and exposes a factory function (get_llm) that returns

an LLM instance based on the chosen AI provider (anthropic, openai, or grok/xai).

Supported providers:
  - "anthropic": Uses ChatAnthropic with the key from ANTHROPIC_API_KEY.
  - "openai": Uses ChatOpenAI with the key from OPENAI_API_KEY.
  - "grok" (or "xai"): Uses ChatOpenAI as an interface to Grok with the key from XAI_API_KEY and a custom base URL.
"""

import os
import logging
from langgraph.checkpoint.memory import MemorySaver

# Import provider libraries.
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI


class Config:
    # REQUIRED: Anthropic API Key (for ChatAnthropic).
    # Also, ensure OPENAI_API_KEY and XAI_API_KEY are set for OpenAI and Grok respectively.
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        raise ValueError("Environment variable 'ANTHROPIC_API_KEY' is required but not set.")

    # Default model settings.
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "grok-2-1212")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0))

    # AI Provider: options are "anthropic", "openai", or "grok" (or "xai"). Default is "anthropic".
    AI_PROVIDER = os.environ.get("AI_PROVIDER", "xai").lower()

    # Logging configuration.
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Persistent checkpointer instance for use across workflows.
    PERSISTENT_CHECKPOINTER = MemorySaver()

    @classmethod
    def init_logging(cls):
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        logging.info("Logging initialized at level: %s", cls.LOG_LEVEL)


# Initialize logging immediately.
Config.init_logging()


def get_llm(model_name: str = None, temperature: float = None):
    """
    Returns an LLM instance based on the AI_PROVIDER configuration.

    This function supports three providers:
      - "anthropic": Uses ChatAnthropic.
      - "openai": Uses ChatOpenAI with GPT-4o-mini.
      - "grok" (or "xai"): Uses ChatOpenAI as an interface to Grok with a custom base URL.

    :param model_name: Optional model name override.
    :param temperature: Optional temperature override.
    :return: An LLM instance.
    :raises ValueError: if a required API key is missing or provider is unsupported.
    """
    # Use provided parameters or fallback to defaults.
    model_name = model_name or Config.DEFAULT_MODEL
    temperature = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE
    provider = Config.AI_PROVIDER

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        model = ChatOpenAI(
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            model_name="gpt-4o-mini",
            api_key=api_key
        )
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        model = ChatAnthropic(
            temperature=0,
            max_tokens=500,
            top_p=0.95,
            model_name=model_name,
            anthropic_api_key=api_key
        )
    elif provider in ("grok", "xai"):
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment")
        # Using ChatOpenAI as an interface to Grok with a custom base URL.
        model = ChatOpenAI(
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            model_name="grok-2-1212",
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

    return model
