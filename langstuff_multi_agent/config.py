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
from typing import Optional

# Import provider libraries.
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class Config:
    # REQUIRED: Anthropic API Key (for ChatAnthropic).
    # Also, ensure OPENAI_API_KEY and XAI_API_KEY are set for OpenAI and Grok respectively.
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        raise ValueError("Environment variable 'ANTHROPIC_API_KEY' is required but not set.")

    # Default model settings.
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0))


    # AI Provider: options are "anthropic", "openai", or "grok" (or "xai"). Default is "anthropic".
    AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()


    # Logging configuration.
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Persistent checkpointer instance for use across workflows.
    PERSISTENT_CHECKPOINTER = MemorySaver()

    # Model settings
    MODEL_NAME = "gpt-4o-mini"  # Using GPT-4o-mini for best tool use support
    TEMPERATURE = 0.0
    TOP_P = 0.1
    MAX_TOKENS = 4000  # Increased for better response handling



    @classmethod
    def init_logging(cls):
        """Initialize logging with configured settings."""
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        logging.info("Logging initialized at level: %s", cls.LOG_LEVEL)

    @classmethod
    def get_api_key(cls) -> str:
        """Get the OpenAI API key from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key


# Initialize logging immediately.
Config.init_logging()


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """
    Factory function to create a GPT-4o-mini language model instance.
    
    Args:
        model_name: Name of the model to use (defaults to GPT-4o-mini)
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        max_tokens: Maximum tokens to generate
        

    Returns:
        An instance of BaseChatModel configured for GPT-4o-mini
    """
    return ChatOpenAI(
        model_name=model_name or Config.MODEL_NAME,
        temperature=temperature or Config.TEMPERATURE,
        top_p=top_p or Config.TOP_P,
        max_tokens=max_tokens or Config.MAX_TOKENS,
        api_key=Config.get_api_key()
    )
