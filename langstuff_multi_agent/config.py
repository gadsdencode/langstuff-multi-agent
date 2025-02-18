"""
Configuration for the LangGraph multi-agent AI project.
Handles LLM instantiation, checkpointer setup, and logging.
"""

import os
import logging
import json
from functools import lru_cache, wraps
from typing import Optional, Dict, Literal, Callable
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langstuff_multi_agent.utils.memory import MemoryManager, LangGraphMemoryCheckpointer

class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    provider: Literal["openai", "anthropic", "grok"]

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    XAI_API_KEY = os.environ.get("XAI_API_KEY")

    # Defaults
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0.4))
    DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()
    LLM_CACHE_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "8"))

    # Model configs
    MODEL_CONFIGS = {
        "anthropic": {"model_name": "claude-3-5-sonnet-20240620", "temperature": 0.0, "top_p": 0.9, "max_tokens": 4000},
        "openai": {"model_name": "gpt-4o-mini", "temperature": 0.4, "top_p": 0.9, "max_tokens": 4000},
        "grok": {"model_name": "grok-2-1212", "temperature": 0.4, "top_p": 0.9, "max_tokens": 4000}  # Grok model
    }

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Checkpointer
    MEMORY_MANAGER = MemoryManager()
    PERSISTENT_CHECKPOINTER = LangGraphMemoryCheckpointer(MEMORY_MANAGER)

    @classmethod
    def get_api_key(cls, provider: str) -> str:
        key_map = {"openai": cls.OPENAI_API_KEY, "anthropic": cls.ANTHROPIC_API_KEY, "grok": cls.XAI_API_KEY}
        key = key_map.get(provider)
        if not key:
            raise ValueError(f"{provider.upper()}_API_KEY not set")
        return key

class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic", "grok"]
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048

def get_model_instance(provider: str, **kwargs) -> BaseChatModel:
    config_data = {**Config.MODEL_CONFIGS[provider], **kwargs}
    config_obj = ModelConfig(provider=provider, **config_data)
    params = config_obj.dict(exclude={"provider"})
    if provider == "anthropic":
        return ChatAnthropic(api_key=Config.get_api_key("anthropic"), **params)
    elif provider in ["openai", "grok"]:
        return ChatOpenAI(api_key=Config.get_api_key(provider), **params)
    raise ValueError(f"Unsupported provider: {provider}")

@lru_cache(maxsize=Config.LLM_CACHE_SIZE)
def get_llm(configurable: Dict[str, Any] = {}) -> BaseChatModel:
    provider = configurable.get("provider", Config.DEFAULT_PROVIDER)
    model_kwargs = configurable.get("model_kwargs", {})
    llm = get_model_instance(provider, **model_kwargs)
    return llm

def create_model_config(model: Optional[str] = None, system_message: Optional[str] = None, **kwargs) -> RunnableConfig:
    validated = ConfigSchema(
        model=model or Config.DEFAULT_MODEL,
        system_message=system_message,
        temperature=kwargs.get("temperature", Config.DEFAULT_TEMPERATURE),
        provider=kwargs.get("provider", Config.DEFAULT_PROVIDER),
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_tokens", 4000)
    )
    return {"configurable": validated}

__all__ = ["Config", "get_llm", "create_model_config"]