# my_agent/config.py
"""
Configuration settings for the LangGraph multi-agent AI project.

This file reads critical configuration values from environment variables,
provides default settings, and initializes logging. Adjust or extend this
configuration as needed for additional services or tools.
"""

import os
import logging


class Config:
    # REQUIRED: Anthropic API Key (for ChatAnthropic)
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        raise ValueError("Environment variable 'ANTHROPIC_API_KEY' is required but not set.")

    # Model settings
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-2")
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0))

    # Logging configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add additional configurations here (e.g., for other tools or databases) if needed.

    @classmethod
    def init_logging(cls):
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        logging.info("Logging initialized at level: %s", cls.LOG_LEVEL)


# Initialize logging as soon as this module is imported.
Config.init_logging()
