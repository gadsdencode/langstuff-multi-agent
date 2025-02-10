# agent.py
"""
Main entry point for the LangGraph Multi-Agent AI system.

This module exposes the compiled workflow for LangGraph Studio deployment.
The workflow is compiled with persistent checkpointing enabled via
Config.PERSISTENT_CHECKPOINTER.
"""

from langstuff_multi_agent.config import Config
from langstuff_multi_agent.agents.supervisor import supervisor_workflow

# Compile the workflow with persistent checkpointing for deployment
graph = supervisor_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER)
