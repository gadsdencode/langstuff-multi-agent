# langstuff_multi_agent/agent.py
"""
Main agent module that exports the graph for LangGraph Studio.

This module serves as the entry point for LangGraph Studio, exporting only
the primary supervisor workflow. This avoids potential MultipleSubgraphsError
by isolating internal subgraphs.
"""

import logging
from langgraph.graph import Graph
from langstuff_multi_agent.agents.supervisor import supervisor_workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")

# Compile only the supervisor workflow as the entry point.
graph = supervisor_workflow.compile()

__all__ = ["graph"]

logger.info("Primary supervisor workflow successfully initialized.")
