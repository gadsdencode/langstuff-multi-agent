# agent.py
"""
Main agent module that exports the graph for LangGraph Studio.

This module serves as the entry point for LangGraph Studio, exporting both
the main supervisor workflow and all individual agent workflows.
"""

import logging
from typing import Dict, Any
from langgraph.graph import Graph

from langstuff_multi_agent.agents.supervisor import (
    supervisor_workflow,
    AGENT_OPTIONS,
    workflow_map
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing agent workflows...")

# Compile all workflows with their configurations
compiled_workflows: Dict[str, Graph] = {
    "supervisor": supervisor_workflow.compile(),
    **{
        name.lower(): workflow.compile()
        for name, workflow in workflow_map.items()
    }
}

# Export the main graph for LangGraph Studio
graph = compiled_workflows["supervisor"]

# Explicitly register individual agent graphs
for name, workflow in compiled_workflows.items():
    globals()[f"{name}_graph"] = workflow
    logger.info(f"Registered workflow: {name}_graph")

# Ensure metadata for all agents is accessible
agent_metadata: Dict[str, Dict[str, Any]] = {
    name.lower(): {
        "name": name,
        "description": desc,
        "graph": compiled_workflows.get(name.lower(), None)
    }
    for name, desc in AGENT_OPTIONS.items()
}

# Ensure all graphs are explicitly listed in __all__
__all__ = [
    "graph",  # Main supervisor graph
    *[f"{name.lower()}_graph" for name in compiled_workflows.keys()],  # Individual agent graphs
    "agent_metadata",  # Agent metadata
    "compiled_workflows"  # All compiled workflows
]

logger.info("Agent workflows successfully initialized.")
