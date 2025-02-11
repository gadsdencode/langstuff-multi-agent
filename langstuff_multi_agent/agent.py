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
from langstuff_multi_agent.agents.debugger import debugger_workflow as debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_workflow as context_manager_graph
from langstuff_multi_agent.agents.project_manager import project_manager_workflow as project_manager_graph
from langstuff_multi_agent.agents.professional_coach import professional_coach_workflow as professional_coach_graph
from langstuff_multi_agent.agents.life_coach import life_coach_workflow as life_coach_graph
from langstuff_multi_agent.agents.coder import coder_workflow as coder_graph
from langstuff_multi_agent.agents.analyst import analyst_workflow as analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_workflow as researcher_graph
from langstuff_multi_agent.agents.general_assistant import general_assistant_workflow as general_assistant_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")

# Compile only the supervisor workflow as the entry point.
graph = supervisor_workflow.compile()

# Export all graphs required by langgraph.json
__all__ = [
    "graph",  # Main supervisor graph
    "debugger_graph",
    "context_manager_graph",
    "project_manager_graph",
    "professional_coach_graph",
    "life_coach_graph",
    "coder_graph",
    "analyst_graph",
    "researcher_graph",
    "general_assistant_graph"
]

logger.info("Primary supervisor workflow successfully initialized.")
