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
from langstuff_multi_agent.agents.debugger import debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_graph
from langstuff_multi_agent.agents.project_manager import project_manager_graph
from langstuff_multi_agent.agents.professional_coach import professional_coach_graph
from langstuff_multi_agent.agents.life_coach import life_coach_graph
from langstuff_multi_agent.agents.coder import coder_graph
from langstuff_multi_agent.agents.analyst import analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_graph
from langstuff_multi_agent.agents.general_assistant import general_assistant_graph
import threading
from langgraph_supervisor.handoff import create_supervisor
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.config import get_llm
from langstuff_multi_agent.config import Config

config = Config()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")


def create_agent_graphs():
    return {
        "debugger": debugger_graph,
        "context_manager": context_manager_graph,
        "project_manager": project_manager_graph,
        "professional_coach": professional_coach_graph,
        "life_coach": life_coach_graph,
        "coder": coder_graph,
        "analyst": analyst_graph,
        "researcher": researcher_graph,
        "general_assistant": general_assistant_graph
    }


# Replace manual supervisor setup with official pattern
supervisor_graph = create_supervisor(
    create_agent_graphs(),
    model=get_llm(config.get("configurable", {})),
    output_mode="last_message",
    supervisor_name="main_supervisor"
)

# Export all graphs required by langgraph.json
__all__ = [
    "supervisor_workflow",  # Main supervisor graph
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

# Add monitoring after graph initialization
available_agents = [  # Define available agents list
    "debugger", "context_manager", "project_manager",
    "professional_coach", "life_coach", "coder",
    "analyst", "researcher", "general_assistant"
]


def monitor_agents():
    """Prints agent statuses every 10 seconds"""
    import time
    while True:
        print("Active agents:", ", ".join(available_agents))
        time.sleep(10)


# Start monitoring thread
threading.Thread(target=monitor_agents, daemon=True).start()

logger.info("Primary supervisor workflow successfully initialized.")
