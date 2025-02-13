# langstuff_multi_agent/agent.py
"""
Main agent module that exports the graph for LangGraph Studio.

This module serves as the entry point for LangGraph Studio, exporting only
the primary supervisor workflow. This avoids potential MultipleSubgraphsError
by isolating internal subgraphs.
"""

import logging
from langstuff_multi_agent.agents.debugger import debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_graph
from langstuff_multi_agent.agents.project_manager import project_manager_graph
from langstuff_multi_agent.agents.professional_coach import (
    professional_coach_graph
)
from langstuff_multi_agent.agents.life_coach import life_coach_graph
from langstuff_multi_agent.agents.coder import coder_graph
from langstuff_multi_agent.agents.analyst import analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_graph
from langstuff_multi_agent.agents.general_assistant import (
    general_assistant_graph
)
from langstuff_multi_agent.agents.news_reporter import news_reporter_graph
from langstuff_multi_agent.agents.customer_support import (
    customer_support_graph
)
from langstuff_multi_agent.agents.marketing_strategist import (
    marketing_strategist_graph
)
from langstuff_multi_agent.agents.creative_content import (
    creative_content_graph
)
from langstuff_multi_agent.agents.financial_analyst import (
    financial_analyst_graph
)
import threading
from langstuff_multi_agent.agents.supervisor import create_supervisor
from langstuff_multi_agent.config import Config

config = Config()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory system
logger.info("Initializing memory system...")
Config.init_checkpointer()

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
        "general_assistant": general_assistant_graph,
        "news_reporter": news_reporter_graph,
        "customer_support": customer_support_graph,
        "marketing_strategist": marketing_strategist_graph,
        "creative_content": creative_content_graph,
        "financial_analyst": financial_analyst_graph
    }


# Replace manual supervisor setup with official pattern
supervisor_graph = create_supervisor(
    create_agent_graphs(),
    getattr(config, 'configurable', {}),
    supervisor_name="main_supervisor"
)

# Export all graphs required by langgraph.json
__all__ = [
    "supervisor_graph",  # Renamed from supervisor_workflow
    "debugger_graph",
    "context_manager_graph",
    "project_manager_graph",
    "professional_coach_graph",
    "life_coach_graph",
    "coder_graph",
    "analyst_graph",
    "researcher_graph",
    "general_assistant_graph",
    "news_reporter_graph",
    "customer_support_graph",
    "marketing_strategist_graph",
    "creative_content_graph",
    "financial_analyst_graph"
]

# Add explicit graph alias for entry point
graph = supervisor_graph
__all__.insert(0, "graph")  # Add to beginning of exports list

# Add monitoring after graph initialization
available_agents = [  # Define available agents list
    "debugger", "context_manager", "project_manager",
    "professional_coach", "life_coach", "coder",
    "analyst", "researcher", "general_assistant",
    "news_reporter", "customer_support", "marketing_strategist",
    "creative_content", "financial_analyst"
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


def handle_user_request(user_input: str, user_id: str):
    """Handle a user request with memory context.

    Args:
        user_input: The user's input message
        user_id: Unique identifier for the user (used for memory isolation)

    Returns:
        The supervisor graph's response
    """
    return supervisor_graph.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "user_id": user_id,  # For memory isolation
                "thread_id": user_id,  # For conversation threading
                "checkpointer": Config.PERSISTENT_CHECKPOINTER
            }
        }
    )
