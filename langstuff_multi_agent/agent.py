# langstuff_multi_agent/agent.py
"""
Main agent module that exports the graph for LangGraph Studio.
Now using an extended state schema with separate keys for messages, metadata, and state.
"""

import logging
import threading
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langstuff_multi_agent.agents.debugger import debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_graph
from langstuff_multi_agent.agents.project_manager import project_manager_graph
from langstuff_multi_agent.agents.professional_coach import professional_coach_graph
from langstuff_multi_agent.agents.life_coach import life_coach_graph
from langstuff_multi_agent.agents.coder import coder_graph
from langstuff_multi_agent.agents.analyst import analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_graph
from langstuff_multi_agent.agents.general_assistant import general_assistant_graph
from langstuff_multi_agent.agents.news_reporter import news_reporter_graph
from langstuff_multi_agent.agents.customer_support import customer_support_graph
from langstuff_multi_agent.agents.marketing_strategist import marketing_strategist_graph
from langstuff_multi_agent.agents.creative_content import creative_content_graph
from langstuff_multi_agent.agents.financial_analyst import financial_analyst_graph
from langstuff_multi_agent.agents.supervisor import create_supervisor, member_graphs
from langstuff_multi_agent.config import Config, get_llm
from typing_extensions import TypedDict, Annotated

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# New unified state schema: GraphState

class GraphState(TypedDict):
    # conversation messages – using the add_messages reducer for append-only behavior
    messages: Annotated[list, add_messages]
    # extra metadata (e.g., for logging, UI fields)
    metadata: Dict[str, Any]
    # additional persistent state (arbitrary)
    state: Dict[str, Any]

# Initialize memory system and checkpointer as before
logger.info("Initializing memory system...")
Config.init_checkpointer()

logger.info("Initializing primary supervisor workflow...")

# (No changes to agent graph creation; we simply export the graphs as before)
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
        "financial_analyst": financial_analyst_graph,
    }

# Create the primary supervisor graph using official pattern
supervisor_graph = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs
)

# Export all graphs required by langgraph.json
__all__ = [
    "supervisor_graph",
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

# Define alias and export primary graph entry point
graph = supervisor_graph
__all__.insert(0, "graph")

# Monitoring thread remains unchanged
available_agents = [
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

threading.Thread(target=monitor_agents, daemon=True).start()

logger.info("Primary supervisor workflow successfully initialized.")

def handle_user_request(user_input: str, user_id: str, metadata: Dict[str, Any] = None, state_data: Dict[str, Any] = None):
    """
    Handle a user request.
    Now accepts extra 'metadata' and 'state' inputs.
    """
    input_state: GraphState = {
        "messages": [HumanMessage(content=user_input)],
        "metadata": metadata or {},
        "state": state_data or {}
    }
    return supervisor_graph.invoke(
        input_state,
        config={
            "configurable": {
                "user_id": user_id,
                "thread_id": user_id,
                "checkpointer": Config.PERSISTENT_CHECKPOINTER
            }
        }
    )
