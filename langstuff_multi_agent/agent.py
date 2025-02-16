# langstuff_multi_agent/agent.py
import logging
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
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage

config = Config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory system
logger.info("Initializing memory system...")
Config.init_checkpointer()

logger.info("Initializing primary supervisor workflow...")


# --- Input Schema Definition ---
class GraphInput(BaseModel):
    """Input for the multi-agent graph."""
    messages: List[Dict[str, Any]] = Field(
        ..., description="User messages to route"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for the graph, like user_id"
    )


# --- Supervisor State ---
class SupervisorState(TypedDict):
    messages: List[BaseMessage]
    next: str
    error_count: int
    reasoning: Optional[str]


# --- Create Agent Graphs ---
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


# --- Create Supervisor Graph ---
supervisor_graph = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs,
    input_type=GraphInput,  # Pass types directly to create_supervisor
    state_type=SupervisorState
)

# --- Exports and Aliases ---
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

# Add explicit graph alias for entry point
graph = supervisor_graph
__all__.insert(0, "graph")

logger.info("Primary supervisor workflow successfully initialized.")
