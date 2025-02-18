"""
Entry point for the LangGraph multi-agent system.

This module initializes the supervisor graph, importing the supervisor and its member graphs,
and configures the system with a persistent checkpointer for state management.
"""

import logging
from langstuff_multi_agent.agents.supervisor import create_supervisor, member_graphs
from langstuff_multi_agent.config import Config, get_llm
from langstuff_multi_agent.utils.memory import SupervisorState
from pydantic.v1 import BaseModel, Field
from typing import List, Optional, Dict, Any

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")


class GraphInput(BaseModel):
    """Input schema for the multi-agent graph."""
    messages: List[Dict[str, Any]] = Field(..., description="User messages to route")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration")


# Create and compile the supervisor graph
supervisor_graph = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs,
    input_type=GraphInput,
    state_type=SupervisorState
).compile(checkpointer=Config.checkpointer)

# Alias for entry point
graph = supervisor_graph

# Export all relevant symbols
__all__ = ["graph", "supervisor_graph"] + list(member_graphs.keys())

logger.info("Primary supervisor workflow successfully initialized.")
