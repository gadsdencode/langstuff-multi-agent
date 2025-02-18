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
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")

class GraphInput(BaseModel):
    """Input schema for the multi-agent graph."""
    messages: List[Dict[str, Any]] = Field(..., description="User messages to route")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration")

    def to_messages(self) -> List[BaseMessage]:
        """Convert input dictionaries to BaseMessage objects."""
        messages = []
        for msg in self.messages:
            role = msg.get("type", "human")  # Use 'type' from Studio
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))  # Fallback
        return messages

# Create and compile the supervisor graph
supervisor_graph = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs,
    input_type=GraphInput,
    state_type=SupervisorState
).compile(checkpointer=Config.checkpointer)

# Override stream to handle input conversion
def stream_with_conversion(input_data: Dict[str, Any], config: Dict[str, Any]):
    graph_input = GraphInput(**input_data)
    messages = graph_input.to_messages()
    return supervisor_graph.stream({"messages": messages}, config)

# Alias for entry point (Studio uses 'graph')
graph = supervisor_graph
graph.stream = stream_with_conversion  # Override default stream method

__all__ = ["graph", "supervisor_graph"] + list(member_graphs.keys())

logger.info("Primary supervisor workflow successfully initialized.")