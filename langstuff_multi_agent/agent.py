import logging
from langstuff_multi_agent.agents import *  # Import all agent graphs
from langstuff_multi_agent.config import Config, get_llm
from langstuff_multi_agent.agents.supervisor import create_supervisor, member_graphs
from pydantic.v1 import BaseModel, Field
from typing import List, Optional, Dict, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph.message import add_messages

config = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Config.init_checkpointer()
logger.info("Initializing primary supervisor workflow...")


class GraphInput(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="User messages to route")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration")


class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]
    reasoning: Optional[str]


supervisor_graph = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs,
    input_type=GraphInput,
    state_type=SupervisorState
).compile(checkpointer=Config.checkpointer)

graph = supervisor_graph
__all__ = ["graph", "supervisor_graph"] + list(member_graphs.keys())

logger.info("Primary supervisor workflow successfully initialized.")
