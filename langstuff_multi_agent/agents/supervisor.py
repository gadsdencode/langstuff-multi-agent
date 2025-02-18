"""
Minimal supervisor module for testing.
"""

import logging
from typing import Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import operator
from langstuff_multi_agent.config import get_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add, add_messages]

def supervisor_logic(state: SupervisorState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simple supervisor that returns a response."""
    messages = state["messages"]
    if not messages:
        return {"messages": [AIMessage(content="No input provided.")]}
    response = get_llm().invoke(messages)
    return {"messages": [response]}

def create_supervisor(llm) -> StateGraph:
    """Create a minimal supervisor workflow."""
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_logic)
    workflow.add_edge("supervisor", END)
    workflow.set_entry_point("supervisor")
    return workflow

supervisor_workflow = create_supervisor(llm=get_llm())

__all__ = ["create_supervisor", "supervisor_workflow", "SupervisorState"]