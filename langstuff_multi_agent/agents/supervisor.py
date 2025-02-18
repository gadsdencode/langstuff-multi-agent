"""
Supervisor module with preprocessing.
"""

import logging
from typing import List, Literal, Dict, Any, TypedDict, Annotated
from pydantic.v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
import operator
from langstuff_multi_agent.config import get_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AVAILABLE_AGENTS = [
    'debugger', 'context_manager', 'project_manager', 'professional_coach',
    'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
    'news_reporter', 'customer_support', 'marketing_strategist',
    'creative_content', 'financial_analyst'
]

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]
    reasoning: str | None

class RouteDecision(BaseModel):
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst', 'FINISH'
    ] = Field(..., description="Target agent or FINISH")

def preprocess_input(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Convert Studio input format to BaseMessage list."""
    messages = state.get("messages", [])
    if not messages:
        raw_input = state.get("messages", []) or [{"type": "human", "content": "Hello"}]
        messages = []
        for msg in raw_input:
            role = msg.get("type", "human")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
    return {"messages": messages, "error_count": 0}

def supervisor_logic(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Core supervisor logic to route queries or finalize."""
    messages = state["messages"]
    if not messages:
        return {
            "next": "general_assistant",
            "error_count": 0,
            "messages": messages,
            "reasoning": "No messages provided, defaulting to general_assistant"
        }
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Simplified tool handling for testing
        return {"next": "general_assistant", "messages": messages, "error_count": 0, "reasoning": "Tool call detected"}

    options = AVAILABLE_AGENTS + ["FINISH"]
    system_prompt = (
        f"You manage these workers: {', '.join(AVAILABLE_AGENTS)}. "
        "Analyze the query and route to ONE specialized agent or FINISH.\n"
        "Rules:\n1. Route complex queries through multiple agents sequentially if needed.\n"
        "2. Use FINISH only when the task is fully resolved.\n"
        "3. On errors, route to general_assistant.\n"
        "Provide step-by-step reasoning and your decision."
    )
    structured_llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = structured_llm.invoke([SystemMessage(content=system_prompt), *messages])
        next_destination = decision.destination if decision.destination in options else "general_assistant"
        return {
            "next": next_destination,
            "reasoning": decision.reasoning,
            "error_count": 0,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Routing failed: {str(e)}")
        return {
            "next": "general_assistant",
            "error_count": 1,
            "messages": messages + [SystemMessage(content=f"Routing error: {str(e)}")],
            "reasoning": "Fallback to general_assistant due to routing failure"
        }

def create_supervisor(llm) -> StateGraph:
    """Create supervisor workflow with preprocessing."""
    workflow = StateGraph(SupervisorState)
    workflow.add_node("preprocess", preprocess_input)
    workflow.add_node("supervisor", supervisor_logic)
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"general_assistant": "general_assistant", "FINISH": END}
    )
    workflow.add_edge("preprocess", "supervisor")
    workflow.set_entry_point("preprocess")
    return workflow

supervisor_workflow = create_supervisor(llm=get_llm())

__all__ = ["create_supervisor", "supervisor_workflow", "SupervisorState"]