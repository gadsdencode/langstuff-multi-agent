"""
Supervisor module for managing a hierarchical multi-agent system.

This module defines the supervisor workflow that routes user queries to specialized agents and coordinates their execution until the task is complete.
"""

import logging
from typing import List, Literal, Optional, Dict, Any, TypedDict, Annotated
from pydantic.v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
import operator
from langstuff_multi_agent.config import get_llm

# Import all agent graphs
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AVAILABLE_AGENTS = [
    'debugger', 'context_manager', 'project_manager', 'professional_coach',
    'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
    'news_reporter', 'customer_support', 'marketing_strategist',
    'creative_content', 'financial_analyst'
]

member_graphs = {
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

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]
    reasoning: Optional[str]

class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning"""
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
        # Handle Studio input format: [{"type": "human", "content": "..."}]
        raw_input = state.get("messages", []) or [{"type": "human", "content": "Hello"}]
        messages = []
        for msg in raw_input:
            role = msg.get("type", "human")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))  # Fallback
    return {"messages": messages, "error_count": 0}

def process_tool_results(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Process any pending tool calls in the message chain."""
    messages = state["messages"]
    if not messages:
        return {"messages": messages, "next": "general_assistant", "error_count": 0}
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": messages}

    tool_messages = []
    next_destination = state.get("next", "supervisor")
    for tc in last_message.tool_calls:
        if tc["name"].startswith("transfer_to_"):
            agent_name = tc["name"].replace("transfer_to_", "")
            next_destination = agent_name
            tool_messages.append(ToolMessage(
                content=f"Transferred to {agent_name}",
                tool_call_id=tc["id"],
                name=tc["name"]
            ))
        else:
            output = f"Tool {tc['name']} result: (Mocked output)"
            tool_messages.append(ToolMessage(
                content=output,
                tool_call_id=tc["id"],
                name=tc["name"]
            ))
    return {
        "messages": messages + tool_messages,
        "next": next_destination,
        "error_count": state.get("error_count", 0),
        "reasoning": state.get("reasoning", None)
    }

def supervisor_logic(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Core supervisor logic to route queries or finalize the task."""
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
        return process_tool_results(state, config)
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("final_answer", False):
        return {
            "next": "FINISH",
            "error_count": state.get("error_count", 0),
            "messages": messages,
            "reasoning": "Agent marked response as final"
        }

    options = AVAILABLE_AGENTS + ["FINISH"]
    system_prompt = (
        f"You manage these workers: {', '.join(AVAILABLE_AGENTS)}. "
        "Analyze the query and route to ONE specialized agent or FINISH if all needs are met.\n"
        "Rules:\n"
        "1. Route complex queries through multiple agents sequentially if needed.\n"
        "2. Use FINISH only when the task is fully resolved.\n"
        "3. On errors or uncertainty, route to general_assistant.\n"
        "Provide step-by-step reasoning and your decision."
    )
    structured_llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = structured_llm.invoke([SystemMessage(content=system_prompt), *messages])
        next_destination = decision.destination if decision.destination in options else "general_assistant"
        error_increment = 1 if decision.destination not in options else 0
        return {
            "next": next_destination,
            "reasoning": decision.reasoning,
            "error_count": state.get("error_count", 0) + error_increment,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Routing failed: {str(e)}")
        return {
            "next": "general_assistant",
            "error_count": state.get("error_count", 0) + 1,
            "messages": messages + [SystemMessage(content=f"Routing error: {str(e)}")],
            "reasoning": "Fallback to general_assistant due to routing failure"
        }

def create_supervisor(
    llm: BaseChatModel,
    members: List[str],
    member_graphs: Dict[str, StateGraph],
    state_type: Optional[type] = None
) -> StateGraph:
    """Create and configure the supervisor workflow."""
    workflow = StateGraph(state_type or SupervisorState)
    workflow.add_node("preprocess", preprocess_input)  # Added preprocessing node
    workflow.add_node("supervisor", supervisor_logic)
    for name in members:
        if name not in member_graphs:
            logger.error(f"Member {name} not found in member_graphs")
            continue
        workflow.add_node(name, member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True))
    workflow.add_edge("preprocess", "supervisor")  # Connect preprocess to supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {name: name for name in members} | {"FINISH": END}
    )
    for member in members:
        workflow.add_edge(member, "supervisor")
    workflow.set_entry_point("preprocess")  # Start at preprocess
    return workflow

supervisor_workflow = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs
)

__all__ = ["create_supervisor", "supervisor_workflow", "SupervisorState", "member_graphs"]