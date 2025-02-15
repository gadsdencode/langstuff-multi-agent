# langstuff_multi_agent/agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
Updated to use extended state with 'metadata' and 'state' keys.
All state updates from nodes now propagate these fields unchanged.
"""

import logging
import re
import uuid
import operator
from typing import List, Optional, Literal, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage, ToolCall
from langchain_core.tools import BaseTool
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolCallId
from langstuff_multi_agent.config import get_llm
from langgraph.graph.message import add_messages, convert_to_messages, add_messages as _add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langstuff_multi_agent.agents.project_manager import project_manager_graph
from langstuff_multi_agent.agents.financial_analyst import financial_analyst_graph
from langstuff_multi_agent.agents.coder import coder_graph
from langstuff_multi_agent.agents.general_assistant import general_assistant_graph
from langstuff_multi_agent.agents.news_reporter import news_reporter_graph
from langstuff_multi_agent.agents.customer_support import customer_support_graph
from langstuff_multi_agent.agents.marketing_strategist import marketing_strategist_graph
from langstuff_multi_agent.agents.creative_content import creative_content_graph
from langstuff_multi_agent.agents.life_coach import life_coach_graph
from langstuff_multi_agent.agents.professional_coach import professional_coach_graph
from langstuff_multi_agent.agents.analyst import analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_graph
from langstuff_multi_agent.agents.debugger import debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_graph
from langchain_community.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define AVAILABLE_AGENTS constant (unchanged)
AVAILABLE_AGENTS = [
    'debugger', 'context_manager', 'project_manager', 'professional_coach',
    'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
    'news_reporter', 'customer_support', 'marketing_strategist',
    'creative_content', 'financial_analyst'
]

def log_agent_failure(agent_name, query):
    logger.error(f"Agent '{agent_name}' failed to process query: {query}")

WHITESPACE_RE = re.compile(r"\s+")

def _normalize_agent_name(agent_name: str) -> str:
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()

def create_handoff_tool(*, agent_name: str) -> BaseTool:
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"
      # ensure tool decorator available

    @tool(tool_name)
    def handoff_to_agent(tool_call_id: Annotated[str, InjectedToolCallId]):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return ToolMessage(
            goto=agent_name,
            graph=ToolMessage.PARENT,
            update={"messages": [tool_message]},
        )
    return handoff_to_agent

def create_handoff_back_messages(agent_name: str, supervisor_name: str):
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        ),
    )

# Extended RouterState now includes optional metadata and state fields.
class RouterInput(BaseModel):
    messages: List[HumanMessage] = Field(..., description="User messages to route")
    last_route: Optional[str] = Field(None, description="Previous routing destination")

class RouterState(RouterInput):
    reasoning: Optional[str] = Field(None, description="Routing decision rationale")
    destination: Optional[str] = Field(None, description="Selected agent target")
    memories: List[str] = Field(default_factory=list, description="Relevant memory entries")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    state: Optional[dict] = Field(None, description="Additional persistent state")

class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning"""
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst'
    ] = Field(..., description="Target agent")

def route_query(state: RouterState):
    structured_llm = get_llm().bind_tools([RouteDecision])
    latest_message = state.messages[-1].content if state.messages else ""
    system = (
        "You are an expert router for a multi-agent system. Analyze the user's query "
        "and route to ONE specialized agent from the following: Context Manager, Project Manager, "
        "Professional Coach, Life Coach, Coder, Analyst, Researcher, General Assistant, News Reporter, "
        "Customer Support, Marketing Strategist, Creative Content, Financial Analyst."
    )
    decision = structured_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Route this query: {latest_message}")
    ])
    if decision.destination not in AVAILABLE_AGENTS:
        logger.error(f"Invalid agent {decision.destination} selected")
        return RouterState(
            messages=state.messages,
            destination="general_assistant",
            reasoning="Fallback due to invalid agent selection",
            metadata=state.metadata,
            state=state.state,
        )
    else:
        return RouterState(
            messages=state.messages,
            reasoning=decision.reasoning,
            destination=decision.destination,
            memories=state.memories,
            metadata=state.metadata,
            state=state.state,
        )

def process_tool_results(state, config):
    """Process tool outputs and route correctly"""
    messages = state.get("messages", [])
    metadata = state.get("metadata", {})
    persistent_state = state.get("state", {})
    
    # Critical fix: Properly handle transfer tool calls
    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, 'goto'):
            return {
                "messages": [msg],
                "metadata": metadata,
                "state": persistent_state,
                "next": msg.goto
            }
    
    # Existing processing logic...
    tool_outputs = []
    for msg in messages:
        if isinstance(msg, AIMessage) and (tool_calls := getattr(msg, "tool_calls", None)):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    agent_name = tc['name'].replace('transfer_to_', '')
                    return {
                        "messages": [ToolMessage(
                            content=f"Transferring to {agent_name}",
                            name=tc['name'],
                            tool_call_id=tc['id'],
                            goto=agent_name
                        )],
                        "metadata": metadata,
                        "state": persistent_state,
                        "next": agent_name
                    }

    # Return updated state while preserving metadata and state
    return {
        "messages": messages + [
            ToolMessage(content=to["output"], tool_call_id=to["tool_call_id"])
            for to in tool_outputs
        ],
        "metadata": metadata,
        "state": persistent_state
    }

def should_continue(state: dict) -> bool:
    messages = state.get("messages", [])
    if not messages:
        return True
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("final_answer", False):
        if not getattr(last_message, "tool_calls", []):
            return False
    return True

def end_state(state: RouterState):
    return state

# SupervisorState now includes metadata and state updates.
class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    error_count: int
    metadata: Dict[str, Any]
    state: Dict[str, Any]

def create_supervisor(llm, members: list[str], member_graphs: dict, **kwargs) -> StateGraph:
    options = members + ["FINISH"]
    system_prompt = (
        f"You manage these workers: {', '.join(members)}. Strict rules:\n"
        "1. Route complex queries through multiple agents sequentially.\n"
        "2. Return FINISH only when ALL user needs are met.\n"
        "3. On errors, route to general_assistant.\n"
        "4. Never repeat a failed agent immediately."
    )

    class Router(BaseModel):
        next: Literal[*options]  # type: ignore

    def _supervisor_logic(state: SupervisorState):
        # Add debug logging with type checking
        logger.debug(f"Supervisor state type: {type(state.get('messages', [])[-1]) if state.get('messages') else 'No messages'}")
        
        if state.get("messages"):
            last_msg = state["messages"][-1]
            # Handle both Message instances and raw dictionaries
            if isinstance(last_msg, (ToolMessage, dict)) and 'goto' in (last_msg if isinstance(last_msg, dict) else last_msg.__dict__):
                goto_target = last_msg['goto'] if isinstance(last_msg, dict) else last_msg.goto
                logger.info(f"Transferring to {goto_target}")
                return {
                    "next": goto_target,
                    "error_count": 0,
                    "metadata": state.get("metadata", {}),
                    "state": state.get("state", {})
                }
        
        # Fix message content extraction
        try:
            last_message = state["messages"][-1]
            message_content = last_message.content if isinstance(last_message, HumanMessage) else last_message.get('content', '')
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Message access error: {str(e)}")
            message_content = ""

        # Rest of the logic with proper content handling
        try:
            result = llm.with_structured_output(Router).invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ])
            if result.next not in members:
                logger.warning(f"Invalid agent {result.next} selected, using fallback")
                result.next = "general_assistant"
            
            return {
                **result.dict(),
                "error_count": state.get("error_count", 0),
                "metadata": state.get("metadata", {}),
                "state": state.get("state", {})
            }
        except Exception as e:
            logger.critical(f"Supervisor failure: {str(e)}")
            return {"next": "general_assistant", "error_count": state.get("error_count", 0) + 1,
                    "metadata": state.get("metadata", {}), "state": state.get("state", {})}

    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", _supervisor_logic)
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        )
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: "general_assistant" if state.get("error_count", 0) > 0 else state["next"],
        {member: member for member in members} | {"FINISH": END, "general_assistant": "general_assistant"}
    )
    for member in members:
        workflow.add_edge(member, "supervisor")
    workflow.set_entry_point("supervisor")
    return workflow


# Initialize member graphs (unchanged)
member_graphs = {
    "project_manager": project_manager_graph,
    "financial_analyst": financial_analyst_graph,
    "coder": coder_graph,
    "general_assistant": general_assistant_graph,
    "news_reporter": news_reporter_graph,
    "customer_support": customer_support_graph,
    "marketing_strategist": marketing_strategist_graph,
    "creative_content": creative_content_graph,
    "life_coach": life_coach_graph,
    "professional_coach": professional_coach_graph,
    "analyst": analyst_graph,
    "researcher": researcher_graph,
    "debugger": debugger_graph,
    "context_manager": context_manager_graph
}

supervisor_workflow = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs
)

__all__ = ["create_supervisor", "supervisor_workflow"]
