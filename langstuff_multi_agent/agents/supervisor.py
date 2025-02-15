# supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
This refactored version follows the official LangGraph supervisor pattern,
returning Command objects for precise routing decisions.
"""

import re
import uuid
import logging
import operator
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolCall
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command  # Using Command for routing decisions
from langchain_core.language_models.chat_models import BaseChatModel
from langstuff_multi_agent.config import get_llm

# Import member graphs (including debugger if needed)
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

WHITESPACE_RE = re.compile(r"\s+")

def _normalize_agent_name(agent_name: str) -> str:
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()

def create_handoff_tool(*, agent_name: str) -> BaseTool:
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"
    @tool(tool_name)
    def handoff_to_agent(tool_call_id: str):
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

# Router schemas for structured output from the LLM
class RouterInput(BaseModel):
    messages: List[HumanMessage] = Field(..., description="User messages to route")
    last_route: Optional[str] = Field(None, description="Previous routing destination")

class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning"""
    reasoning: str = Field(..., description="Step-by-step routing logic")
    # The official pattern uses a destination (or next_agent) value
    destination: Literal[
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst', 'FINISH'
    ] = Field(..., description="Target agent or FINISH")

# Our extended router state includes our decision field
class RouterState(RouterInput):
    reasoning: Optional[str] = Field(None, description="Routing decision rationale")
    destination: Optional[str] = Field(None, description="Selected agent target")
    memories: List[str] = Field(default_factory=list, description="Relevant memory entries")

# Refactored supervisor logic using Command objects per official guidelines [&#8203;:contentReference[oaicite:2]{index=2}]
def _supervisor_logic(state: RouterState, llm: BaseChatModel) -> dict:
    # Check if the conversation is complete: last AIMessage marked as final and with no pending tool calls.
    if state.messages:
        last_msg = state.messages[-1]
        if (isinstance(last_msg, AIMessage) and 
            last_msg.additional_kwargs.get("final_answer", False) and 
            not getattr(last_msg, "tool_calls", [])):
            return Command(goto=END, update={}).dict()
    
    # Process any pending tool calls (if any)
    # This function processes tool calls and returns a Command dict if a handoff is needed.
    processed = process_tool_results(state, config={})
    if processed:
        return processed

    try:
        # If too many errors, fallback to a safe agent.
        if state.get("error_count", 0) > 2:
            return Command(goto="general_assistant", update={"error_count": 0}).dict()
        
        latest_message = state.messages[-1].content if state.messages else ""
        system_prompt = (
            "You are an expert router for a multi-agent system. Analyze the user's query "
            "and route to ONE specialized agent from the following: " + ", ".join(AVAILABLE_AGENTS)
        )
        decision = llm.with_structured_output(RouteDecision).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Route this query: {latest_message}")
        ])
        # Fallback for invalid agent selection
        next_agent = decision.destination if decision.destination in AVAILABLE_AGENTS else "general_assistant"
        if next_agent == "FINISH":
            return Command(goto=END, update={"next": "FINISH"}).dict()
        return Command(goto=next_agent, update={"reasoning": decision.reasoning, "next": next_agent}).dict()
    except Exception as e:
        logger.critical(f"Supervisor failure: {str(e)}")
        return Command(goto="general_assistant", update={"error_count": state.get("error_count", 0) + 1}).dict()

def process_tool_results(state: dict, config: dict) -> Optional[dict]:
    """Process any pending tool calls in the message chain and return a Command if needed."""
    tool_outputs = []
    final_messages = []
    if state.get("reasoning"):
        final_messages.append(AIMessage(content=f"Routing Reason: {state['reasoning']}"))
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            final_messages.append(msg)
        if tool_calls := getattr(msg, "tool_calls", None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    # Immediately handoff if a transfer tool call is detected.
                    return Command(goto=tc['name'].replace('transfer_to_', ''), update={}).dict()
                try:
                    output = f"Tool {tc['name']} result: {tc.get('output', '')}"
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": f"Error: {str(e)}"
                    })
    if tool_outputs:
        combined = final_messages + [ToolMessage(content=to["output"], tool_call_id=to["tool_call_id"]) for to in tool_outputs]
        # Return a Command that updates the messages with tool results.
        return Command(goto="supervisor", update={"messages": combined}).dict()
    return None

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

def create_supervisor(llm: BaseChatModel, members: List[str], member_graphs: dict, **kwargs) -> StateGraph:
    options = members + ["FINISH"]
    system_prompt = (
        "You manage these workers: " + ", ".join(members) + 
        ". Strict rules:\n1. Route complex queries through multiple agents sequentially.\n"
        "2. Return FINISH only when ALL user needs are met.\n"
        "3. On errors, route to general_assistant.\n"
        "4. Never repeat a failed agent immediately."
    )
    
    # Router model for structured output is already defined as RouteDecision.
    # Define a simple Pydantic model for routing if needed (omitted here for brevity).

    def _supervisor_wrapper(state: dict):
        # First, if the last message is a final answer with no pending tool calls, finish.
        if state.get("messages"):
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.additional_kwargs.get("final_answer", False) and not getattr(last_msg, "tool_calls", []):
                return {"next": END, "error_count": state.get("error_count", 0)}
            else:
                state = process_tool_results(state, config={}) or state
        # Use the refactored supervisor logic.
        return _supervisor_logic(state, llm)

    workflow = StateGraph(dict)  # Using a generic dict for state in this example.
    workflow.add_node("supervisor", _supervisor_wrapper)
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        )
    # Conditional edges: extract the routing decision from the state "next" key.
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next"),
        {member: member for member in members} | {"FINISH": END, "general_assistant": "general_assistant"}
    )
    for member in members:
        workflow.add_edge(member, "supervisor")
    workflow.set_entry_point("supervisor")
    return workflow

# Initialize member graphs
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

# Create the supervisor workflow
supervisor_workflow = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs
)

__all__ = ["create_supervisor", "supervisor_workflow"]
