# supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.

Key changes from previous version:
  - We no longer manually reference START. Instead, we call workflow.set_entry_point("collect_input"),
    which designates 'collect_input' as the entry node. This prevents the "START cannot be an end node" error.
  - 'collect_input' is the node where you provide {"user_input": "..."} in the LangGraph Studio "State".
  - The graph then transitions to 'supervisor', which uses Command objects for precise routing.
"""

import re
import uuid
import logging
import operator
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
)
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import ToolCall
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel
from langstuff_multi_agent.config import get_llm

# Import member graphs
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
    """
    Creates a specialized tool that, if invoked by the LLM, returns a ToolMessage
    instructing the graph to route to the given agent_name.
    """
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
    """
    Utility that returns two messages (AIMessage + ToolMessage) for handing
    control back to supervisor_name from agent_name.
    """
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


# ---------------------------------------------------------
# Pydantic classes for structured LLM routing decisions
# ---------------------------------------------------------
class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning."""
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst', 'FINISH'
    ] = Field(..., description="Target agent or FINISH")


class RouterState(BaseModel):
    """
    State object that the supervisor will read/write.

    Example:
      {
        "messages": [...],
        "reasoning": null,
        "destination": null,
        "memories": [],
        "error_count": 0
      }
    """
    messages: List[BaseMessage] = Field(default_factory=list)
    reasoning: Optional[str] = None
    destination: Optional[str] = None
    memories: List[str] = Field(default_factory=list)
    error_count: int = 0


# ---------------------------------------------------------
# Implementation details
# ---------------------------------------------------------
def process_tool_results(state: Dict[str, Any], config: Dict[str, Any]) -> Optional[dict]:
    """
    Process any pending tool calls in the message chain. If any calls exist,
    we return a Command that updates the messages with tool results or transfers control.
    """
    tool_outputs = []
    final_messages = []

    # If there's an existing "reasoning" field, add it as an AI message (optional)
    if state.get("reasoning"):
        final_messages.append(AIMessage(content=f"Routing Reason: {state['reasoning']}"))

    for msg in state["messages"]:
        # Keep normal AI messages that have no tool calls
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            final_messages.append(msg)

        # If the message has tool calls, process them
        if tool_calls := getattr(msg, "tool_calls", None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    # Immediately route if there's a "transfer_to_X" call
                    next_agent = tc['name'].replace('transfer_to_', '')
                    return Command(goto=next_agent, update={}).dict()
                # Otherwise store the "tool result" for each call
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
        # Incorporate the tool results as ToolMessages
        combined = final_messages + [
            ToolMessage(content=to["output"], tool_call_id=to["tool_call_id"])
            for to in tool_outputs
        ]
        # Return a Command that updates the messages
        return Command(goto="supervisor", update={"messages": combined}).dict()

    return None


def _supervisor_logic(state: RouterState, llm: BaseChatModel) -> dict:
    """
    The main supervisor logic. It checks if conversation is done,
    processes tool calls, or uses the LLM to decide the next agent.
    """
    # 1) If conversation is complete (final answer with no pending tool calls)
    if state.messages:
        last_msg = state.messages[-1]
        if (isinstance(last_msg, AIMessage)
                and last_msg.additional_kwargs.get("final_answer", False)
                and not getattr(last_msg, "tool_calls", [])):
            return Command(goto=END, update={}).dict()

    # 2) Check any pending tool calls
    processed = process_tool_results(state, config={})
    if processed:
        return processed

    # 3) If too many errors, fallback to general_assistant
    if state.error_count > 2:
        return Command(goto="general_assistant", update={"error_count": 0}).dict()

    # 4) Use the LLM to pick the next agent
    try:
        latest_message_text = ""
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            latest_message_text = state.messages[-1].content

        system_prompt = (
            "You are an expert router for a multi-agent system. Analyze the user's query "
            f"and route to ONE specialized agent from: {', '.join(AVAILABLE_AGENTS)}. "
            "If the query is fully answered, you may route to FINISH."
        )

        decision = llm.with_structured_output(RouteDecision).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Route this query: {latest_message_text}")
        ])

        next_agent = decision.destination
        # fallback
        if next_agent not in AVAILABLE_AGENTS and next_agent != "FINISH":
            next_agent = "general_assistant"
        if next_agent == "FINISH":
            return Command(goto=END, update={"next": "FINISH"}).dict()

        return Command(
            goto=next_agent,
            update={
                "reasoning": decision.reasoning,
                "next": next_agent
            }
        ).dict()

    except Exception as e:
        logger.critical(f"Supervisor failure: {str(e)}")
        return Command(
            goto="general_assistant",
            update={"error_count": state.error_count + 1}
        ).dict()


def collect_input_node(user_input_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that collects user_input from the "State" in LangGraph Studio.
    The user can provide: { "user_input": "Hello" }
    This node transforms that into a RouterState with messages=[HumanMessage(...)].
    """
    user_text = user_input_state.get("user_input", "")
    # Build initial RouterState
    # - The "messages" start with a single HumanMessage
    # - "error_count" starts at 0
    # - "reasoning" / "destination" / "memories" are empty
    new_state = {
        "messages": [HumanMessage(content=user_text)],
        "error_count": 0,
        "reasoning": None,
        "destination": None,
        "memories": []
    }
    return new_state


def create_supervisor(llm: BaseChatModel,
                      members: List[str],
                      member_graphs: dict,
                      **kwargs) -> StateGraph:
    """
    Build a StateGraph with:
      - 'collect_input' node for user input
      - 'supervisor' node for multi-agent routing
      - subgraph nodes for each specialized agent
    """
    workflow = StateGraph(dict)

    # Node: collect_input
    workflow.add_node("collect_input", collect_input_node)

    # Node: supervisor
    def _supervisor_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        # Convert raw dict to RouterState, then call _supervisor_logic
        router_state = RouterState(**state)
        return _supervisor_logic(router_state, llm)

    workflow.add_node("supervisor", _supervisor_wrapper)

    # Add each specialized agent subgraph
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        )

    # Set 'collect_input' as the entry point
    workflow.set_entry_point("collect_input")

    # collect_input -> supervisor
    workflow.add_edge("collect_input", "supervisor")

    # supervisor uses "next" field to decide next agent or FINISH
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next"),
        {m: m for m in members} | {"FINISH": END, "general_assistant": "general_assistant"}
    )

    # Each agent returns to supervisor after it finishes
    for member in members:
        workflow.add_edge(member, "supervisor")

    return workflow


# Subgraphs for each specialized agent
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

# Instantiate the workflow
supervisor_workflow = create_supervisor(
    llm=get_llm(),
    members=list(member_graphs.keys()),
    member_graphs=member_graphs
)

__all__ = ["create_supervisor", "supervisor_workflow"]
