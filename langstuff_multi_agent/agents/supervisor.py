# langstuff_multi_agent/agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
"""

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langstuff_multi_agent.config import get_llm
from typing import Literal, Optional
from pydantic import BaseModel, Field
import re
import uuid
from langchain_community.tools import tool
from langchain_core.tools import BaseTool, ToolCall
from langchain.schema import Command
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolCallId

# Import individual workflows.
from langstuff_multi_agent.agents.debugger import debugger_graph
from langstuff_multi_agent.agents.context_manager import context_manager_graph
from langstuff_multi_agent.agents.project_manager import project_manager_graph
from langstuff_multi_agent.agents.professional_coach import professional_coach_graph
from langstuff_multi_agent.agents.life_coach import life_coach_graph
from langstuff_multi_agent.agents.coder import coder_graph
from langstuff_multi_agent.agents.analyst import analyst_graph
from langstuff_multi_agent.agents.researcher import researcher_graph
from langstuff_multi_agent.agents.general_assistant import general_assistant_graph

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define valid agent destinations at the top of the file
AVAILABLE_AGENTS = [
    'debugger',
    'context_manager',
    'project_manager',
    'professional_coach',
    'life_coach',
    'coder',
    'analyst',
    'researcher',
    'general_assistant'
]


def log_agent_failure(agent_name, query):
    """Logs agent failures for better debugging"""
    logger.error(f"Agent '{agent_name}' failed to process query: {query}")


# ======================
# Handoff Implementation
# ======================
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_agent_name(agent_name: str) -> str:
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(*, agent_name: str) -> BaseTool:
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    @tool(tool_name)
    def handoff_to_agent(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
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


# ======================
# Core Supervisor Logic
# ======================
class RouterInput(BaseModel):
    messages: list[HumanMessage] = Field(..., description="User messages to route")
    last_route: Optional[str] = Field(None, description="Previous routing destination")


class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning"""
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[tuple(AVAILABLE_AGENTS)] = Field(..., description="Target agent")


class RouterState(RouterInput):
    """Combined state for routing workflow"""
    reasoning: Optional[str] = Field(None, description="Routing decision rationale")
    destination: Optional[str] = Field(None, description="Selected agent target")


def route_query(state: RouterState):
    """Classifies and routes user queries using structured LLM output."""
    # Get config from state and add structured output method
    config = getattr(state, "configurable", {})
    config["structured_output_method"] = "json_mode"
    llm = get_llm(config)
    system = """You are an expert router for a multi-agent system. Analyze the user's query 
    and route to ONE specialized agent. Consider these specialties:
    - Debugger: Code errors, troubleshooting
    - Coder: Writing/explaining code
    - Analyst: Data analysis requests
    - Researcher: Fact-finding, web research
    - Project Manager: Task planning
    - Life Coach: Personal life strategies and advice
    - Professional Coach: Professional career strategies and advice
    - General Assistant: General purpose assistant for generic requests"""

    structured_llm = llm.with_structured_output(RouteDecision)

    decision = structured_llm.invoke([{
        "role": "user",
        "content": f"Route this query: {state.messages[-1].content}"
    }], config={"system": system})

    # Use the defined constant for validation
    if decision.destination not in AVAILABLE_AGENTS:
        log_agent_failure(decision.destination, state.messages[-1].content)
        return RouterState(
            messages=state.messages,
            reasoning="Fallback due to failure",
            destination="general_assistant"
        )
    else:
        return RouterState(
            messages=state.messages,
            reasoning=decision.reasoning,
            destination=decision.destination
        )


def process_tool_results(state, config):
    """Updated to handle command routing"""
    tool_outputs = []
    for msg in state["messages"]:
        if tool_calls := getattr(msg, "tool_calls", None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {"messages": [Command(
                        goto=tc['name'].replace('transfer_to_', ''),
                        graph=Command.PARENT
                    )]}
                # Existing tool processing logic
                try:
                    output = f"Tool {tc['name']} result: {tc['output']}"
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "error": str(e)
                    })

    return {"messages": [ToolMessage(
        content=to["output"],
        tool_call_id=to["tool_call_id"]
    ) for to in tool_outputs]}


# ======================
# Workflow Construction
# ======================
def create_supervisor_workflow():
    builder = StateGraph(RouterState)

    # Add nodes using imported compiled graphs
    builder.add_node("route_query", route_query)
    builder.add_node("debugger", debugger_graph)
    builder.add_node("context_manager", context_manager_graph)
    builder.add_node("project_manager", project_manager_graph)
    builder.add_node("professional_coach", professional_coach_graph)
    builder.add_node("life_coach", life_coach_graph)
    builder.add_node("coder", coder_graph)
    builder.add_node("analyst", analyst_graph)
    builder.add_node("researcher", researcher_graph)
    builder.add_node("general_assistant", general_assistant_graph)

    # Conditional edges
    builder.add_conditional_edges(
        "route_query",
        lambda s: s.destination if s.destination in AVAILABLE_AGENTS else "general_assistant",
        {agent: agent for agent in AVAILABLE_AGENTS}
    )

    # Tool processing edge
    builder.add_edge("process_results", "route_query")

    builder.set_entry_point("route_query")
    return builder.compile()


supervisor_workflow = create_supervisor_workflow()

__all__ = ["supervisor_workflow"]
