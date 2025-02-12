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
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
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
from langstuff_multi_agent.agents.news_reporter import news_reporter_graph
from langstuff_multi_agent.agents.customer_support import customer_support_graph
from langstuff_multi_agent.agents.marketing_strategist import marketing_strategist_graph
from langstuff_multi_agent.agents.creative_content import creative_content_graph
from langstuff_multi_agent.agents.financial_analyst import financial_analyst_graph
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
    'general_assistant',
    'news_reporter',
    'customer_support',
    'marketing_strategist',
    'creative_content',
    'financial_analyst'
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
    - Debugger: Code errors solutions, troubleshooting
    - Coder: Writing/explaining code
    - Analyst: Data analysis requests
    - Researcher: Fact-finding, web research, news research
    - Project Manager: Task planning
    - Life Coach: Personal life strategies and advice
    - Professional Coach: Professional career strategies and advice
    - General Assistant: General purpose assistant for generic requests
    - News Reporter: News searching, reporting and summaries
    - Customer Support: Customer support queries
    - Marketing Strategist: Marketing strategy, insights, trends, and planning
    - Creative Content: Creative writing, marketing copy, social media posts, or brainstorming ideas
    - Financial Analyst: Financial analysis, market data, forecasting, and investment insights"""

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
    """Updated to preserve final assistant messages"""
    tool_outputs = []
    final_messages = []

    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_messages.append(msg)  # Capture final assistant response
        if tool_calls := getattr(msg, "tool_calls", None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {"messages": [ToolMessage(
                        goto=tc['name'].replace('transfer_to_', ''),
                        graph=ToolMessage.PARENT
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

    return {
        "messages": [
            *final_messages,  # Preserve final responses
            *[ToolMessage(content=to["output"], tool_call_id=to["tool_call_id"]) 
              for to in tool_outputs]
        ]
    }


def should_continue(state: dict) -> bool:
    """
    Determine if the workflow should continue processing.

    Returns True if there are pending tool calls or no final assistant message.
    """
    messages = state.get("messages", [])
    if not messages:
        return True

    last_message = messages[-1]
    # Continue if not an AI message or has tool calls
    return not isinstance(last_message, AIMessage) or bool(getattr(last_message, "tool_calls", None))


def end_state(state: RouterState):
    """Terminal node that returns the final state."""
    return state


# ======================
# Workflow Construction
# ======================
def create_supervisor(agent_graphs=None, configurable=None, supervisor_name=None):
    """Create supervisor workflow with enhanced configurability"""
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
    builder.add_node("news_reporter", news_reporter_graph)
    builder.add_node("customer_support", customer_support_graph)
    builder.add_node("marketing_strategist", marketing_strategist_graph)
    builder.add_node("creative_content", creative_content_graph)
    builder.add_node("financial_analyst", financial_analyst_graph)
    builder.add_node("process_results", process_tool_results)
    builder.add_node("end", end_state)  # Add terminal node

    # Conditional edges
    builder.add_conditional_edges(
        "route_query",
        lambda s: s.destination if s.destination in AVAILABLE_AGENTS else "general_assistant",
        {agent: agent for agent in AVAILABLE_AGENTS}
    )

    # Add conditional edge from process_results to either end or route_query
    builder.add_conditional_edges(
        "process_results",
        lambda state: "route_query" if should_continue(state) else "end",
        {"route_query": "route_query", "end": "end"}
    )

    builder.set_entry_point("route_query")
    return builder.compile()


supervisor_workflow = create_supervisor()

__all__ = ["create_supervisor", "supervisor_workflow"]
