# langstuff_multi_agent/agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langstuff_multi_agent.config import get_llm
from typing import Literal, Optional, List, TypedDict
from pydantic import BaseModel, Field
import re
import uuid
from langchain_community.tools import tool
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolCallId
from langstuff_multi_agent.utils.tools import search_memories
import logging
import operator
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

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
from langstuff_multi_agent.config import get_llm

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
    destination: Literal[
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
    ] = Field(..., description="Target agent")


class RouterState(RouterInput):
    """Combined state for routing workflow"""
    reasoning: Optional[str] = Field(None, description="Routing decision rationale")
    destination: Optional[str] = Field(None, description="Selected agent target")
    memories: List[str] = Field(default_factory=list, description="Relevant memory entries")


def route_query(state: RouterState):
    """Original routing logic with complete system message"""
    structured_llm = get_llm().bind_tools([RouteDecision])  # <-- INIT HERE

    latest_message = state.messages[-1].content if state.messages else ""

    # FULL ORIGINAL SYSTEM PROMPT
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

    # ORIGINAL INVOCATION PATTERN
    decision = structured_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Route this query: {latest_message}")
    ])

    # ORIGINAL VALIDATION LOGIC
    if decision.destination not in AVAILABLE_AGENTS:
        logger.error(f"Invalid agent {decision.destination} selected")
        return RouterState(
            messages=state.messages,
            destination="general_assistant",
            reasoning="Fallback due to invalid agent selection"
        )
    else:
        return RouterState(
            messages=state.messages,
            reasoning=decision.reasoning,
            destination=decision.destination,
            memories=state.memories
        )


def process_tool_results(state, config):
    """Updated to preserve final assistant messages and show reasoning"""
    tool_outputs = []
    final_messages = []

    # Add reasoning to messages if present
    if state.get("reasoning"):
        final_messages.append(AIMessage(content=f"Routing Reason: {state['reasoning']}"))

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
                # Ensure every tool call gets a response
                try:
                    output = f"Tool {tc['name']} result: {tc.get('output', '')}"
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": f"Error: {str(e)}"  # Still provide a response on error
                    })

    return {
        "messages": [
            *final_messages,  # Preserve final responses
            *[ToolMessage(
                content=to["output"],
                tool_call_id=to["tool_call_id"]
            ) for to in tool_outputs]
        ]
    }


def should_continue(state: dict) -> bool:
    """Determine if the workflow should continue processing."""
    # Handle both dict and Pydantic model cases
    messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
    if not messages:
        return True

    last_message = messages[-1]
    return not isinstance(last_message, AIMessage) or bool(getattr(last_message, "tool_calls", None))


def end_state(state: RouterState):
    """Terminal node that returns the final state."""
    return state


# ======================
# Workflow Construction
# ======================
class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]  # Track consecutive errors


def create_supervisor(
    llm: BaseChatModel,
    members: list[str],
    member_graphs: dict,
    **kwargs  # Add to accept unexpected arguments
) -> StateGraph:
    options = members + ["FINISH"]

    # Production-grade system prompt from LangChain docs
    system_prompt = """You manage these workers: {members}. Strict rules:
1. Route complex queries through multiple agents sequentially
2. Return FINISH only when ALL user needs are met
3. On errors, route to debugger then original agent
4. Never repeat failed agent immediately""".format(members=", ".join(members))

    class Router(BaseModel):
        next: Literal[*options]  # type: ignore

    def _supervisor_logic(state: SupervisorState):
        try:
            # Error recovery logic
            if state.get("error_count", 0) > 2:
                return {"next": "debugger", "error_count": 0}

            return llm.with_structured_output(Router).invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=state["messages"][-1].content)
                ]
            ).dict()
        except Exception as e:
            logger.critical(f"Supervisor failure: {str(e)}")
            return {"next": "general_assistant", "error_count": state.get("error_count", 0) + 1}

    # 3. Full workflow construction
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", _supervisor_logic)

    # Add member graphs with error boundaries
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(
                stop_after_attempt=2,
                wait_exponential_jitter=True
            )
        )

    # 4. Conditional edges with error routing
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: "debugger" if state.get("error_count", 0) > 0 else state["next"],
        {member: member for member in members} | {
            "FINISH": END,
            "debugger": "debugger"
        }
    )

    # 5. Complete circular workflow
    for member in members:
        workflow.add_edge(member, "supervisor")

    workflow.set_entry_point("supervisor")
    return workflow


# 6. Proper initialization with member graphs (REQUIRED)
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
