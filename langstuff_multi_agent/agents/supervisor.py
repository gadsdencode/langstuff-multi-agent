# langstuff_multi_agent/supervisor.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage
)
from langstuff_multi_agent.config import get_llm
from typing import (
    Literal,
    Optional,
    List,
    TypedDict,
    Dict,
    Any,
    Type
)
import re
import uuid
from langchain_community.tools import tool
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolCallId
#from langstuff_multi_agent.utils.tools import search_memories #Removed as we don't have access to it.
import logging
import operator
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


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


def log_agent_failure(agent_name, query):
    logger.error(f"Agent '{agent_name}' failed to process query: {query}")


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_agent_name(agent_name: str) -> str:
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(*, agent_name: str) -> BaseTool:
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    @tool(tool_name)
    def handoff_to_agent(tool_call_id: Annotated[str, InjectedToolCallId]):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return {"messages": [tool_message], "next": agent_name}  # Directly route using "next"

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


class RouterInput(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="User messages to route") # Changed to list of Dict
    #last_route: Optional[str] = Field(None, description="Previous routing destination") # Removed


class RouteDecision(BaseModel):
    """Routing decision with chain-of-thought reasoning"""
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst', 'FINISH'
    ] = Field(..., description="Target agent")

#class RouterState(RouterInput): # No longer necessary, the main state is enough
#    reasoning: Optional[str] = Field(None, description="Routing decision rationale")
#    destination: Optional[str] = Field(None, description="Selected agent target")
#    memories: List[str] = Field(default_factory=list, description="Relevant memory entries")


class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]
    reasoning: Optional[str]  # Add reasoning to the state


def route_query(state: SupervisorState):
    """Route user query to appropriate agent."""
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": messages,
            "next": "general_assistant",
            "error_count": 0,
            "reasoning": "No messages to route"
        }

    # Get latest message content
    latest_message = ""
    if isinstance(messages[-1], HumanMessage):
        latest_message = messages[-1].content
    elif isinstance(messages[-1], dict):
        latest_message = messages[-1].get("content", "")

    # Get LLM with structured output
    llm = get_llm()
    system = (
        "You are an expert router for a multi-agent system. Analyze the user's query "
        "and route to ONE specialized agent from the following: Context Manager, Project Manager, "
        "Professional Coach, Life Coach, Coder, Analyst, Researcher, General Assistant, News Reporter, "
        "Customer Support, Marketing Strategist, Creative Content, Financial Analyst."
    )

    try:
        # Define structured output
        class RouteDecision(BaseModel):
            reasoning: str = Field(..., description="Step-by-step routing logic")
            destination: Literal[tuple(AVAILABLE_AGENTS)] = Field(..., description="Target agent")

        decision = llm.with_structured_output(RouteDecision).invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Route this query: {latest_message}")
        ])

        if decision.destination not in AVAILABLE_AGENTS:
            logger.error(f"Invalid agent {decision.destination} selected")
            return {
                "messages": messages,
                "next": "general_assistant",
                "reasoning": "Fallback due to invalid agent selection",
                "error_count": state.get("error_count", 0) + 1
            }

        return {
            "messages": messages,
            "reasoning": decision.reasoning,
            "next": decision.destination,
            "error_count": state.get("error_count", 0)
        }

    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
        return {
            "messages": messages,
            "next": "general_assistant",
            "reasoning": f"Error during routing: {str(e)}",
            "error_count": state.get("error_count", 0) + 1
        }


def process_tool_results(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Process tool results and update state."""
    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    if not (isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None)):
        return state

    # Add reasoning if present
    final_messages = messages[:-1]
    if state.get("reasoning"):
        final_messages.append(AIMessage(content=f"Routing Reason: {state['reasoning']}"))

    # Process tool calls
    tool_outputs = []
    for tc in last_message.tool_calls:
        try:
            # Handle transfer tools
            if tc["name"].startswith("transfer_to_"):
                agent = tc["name"].replace("transfer_to_", "")
                if agent in AVAILABLE_AGENTS:
                    return {
                        "messages": messages,
                        "next": agent,
                        "error_count": state.get("error_count", 0)
                    }
            
            # Mock other tool outputs
            output = f"Tool {tc['name']} executed successfully"
            tool_outputs.append({
                "tool_call_id": tc["id"],
                "output": output,
                "name": tc["name"]
            })

        except Exception as e:
            tool_outputs.append({
                "tool_call_id": tc["id"],
                "output": f"Error: {str(e)}",
                "name": tc["name"]
            })

    # Add tool messages
    for to in tool_outputs:
        final_messages.append(
            ToolMessage(
                content=to["output"],
                tool_call_id=to["tool_call_id"],
                name=to["name"]
            )
        )

    return {
        "messages": final_messages,
        "next": state.get("next", "general_assistant"),
        "error_count": state.get("error_count", 0)
    }


def should_continue(state: Dict[str, Any]) -> str:
    """Determine if processing should continue."""
    messages = state.get("messages", [])
    if not messages:
        return "continue"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        # Check for final answer flag
        if last_message.additional_kwargs.get("final_answer", False):
            if not getattr(last_message, "tool_calls", []):
                return "end"
        # Check for too many errors
        if state.get("error_count", 0) > 2:
            return "end"

    return "continue"


# def end_state(state: RouterState): # No longer needed
#    return state

def create_supervisor(
    llm: BaseChatModel,
    members: list[str],
    member_graphs: dict,
    input_type: Optional[type] = None,
    state_type: Optional[type] = None,
    **kwargs
) -> StateGraph:
    """Create supervisor workflow."""
    # Initialize graph
    workflow = StateGraph(state_type if state_type is not None else Dict[str, Any])
    
    # Add nodes
    workflow.add_node("supervisor", route_query)
    workflow.add_node("process_results", process_tool_results)
    
    # Add member nodes
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        )
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", END),
        {member: member for member in members} | {"END": END}
    )
    
    # Add edges from members back to supervisor
    for member in members:
        workflow.add_edge(member, "process_results")
        workflow.add_edge("process_results", "supervisor")
    
    if input_type is not None:
        workflow.input_type = input_type
        
    return workflow.compile()


# Initialize member graphs (including debugger if desired)

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
