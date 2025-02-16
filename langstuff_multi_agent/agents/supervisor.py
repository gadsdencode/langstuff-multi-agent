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
    structured_llm = get_llm().bind_tools([RouteDecision])
    #latest_message = state["messages"][-1].content if state["messages"] else "" # Extract the content
    latest_message = ""
    if state["messages"]:
        if isinstance(state["messages"][-1], HumanMessage):
            latest_message = state["messages"][-1].content
        elif isinstance(state["messages"][-1], dict):
            latest_message = state["messages"][-1].get("content", "")

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
        return {
            "messages": state["messages"],
            "next": "general_assistant",
            "reasoning": "Fallback due to invalid agent selection",  # added reasoning to the state
            "error_count": state.get("error_count", 0)
        }
    else:
        return {
            "messages": state["messages"],
            "reasoning": decision.reasoning,  # Added reasoning
            "next": decision.destination,
            "error_count": state.get("error_count", 0)
           # "memories": state.get("memories", []) #Removed memories
        }


def process_tool_results(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Process any pending tool calls in the message chain."""
    messages = state["messages"]
    if not messages:
        return {"messages": []}

    last_message = messages[-1]

    # If there are no tool calls, just return the messages as is.
    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        return {"messages": messages}

    tool_calls = last_message.tool_calls
    tool_outputs = []
    final_messages = messages[:-1]  # Start with all but last message

    if "reasoning" in state:  # Adds the reasoning
        final_messages.append(AIMessage(content=f"Routing Reason: {state['reasoning']}"))

    for tc in tool_calls:

        if tc['name'].startswith('transfer_to_'):
            agent_name = tc['name'].replace('transfer_to_', '')
            return {"messages": messages, "next": agent_name}  # Use "next" for routing

        try:
            # In a real application, you would execute the tool here.
            # For this example, we just mock the output.
            output = f"Tool {tc['name']} result: (Mocked output)"
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

    # Add ToolMessages for each tool output
    for to in tool_outputs:
        final_messages.append(ToolMessage(content=to["output"], tool_call_id=to["tool_call_id"], name = to["name"]))

    return {"messages": final_messages}


def should_continue(state: Dict[str, Any]) -> str:
    """Determines if the graph should continue based on the last message."""
    messages = state.get("messages", [])
    if not messages:
        return "continue"  # Changed to return strings

    last_message = messages[-1]
    # Only consider the conversation final if the last message is an AI message
    # marked as a final answer and has no pending tool calls.
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("final_answer", False):
        if not getattr(last_message, "tool_calls", []):
            return "end"  # Changed to return strings

    return "continue"  # Changed to return strings


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
    options = members + ["FINISH"]  # Used for runtime validation
    system_prompt = f"""You manage these workers: {', '.join(members)}. Strict rules:
1. Route complex queries through multiple agents sequentially.
2. Return FINISH only when ALL user needs are met.
3. On errors, route to general_assistant.
4. Never repeat a failed agent immediately.
"""

    class Router(BaseModel):
        next: Literal[
            'debugger', 'context_manager', 'project_manager', 'professional_coach',
            'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
            'news_reporter', 'customer_support', 'marketing_strategist',
            'creative_content', 'financial_analyst', 'FINISH'
        ]

    def _supervisor_logic(state: Dict[str, Any]):
        if state["messages"]:
            last_msg = state["messages"][-1]

            if isinstance(last_msg, AIMessage) and last_msg.additional_kwargs.get("final_answer", False):
                if not getattr(last_msg, "tool_calls", []):
                    return {"next": "FINISH", "error_count": state.get("error_count", 0)}
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", []):
                return process_tool_results(state, config={})
        if should_continue(state) == "continue":
            try:
                if state.get("error_count", 0) > 2:
                    return {"next": "general_assistant", "error_count": 0, "messages": state["messages"]}
                result = llm.with_structured_output(Router).invoke([
                    SystemMessage(content=system_prompt),
                    *state["messages"]
                ])
                # Validate the next destination is in our options
                if result.next not in options:
                    logger.error(f"Invalid destination {result.next}")
                    return {"next": "general_assistant", "error_count": state.get("error_count", 0) + 1, "messages": state["messages"]}
                return {"next": result.next, "error_count": state.get("error_count", 0), "messages": state["messages"]}
            except Exception as e:
                logger.critical(f"Supervisor failure: {str(e)}")
                return {"next": "general_assistant", "error_count": state.get("error_count", 0) + 1, "messages": state["messages"]}
        else:
            return {"next": "FINISH"}

    # Use provided state_type or default to Dict[str, Any]
    workflow = StateGraph(state_type if state_type is not None else Dict[str, Any])
    workflow.add_node("supervisor", _supervisor_logic)
    
    for name in members:
        workflow.add_node(
            name,
            member_graphs[name].with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        )
    
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "continue": lambda x: x["next"] if x["next"] != "FINISH" else END,
            "end": END
        }
    )
    
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    workflow.set_entry_point("supervisor")
    
    # If input_type is provided, configure it
    if input_type is not None:
        workflow.input_type = input_type
    
    return workflow


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
