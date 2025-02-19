"""
Supervisor module for managing a hierarchical multi-agent system.
"""

import logging
from typing import List, Literal, Dict, Any, TypedDict, Annotated
from pydantic.v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
import operator
from langstuff_multi_agent.config import get_llm

# Import the personal_assistant graph
from langstuff_multi_agent.agents.personal_assistant import personal_assistant_graph

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define available agents (only personal_assistant)
AVAILABLE_AGENTS = [
    'personal_assistant'
]

# Map agent names to their graphs
member_graphs = {
    "personal_assistant": personal_assistant_graph,
}

# Define the state structure for the supervisor
class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]
    reasoning: str | None

# Define the routing decision model
class RouteDecision(BaseModel):
    reasoning: str = Field(..., description="Step-by-step routing logic")
    destination: Literal[
        'personal_assistant', 'FINISH'
    ] = Field(..., description="Target agent or FINISH")

# Helper function to convert messages to BaseMessage objects
def convert_message(msg):
    if isinstance(msg, dict):
        msg_type = msg.get("type")
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant":
            return AIMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls", []))
        elif msg_type == "system":
            return SystemMessage(content=msg.get("content", ""))
        elif msg_type == "tool":
            return ToolMessage(content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", ""), name=msg.get("name", ""))
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    return msg

def convert_messages(messages):
    return [convert_message(msg) for msg in messages]

# Preprocess input messages
def preprocess_input(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Converts raw input into a list of BaseMessage objects.
    """
    messages = convert_messages(state.get("messages", []))
    if not messages:
        raw_input = state.get("messages", []) or [{"type": "human", "content": "Hello"}]
        messages = convert_messages(raw_input)
    logger.info(f"Preprocess output: {messages}")
    return {"messages": messages, "error_count": 0}

# Supervisor routing logic
def supervisor_logic(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines the next agent to route to based on the current state.
    """
    state["messages"] = convert_messages(state["messages"])
    messages = state["messages"]
    if not messages:
        return {
            "next": "personal_assistant",
            "error_count": 0,
            "messages": messages,
            "reasoning": "No messages provided, routing to personal_assistant"
        }
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("final_answer", False):
        return {
            "next": "FINISH",
            "error_count": state.get("error_count", 0),
            "messages": messages,
            "reasoning": "Agent marked response as final"
        }

    system_prompt = (
        f"You manage this worker: {', '.join(AVAILABLE_AGENTS)}. "
        "Analyze the query and route to the personal_assistant or FINISH if the task is fully resolved.\n"
        "Rules:\n"
        "1. Route queries to the personal_assistant.\n"
        "2. Use FINISH only when the personal_assistant has provided a complete response (marked as final_answer).\n"
        "3. For any queries, including greetings or identity questions, route to personal_assistant.\n"
        "4. On errors or uncertainty, route to personal_assistant.\n"
        "Provide step-by-step reasoning and your decision."
    )
    structured_llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = structured_llm.invoke([SystemMessage(content=system_prompt)] + messages)
        next_destination = decision.destination if decision.destination in AVAILABLE_AGENTS + ["FINISH"] else "personal_assistant"
        return {
            "next": next_destination,
            "reasoning": decision.reasoning,
            "error_count": state.get("error_count", 0),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Routing failed: {str(e)}")
        return {
            "next": "personal_assistant",
            "error_count": state.get("error_count", 0) + 1,
            "messages": messages + [SystemMessage(content=f"Routing error: {str(e)}")],
            "reasoning": "Fallback to personal_assistant due to routing failure"
        }

# Create the supervisor workflow
def create_supervisor(llm) -> StateGraph:
    """
    Sets up the StateGraph with all nodes and edges.
    """
    workflow = StateGraph(SupervisorState)
    workflow.add_node("preprocess", preprocess_input)
    workflow.add_node("supervisor", supervisor_logic)

    added_agents = []
    for name in AVAILABLE_AGENTS:
        try:
            subgraph = member_graphs[name]
            def make_subgraph_node(subgraph):
                def subgraph_node(state: SupervisorState, config: RunnableConfig) -> SupervisorState:
                    state["messages"] = convert_messages(state["messages"])
                    subgraph_state = {"messages": state["messages"]}
                    try:
                        # Handle both compiled graphs and raw StateGraph objects
                        config_dict = config.dict() if hasattr(config, 'dict') else config
                        result = subgraph.invoke(subgraph_state, config_dict)
                    except Exception as e:
                        logger.error(f"Subgraph execution failed: {str(e)}")
                        state["messages"].append(SystemMessage(content=f"Agent error: {str(e)}"))
                        return state
                    state["messages"] = convert_messages(result["messages"])
                    return state
                return subgraph_node

            specific_subgraph_node = make_subgraph_node(subgraph)
            workflow.add_node(name, specific_subgraph_node)
            workflow.add_edge(name, "supervisor")
            added_agents.append(name)
            logger.info(f"Successfully added node: {name}")
        except Exception as e:
            logger.error(f"Failed to add node {name}: {str(e)}", exc_info=True)

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {name: name for name in added_agents} | {"FINISH": END}
    )
    workflow.add_edge("preprocess", "supervisor")
    workflow.set_entry_point("preprocess")
    return workflow

# Instantiate the supervisor workflow
supervisor_workflow = create_supervisor(llm=get_llm())

__all__ = ["create_supervisor", "supervisor_workflow", "SupervisorState", "member_graphs"]