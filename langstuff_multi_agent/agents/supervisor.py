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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define all available agents
AVAILABLE_AGENTS = [
    'debugger', 'context_manager', 'project_manager', 'professional_coach',
    'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
    'news_reporter', 'customer_support', 'marketing_strategist',
    'creative_content', 'financial_analyst'
]

# Map agent names to their respective graphs
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
        'debugger', 'context_manager', 'project_manager', 'professional_coach',
        'life_coach', 'coder', 'analyst', 'researcher', 'general_assistant',
        'news_reporter', 'customer_support', 'marketing_strategist',
        'creative_content', 'financial_analyst', 'FINISH'
    ] = Field(..., description="Target agent or FINISH")

# Preprocess input messages
def preprocess_input(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Converts raw input into a list of BaseMessage objects.
    """
    messages = state.get("messages", [])
    if not messages:
        raw_input = state.get("messages", []) or [{"type": "human", "content": "Hello"}]
        messages = []
        for msg in raw_input:
            if isinstance(msg, dict):
                role = msg.get("type", "human")
                content = msg.get("content", "")
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(BaseMessage(content=content, type=role))
            elif isinstance(msg, BaseMessage):
                messages.append(msg)
    logger.info(f"Preprocess output: {messages}")
    return {"messages": messages, "error_count": 0}

# Supervisor routing logic
def supervisor_logic(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines the next agent to route to based on the current state.
    """
    messages = state["messages"]
    if not messages:
        return {
            "next": "general_assistant",
            "error_count": 0,
            "messages": messages,
            "reasoning": "No messages provided, defaulting to general_assistant"
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
        f"You manage these workers: {', '.join(AVAILABLE_AGENTS)}. "
        "Analyze the query and route to ONE specialized agent or FINISH if the task is fully resolved.\n"
        "Rules:\n"
        "1. Route complex queries through multiple agents sequentially if needed.\n"
        "2. Use FINISH only when an agent has provided a complete response (marked as final_answer).\n"
        "3. For greetings, identity questions (e.g., 'who are you'), or vague/general queries, route to general_assistant.\n"
        "4. On errors or uncertainty, route to general_assistant.\n"
        "Provide step-by-step reasoning and your decision."
    )
    structured_llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = structured_llm.invoke([SystemMessage(content=system_prompt), *messages])
        next_destination = decision.destination if decision.destination in AVAILABLE_AGENTS + ["FINISH"] else "general_assistant"
        return {
            "next": next_destination,
            "reasoning": decision.reasoning,
            "error_count": state.get("error_count", 0),
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

# Create the supervisor workflow
def create_supervisor(llm) -> StateGraph:
    """
    Sets up the StateGraph with all nodes and edges.
    """
    workflow = StateGraph(SupervisorState)
    workflow.add_node("preprocess", preprocess_input)
    workflow.add_node("supervisor", supervisor_logic)

    # Track successfully added agents
    added_agents = []

    for name in AVAILABLE_AGENTS:
        try:
            subgraph = member_graphs[name]

            # Define a wrapper function for the subgraph node
            def subgraph_node(state):
                # Extract messages from supervisor state
                subgraph_state = {"messages": state["messages"]}
                # Run the subgraph
                result = subgraph.run(subgraph_state)
                # Update the supervisor's messages
                state["messages"] = result["messages"]
                return state

            # Add the wrapper function as the node
            workflow.add_node(name, subgraph_node)
            workflow.add_edge(name, "supervisor")
            added_agents.append(name)
            logger.info(f"Successfully added node: {name}")
        except Exception as e:
            logger.error(f"Failed to add node {name}: {str(e)}", exc_info=True)

    # Only include successfully added agents in conditional edges
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

# Export public symbols
__all__ = ["create_supervisor", "supervisor_workflow", "SupervisorState", "member_graphs"]