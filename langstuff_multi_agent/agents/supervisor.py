# langstuff_multi_agent/agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
"""

from langgraph.graph import StateGraph, Node
from langchain_core.messages import HumanMessage
from langstuff_multi_agent.config import get_llm
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Import individual workflows.
from langstuff_multi_agent.agents.debugger import debugger_workflow
from langstuff_multi_agent.agents.context_manager import context_manager_workflow
from langstuff_multi_agent.agents.project_manager import project_manager_workflow
from langstuff_multi_agent.agents.professional_coach import professional_coach_workflow
from langstuff_multi_agent.agents.life_coach import life_coach_workflow
from langstuff_multi_agent.agents.coder import coder_workflow
from langstuff_multi_agent.agents.analyst import analyst_workflow
from langstuff_multi_agent.agents.researcher import researcher_workflow
from langstuff_multi_agent.agents.general_assistant import general_assistant_workflow


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
        'general_assistant'
    ] = Field(..., description="Target agent for this request")
    require_tools: list[str] = Field(
        default_factory=list,
        description="Required tools for this task"
    )


def route_query(state: RouterInput):
    """Classifies and routes user queries using structured LLM output"""
    llm = get_llm({"structured_output_method": "json_mode"})
    system = """You are an expert router for a multi-agent system. Analyze the user's query 
    and route to ONE specialized agent. Consider these specialties:
    - Debugger: Code errors, troubleshooting
    - Coder: Writing/explaining code
    - Analyst: Data analysis requests
    - Researcher: Fact-finding, web research
    - Project Manager: Task planning
    - Life/Professional Coach: Personal/career advice
    - General Assistant: Everything else"""

    structured_llm = llm.with_structured_output(RouteDecision)
    return structured_llm.invoke([{
        "role": "user",
        "content": f"Route this query: {state.messages[-1].content}"
    }], config={"system": system})


# Supervisor workflow construction
def create_supervisor_workflow():
    builder = StateGraph(RouterInput)

    # Define nodes
    builder.add_node("route_query", route_query)
    builder.add_node("debugger", debugger_workflow)
    builder.add_node("context_manager", context_manager_workflow)
    builder.add_node("project_manager", project_manager_workflow)
    builder.add_node("professional_coach", professional_coach_workflow)
    builder.add_node("life_coach", life_coach_workflow)
    builder.add_node("coder", coder_workflow)
    builder.add_node("analyst", analyst_workflow)
    builder.add_node("researcher", researcher_workflow)
    builder.add_node("general_assistant", general_assistant_workflow)

    # Conditional edges
    def decide_routes(state: RouteDecision):
        return state.destination

    builder.add_conditional_edges(
        "route_query",
        decide_routes,
        {
            "debugger": "debugger",
            "context_manager": "context_manager",
            "project_manager": "project_manager",
            "professional_coach": "professional_coach",
            "life_coach": "life_coach",
            "coder": "coder",
            "analyst": "analyst",
            "researcher": "researcher",
            "general_assistant": "general_assistant"
        }
    )

    builder.set_entry_point("route_query")
    return builder.compile()


__all__ = ["create_supervisor_workflow"]
