# agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph 
agent workflows.

This module uses an LLM—instantiated via the get_llm() factory function 
from langstuff_multi_agent/config.py—to classify incoming user requests 
and dynamically route the request to the appropriate specialized agent 
workflow. The available agents include:
  DEBUGGER, CONTEXT_MANAGER, PROJECT_MANAGER, PROFESSIONAL_COACH, 
  LIFE_COACH, CODER, ANALYST, RESEARCHER, and GENERAL_ASSISTANT.

Each agent workflow is compiled with persistent checkpointing enabled 
by explicitly passing the shared checkpointer 
(Config.PERSISTENT_CHECKPOINTER) during compilation.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from langstuff_multi_agent.config import Config, get_llm

# Import individual workflows.
from langstuff_multi_agent.agents.debugger import (
    debugger_workflow
)
from langstuff_multi_agent.agents.context_manager import (
    context_manager_workflow
)
from langstuff_multi_agent.agents.project_manager import (
    project_manager_workflow
)
from langstuff_multi_agent.agents.professional_coach import (
    professional_coach_workflow
)
from langstuff_multi_agent.agents.life_coach import (
    life_coach_workflow
)
from langstuff_multi_agent.agents.coder import (
    coder_workflow
)
from langstuff_multi_agent.agents.analyst import (
    analyst_workflow
)
from langstuff_multi_agent.agents.researcher import (
    researcher_workflow
)
from langstuff_multi_agent.agents.general_assistant import (
    general_assistant_workflow
)

# Create supervisor workflow
supervisor_workflow = StateGraph(MessagesState)

# Define the available agent options.
AGENT_OPTIONS = [
    "DEBUGGER",
    "CONTEXT_MANAGER",
    "PROJECT_MANAGER",
    "PROFESSIONAL_COACH",
    "LIFE_COACH",
    "CODER",
    "ANALYST",
    "RESEARCHER",
    "GENERAL_ASSISTANT"
]

# Map agent names to their workflows
workflow_map = {
    "DEBUGGER": debugger_workflow,
    "CONTEXT_MANAGER": context_manager_workflow,
    "PROJECT_MANAGER": project_manager_workflow,
    "PROFESSIONAL_COACH": professional_coach_workflow,
    "LIFE_COACH": life_coach_workflow,
    "CODER": coder_workflow,
    "ANALYST": analyst_workflow,
    "RESEARCHER": researcher_workflow,
    "GENERAL_ASSISTANT": general_assistant_workflow
}

# Get supervisor LLM
supervisor_llm = get_llm()

# Define the supervisor node
def route_request(state):
    """Route the user request to appropriate agent workflow"""
    request = state["messages"][-1].content
    prompt = (
        "You are a Supervisor Agent tasked with routing user requests to the most appropriate specialized agent. "
        f"Available agents: {', '.join(AGENT_OPTIONS)}.\n\n"
        f"Given the request: '{request}'\n"
        "Select exactly one agent (case-insensitive). Your answer:"
    )
    response = supervisor_llm.invoke([HumanMessage(content=prompt)])
    agent_key = response.content.strip().upper()
    if agent_key not in AGENT_OPTIONS:
        agent_key = "GENERAL_ASSISTANT"
    
    workflow = workflow_map[agent_key].compile()
    result = workflow.invoke({"messages": [HumanMessage(content=request)]})
    return {"messages": result["messages"]}

supervisor_workflow.add_node("route", route_request)

# Define edges
supervisor_workflow.add_edge(START, "route")
supervisor_workflow.add_edge("route", END)

# Export the workflow
__all__ = ["supervisor_workflow"]
