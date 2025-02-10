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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langstuff_multi_agent.config import Config, get_llm, ConfigSchema, create_model_config

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

# Create supervisor workflow with configuration schema
supervisor_workflow = StateGraph(MessagesState, ConfigSchema)

# Define the available agent options with descriptions
AGENT_OPTIONS = {
    "DEBUGGER": "Code debugging and error analysis",
    "CONTEXT_MANAGER": "Conversation context tracking and management",
    "PROJECT_MANAGER": "Project timeline and task management",
    "PROFESSIONAL_COACH": "Career advice and job search strategies",
    "LIFE_COACH": "Personal development and lifestyle guidance",
    "CODER": "Code writing and improvement",
    "ANALYST": "Data analysis and interpretation",
    "RESEARCHER": "Information gathering and research",
    "GENERAL_ASSISTANT": "General purpose assistance"
}

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

def format_agent_list():
    """Format the agent list with descriptions for the prompt."""
    return "\n".join(f"- {name}: {desc}" for name, desc in AGENT_OPTIONS.items())

def route_request(state, config):
    """Route the user request to appropriate agent workflow with enhanced feedback."""
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [
                AIMessage(content="Welcome! I'm your AI assistant. How can I help you today?")
            ]
        }
    
    request = messages[-1].content
    
    # Create a detailed prompt for agent selection
    prompt = (
        "You are a Supervisor Agent tasked with routing user requests to the most appropriate specialized agent.\n\n"
        "Available agents and their specialties:\n"
        f"{format_agent_list()}\n\n"
        f"User Request: '{request}'\n\n"
        "Instructions:\n"
        "1. Analyze the user's request carefully\n"
        "2. Select the most appropriate agent based on their specialties\n"
        "3. Respond with exactly one agent name from the list (case-insensitive)\n"
        "4. If unsure, use GENERAL_ASSISTANT\n\n"
        "Selected agent:"
    )
    
    try:
        # Get supervisor LLM with configuration
        supervisor_llm = get_llm(config.get("configurable", {}))
        
        # Get agent selection
        response = supervisor_llm.invoke([SystemMessage(content=prompt)])
        agent_key = response.content.strip().upper()
        
        if agent_key not in AGENT_OPTIONS:
            agent_key = "GENERAL_ASSISTANT"
        
        # Compile and invoke the selected workflow with configuration
        workflow = workflow_map[agent_key].compile()
        result = workflow.invoke(
            {"messages": [
                SystemMessage(content=f"You are the {agent_key} agent, specialized in {AGENT_OPTIONS[agent_key]}."),
                HumanMessage(content=request)
            ]},
            config=config
        )
        
        # Add routing information to the response
        return {
            "messages": [
                AIMessage(content=f"[Routing to {agent_key} - {AGENT_OPTIONS[agent_key]}]\n\n"),
                *result["messages"]
            ]
        }
        
    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"I apologize, but I encountered an error while processing your request: {str(e)}\n\n"
                                "Please try rephrasing your request or contact support if the issue persists.")
            ]
        }

# Add the routing node with configuration support
supervisor_workflow.add_node("route", route_request)

# Add nodes for each agent
for agent_name, workflow in workflow_map.items():
    supervisor_workflow.add_node(agent_name.lower(), workflow.compile())

# Define edges
supervisor_workflow.add_edge(START, "route")
for agent_name in workflow_map.keys():
    supervisor_workflow.add_edge("route", agent_name.lower())
    supervisor_workflow.add_edge(agent_name.lower(), END)

# Export the workflow
__all__ = ["supervisor_workflow"]
