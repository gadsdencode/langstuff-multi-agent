# langstuff_multi_agent/agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langstuff_multi_agent.config import get_llm, ConfigSchema
from typing_extensions import TypedDict, Literal

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


# Define a structured output schema for routing.
class RouteResponse(TypedDict):
    next: Literal[
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


supervisor_workflow = StateGraph(MessagesState, ConfigSchema)

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
    """Route the user request to the appropriate agent workflow using structured LLM output."""
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [
                AIMessage(content="Welcome! I'm your AI assistant. How can I help you today?")
            ]
        }

    request = messages[-1].content

    prompt = (
        "You are a Supervisor Agent tasked with routing user requests to the most appropriate specialized agent.\n\n"
        "Available agents and their specialties:\n"
        f"{format_agent_list()}\n\n"
        f"User Request: '{request}'\n\n"
        "Instructions:\n"
        "1. Analyze the user's request carefully.\n"
        "2. Select the most appropriate agent based on their specialties.\n"
        "3. Respond with exactly one agent name from the list (case-insensitive) in JSON format, e.g. {\"next\": \"CODER\"}.\n"
        "4. If unsure, respond with {\"next\": \"GENERAL_ASSISTANT\"}.\n"
        "Selected agent:"
    )

    try:
        supervisor_llm = get_llm(config.get("configurable", {}))
        # Use structured output to enforce JSON formatting.
        route_response = supervisor_llm.with_structured_output(RouteResponse).invoke([SystemMessage(content=prompt)])
        agent_key = route_response["next"].strip().upper()

        if agent_key not in AGENT_OPTIONS:
            agent_key = "GENERAL_ASSISTANT"

        workflow = workflow_map[agent_key].compile()
        result = workflow.invoke(
            {"messages": [
                SystemMessage(content=f"You are the {agent_key} agent, specialized in {AGENT_OPTIONS[agent_key]}."), 
                HumanMessage(content=request)
            ]},
            config=config
        )

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


supervisor_workflow.add_node("route", route_request)


def create_agent_node(agent_workflow, agent_name):
    def agent_node(state, config):
        compiled_workflow = agent_workflow.compile()
        result = compiled_workflow.invoke(state, config=config)
        return result
    return agent_node


for agent_name, workflow in workflow_map.items():
    supervisor_workflow.add_node(
        agent_name.lower(),
        create_agent_node(workflow, agent_name)
    )

supervisor_workflow.add_edge(START, "route")
for agent_name in workflow_map.keys():
    supervisor_workflow.add_edge("route", agent_name.lower())
    supervisor_workflow.add_edge(agent_name.lower(), END)

__all__ = ["supervisor_workflow"]
