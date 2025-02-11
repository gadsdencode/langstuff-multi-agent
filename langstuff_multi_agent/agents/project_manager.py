# langstuff_multi_agent/agents/project_manager.py
"""
Project Manager Agent module for task and timeline management.

This module provides a workflow for overseeing project schedules
and coordinating tasks using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, calendar_tool, task_tracker_tool, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

project_manager_graph = StateGraph(MessagesState, ConfigSchema)

# Define project management tools
tools = [search_web, calendar_tool, task_tracker_tool]
tool_node = ToolNode(tools)


def manage_project(state, config):
    """Manage project with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Project Manager Agent. Your task is to oversee project timelines, tasks, and scheduling.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search the web for project management best practices.\n"
                            "- calendar_tool: Access and update project calendars.\n"
                            "- task_tracker_tool: Manage and update project task lists.\n\n"
                            "Instructions:\n"
                            "1. Review project details and timelines.\n"
                            "2. Update project schedules and task lists as needed.\n"
                            "3. Use search_web for additional project management information.\n"
                            "4. Provide clear instructions and updates regarding project progress."
                        ),
                    }
                ]
            )
        ]
    }


project_manager_graph.add_node("manage_project", manage_project)
project_manager_graph.add_node("tools", tool_node)
project_manager_graph.set_entry_point("manage_project")
project_manager_graph.add_edge(START, "manage_project")

project_manager_graph.add_conditional_edges(
    "manage_project",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

project_manager_graph.add_edge("tools", "manage_project")

project_manager_graph = project_manager_graph.compile()

__all__ = ["project_manager_graph"]
