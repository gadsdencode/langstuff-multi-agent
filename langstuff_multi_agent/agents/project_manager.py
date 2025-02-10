# agents/project_manager.py
"""
Project Manager Agent module for task and timeline management.

This module provides a workflow for overseeing project schedules
and coordinating tasks using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    calendar_tool,
    task_tracker_tool
)
from langchain_anthropic import ChatAnthropic

project_manager_workflow = StateGraph(MessagesState)

# Define project management tools
tools = [search_web, calendar_tool, task_tracker_tool]
tool_node = ToolNode(tools)

# Bind the LLM with tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node for managing projects with detailed instructions
project_manager_workflow.add_node(
    "manage_project",
    lambda state: {
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
    },
)
project_manager_workflow.add_node("tools", tool_node)

# Define control flow edges
project_manager_workflow.add_edge(START, "manage_project")

# Add conditional edge from manage_project to either tools or END
project_manager_workflow.add_conditional_edges(
    "manage_project",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to manage_project
project_manager_workflow.add_edge("tools", "manage_project")
