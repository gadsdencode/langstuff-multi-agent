# langstuff_multi_agent/agents/context_manager.py
"""
Context Manager Agent module for tracking conversation context.

This module provides a workflow for managing conversation history
and maintaining context across interactions.
"""
import json
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, read_file, write_file, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

context_manager_workflow = StateGraph(MessagesState, ConfigSchema)

# Define tools for context management
tools = [search_web, read_file, write_file]
tool_node = ToolNode(tools)


def save_context(state):
    """Saves conversation history to a file"""
    with open("context.json", "w") as f:
        json.dump(state["messages"], f)


def load_context():
    """Loads previous conversation history"""
    try:
        with open("context.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def manage_context(state, config):
    """Manages conversation context with persistent storage"""
    previous_messages = load_context()
    updated_messages = previous_messages + state["messages"]

    save_context({"messages": updated_messages})  # Save merged history

    llm = get_llm(config.get("configurable", {}))
    return {
        "messages": [
            llm.invoke(updated_messages + [{
                "role": "system",
                "content": "Track and summarize conversation history."
            }])
        ]
    }


context_manager_graph = context_manager_workflow.compile()

context_manager_graph.add_node("manage_context", manage_context)
context_manager_graph.set_entry_point("manage_context")
context_manager_graph.add_node("tools", tool_node)

context_manager_graph.add_edge(START, "manage_context")

context_manager_graph.add_conditional_edges(
    "manage_context",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

context_manager_graph.add_edge("tools", "manage_context")

context_manager_graph = context_manager_graph.compile()

__all__ = ["context_manager_graph"]
