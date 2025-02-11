# langstuff_multi_agent/agents/context_manager.py
"""
Context Manager Agent module for tracking conversation context.

This module provides a workflow for managing conversation history
and maintaining context across interactions.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, read_file, write_file, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

context_manager_workflow = StateGraph(MessagesState, ConfigSchema)

# Define tools for context management
tools = [search_web, read_file, write_file]
tool_node = ToolNode(tools)


def manage_context(state, config):
    """Manage conversation context with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Context Manager Agent. Your task is to track and manage conversation context.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search the web for general information and recent content.\n"
                            "- read_file: Read the contents of a file.\n"
                            "- write_file: Write content to a file.\n\n"
                            "Instructions:\n"
                            "1. Keep track of key information and topics discussed in the conversation.\n"
                            "2. Summarize important points and decisions made.\n"
                            "3. Use read_file and write_file to store and retrieve context information.\n"
                            "4. If necessary, use search_web to gather additional context.\n"
                            "5. Ensure that the conversation stays focused and relevant."
                        ),
                    }
                ]
            )
        ]
    }


context_manager_workflow.add_node("manage_context", manage_context)
context_manager_workflow.add_node("tools", tool_node)

context_manager_workflow.add_edge(START, "manage_context")

context_manager_workflow.add_conditional_edges(
    "manage_context",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

context_manager_workflow.add_edge("tools", "manage_context")

__all__ = ["context_manager_workflow"]
