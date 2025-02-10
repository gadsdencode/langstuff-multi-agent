# agents/general_assistant.py
"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests
using a variety of tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

general_assistant_workflow = StateGraph(MessagesState, ConfigSchema)

# Define general assistant tools
tools = [search_web, get_current_weather]
tool_node = ToolNode(tools)

def assist(state, config):
    """Provide general assistance with configuration support."""
    # Get LLM with configuration
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a General Assistant Agent. Your task is to assist with a variety of general queries and tasks.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Provide general information and answer questions.\n"
                            "- get_current_weather: Retrieve current weather updates.\n\n"
                            "Instructions:\n"
                            "1. Understand the user's request.\n"
                            "2. Use the available tools to gather relevant information when needed.\n"
                            "3. Provide clear, concise, and helpful responses to assist the user."
                        ),
                    }
                ]
            )
        ]
    }

# Define the main node with configuration support
general_assistant_workflow.add_node("assist", assist)
general_assistant_workflow.add_node("tools", tool_node)

# Define control flow edges
general_assistant_workflow.add_edge(START, "assist")

# Add conditional edge from assist to either tools or END
general_assistant_workflow.add_conditional_edges(
    "assist",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to assist
general_assistant_workflow.add_edge("tools", "assist")

# Export the workflow
__all__ = ["general_assistant_workflow"]
