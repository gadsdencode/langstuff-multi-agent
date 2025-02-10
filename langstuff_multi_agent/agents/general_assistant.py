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

general_assistant_workflow = StateGraph(MessagesState)

# Define general assistant tools
tools = [search_web, get_current_weather]
tool_node = ToolNode(tools)

# Bind the LLM with the general assistant tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node with a system prompt for general assistance
general_assistant_workflow.add_node(
    "assist",
    lambda state: {
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
    },
)
general_assistant_workflow.add_node("tools", tool_node)

# Define control flow edges
general_assistant_workflow.add_edge(START, "assist")
general_assistant_workflow.add_edge(
    "assist",
    "tools",
    if_=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
general_assistant_workflow.add_edge("tools", "assist")
general_assistant_workflow.add_edge(
    "assist",
    END,
    if_=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
