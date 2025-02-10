# agents/life_coach.py
"""
Life Coach Agent module for personal advice and guidance.

This module provides a workflow for offering lifestyle tips and
personal development advice using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather, calendar_tool
from langchain_anthropic import ChatAnthropic

life_coach_workflow = StateGraph(MessagesState)

# Define tools for life coaching
tools = [search_web, get_current_weather, calendar_tool]
tool_node = ToolNode(tools)

# Bind the LLM with tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node with instructions for personal and lifestyle advice
life_coach_workflow.add_node(
    "life_coach",
    lambda state: {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Life Coach Agent. Your task is to provide personal advice and lifestyle tips.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up general lifestyle tips and motivational content.\n"
                            "- get_current_weather: Provide weather updates to help plan outdoor activities.\n"
                            "- calendar_tool: Assist in scheduling and planning daily routines.\n\n"
                            "Instructions:\n"
                            "1. Listen to the user's personal queries and lifestyle challenges.\n"
                            "2. Offer practical advice and motivational support.\n"
                            "3. Use the available tools to supply additional context when necessary.\n"
                            "4. Maintain an empathetic and encouraging tone throughout the conversation."
                        ),
                    }
                ]
            )
        ]
    },
)
life_coach_workflow.add_node("tools", tool_node)

# Define control flow edges
life_coach_workflow.add_edge(START, "life_coach")

# Add conditional edge from life_coach to either tools or END
life_coach_workflow.add_conditional_edges(
    "life_coach",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to life_coach
life_coach_workflow.add_edge("tools", "life_coach")
