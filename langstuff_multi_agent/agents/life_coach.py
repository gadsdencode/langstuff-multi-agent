# langstuff_multi_agent/agents/life_coach.py
"""
Life Coach Agent module for personal advice and guidance.

This module provides a workflow for offering lifestyle tips and
personal development advice using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather, calendar_tool, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

life_coach_graph = StateGraph(MessagesState, ConfigSchema)

# Define tools for life coaching
tools = [search_web, get_current_weather, calendar_tool]
tool_node = ToolNode(tools)


def life_coach(state, config):
    """Provide life coaching with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
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
    }


life_coach_graph.add_node("life_coach", life_coach)
life_coach_graph.add_node("tools", tool_node)
life_coach_graph.set_entry_point("life_coach")
life_coach_graph.add_edge(START, "life_coach")

life_coach_graph.add_conditional_edges(
    "life_coach",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

life_coach_graph.add_edge("tools", "life_coach")

life_coach_graph = life_coach_graph.compile()

__all__ = ["life_coach_graph"]
