# langstuff_multi_agent/agents/professional_coach.py
"""
Professional Coach Agent module for career guidance.

This module provides a workflow for offering career advice and
job search strategies using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, job_search_tool, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

professional_coach_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools for professional coaching
tools = [search_web, job_search_tool]
tool_node = ToolNode(tools)


def coach(state, config):
    """Provide professional coaching with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Professional Coach Agent. Your task is to provide career advice and job search strategies.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search for career advice and job market trends.\n"
                            "- job_search_tool: Retrieve job listings and career opportunities.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's career-related queries.\n"
                            "2. Offer actionable advice and strategies for job searching.\n"
                            "3. Use the available tools to provide up-to-date information and resources.\n"
                            "4. Communicate in a supportive and motivational tone."
                        ),
                    }
                ]
            )
        ]
    }


professional_coach_graph.add_node("coach", coach)
professional_coach_graph.add_node("tools", tool_node)
professional_coach_graph.set_entry_point("coach")
professional_coach_graph.add_edge(START, "coach")

professional_coach_graph.add_conditional_edges(
    "coach",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

professional_coach_graph.add_edge("tools", "coach")

professional_coach_graph = professional_coach_graph.compile()

__all__ = ["professional_coach_graph"]
