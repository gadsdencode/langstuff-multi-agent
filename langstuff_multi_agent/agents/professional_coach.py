# agents/professional_coach.py
"""
Professional Coach Agent module for career guidance.

This module provides a workflow for offering career advice and
job search strategies using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, job_search_tool
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

professional_coach_workflow = StateGraph(MessagesState, ConfigSchema)

# Define the tools for professional coaching
tools = [search_web, job_search_tool]
tool_node = ToolNode(tools)

def coach(state, config):
    """Provide professional coaching with configuration support."""
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

# Define the main node with configuration support
professional_coach_workflow.add_node("coach", coach)
professional_coach_workflow.add_node("tools", tool_node)

# Define control flow edges
professional_coach_workflow.add_edge(START, "coach")

# Add conditional edge from coach to either tools or END
professional_coach_workflow.add_conditional_edges(
    "coach",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to coach
professional_coach_workflow.add_edge("tools", "coach")

# Export the workflow
__all__ = ["professional_coach_workflow"]
