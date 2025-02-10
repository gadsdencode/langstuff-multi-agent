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

professional_coach_workflow = StateGraph(MessagesState)

# Define the tools for professional coaching
tools = [search_web, job_search_tool]
tool_node = ToolNode(tools)

# Bind the LLM with the available tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node with a system prompt for career guidance
professional_coach_workflow.add_node(
    "coach",
    lambda state: {
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
    },
)
professional_coach_workflow.add_node("tools", tool_node)

# Define control flow edges
professional_coach_workflow.add_edge(START, "coach")
professional_coach_workflow.add_edge(
    "coach",
    "tools",
    condition=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
professional_coach_workflow.add_edge("tools", "coach")
professional_coach_workflow.add_edge(
    "coach",
    END,
    condition=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
