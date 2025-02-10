# agents/analyst.py
"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing
calculations using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    python_repl,
    calc_tool
)
from langchain_anthropic import ChatAnthropic


analyst_workflow = StateGraph(MessagesState)

# Define tools for analysis tasks
tools = [search_web, python_repl, calc_tool]
tool_node = ToolNode(tools)

# Bind the LLM with analytical tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node for data analysis with a detailed system prompt
analyst_workflow.add_node(
    "analyze_data",
    lambda state: {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are an Analyst Agent. Your task is to analyze data, perform calculations, and interpret results.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Retrieve background information and data.\n"
                            "- python_repl: Run Python code to perform calculations and tests.\n"
                            "- calc_tool: Execute specific calculations and numerical analysis.\n\n"
                            "Instructions:\n"
                            "1. Review the data or query provided by the user.\n"
                            "2. Perform necessary calculations and analyze the results.\n"
                            "3. Summarize your findings in clear, concise language."
                        ),
                    }
                ]
            )
        ]
    },
)
analyst_workflow.add_node("tools", tool_node)

# Define control flow edges
analyst_workflow.add_edge(START, "analyze_data")
analyst_workflow.add_edge(
    "analyze_data",
    "tools",
    if_=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
analyst_workflow.add_edge("tools", "analyze_data")
analyst_workflow.add_edge(
    "analyze_data",
    END,
    if_=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
