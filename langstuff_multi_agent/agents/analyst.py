# langstuff_multi_agent/agents/analyst.py
"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing
calculations using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, python_repl, calc_tool, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

analyst_workflow = StateGraph(MessagesState, ConfigSchema)

# Define tools for analysis tasks
tools = [search_web, python_repl, calc_tool]
tool_node = ToolNode(tools)


def analyze_data(state, config):
    """Analyze data with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
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
    }


analyst_workflow.add_node("analyze_data", analyze_data)
analyst_workflow.add_node("tools", tool_node)

analyst_workflow.add_edge(START, "analyze_data")

analyst_workflow.add_conditional_edges(
    "analyze_data",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

analyst_workflow.add_edge("tools", "analyze_data")

__all__ = ["analyst_workflow"]
