# agents/debugger.py
"""
Debugger Agent module for analyzing code and identifying errors.

This module provides a workflow for debugging code using various tools
and LLM-based analysis.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    python_repl,
    read_file,
    write_file
)
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

debugger_workflow = StateGraph(MessagesState, ConfigSchema)

# Define the tools available to the Debugger Agent
tools = [search_web, python_repl, read_file, write_file]
tool_node = ToolNode(tools)

def analyze_code(state, config):
    """Analyze code and identify errors with configuration support."""
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
                            "You are a Debugger Agent. Your task is to identify and analyze code errors.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search the web for general information and recent content.\n"
                            "- python_repl: Execute Python code.\n"
                            "- read_file: Read the contents of a file.\n"
                            "- write_file: Write content to a file.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's code and identify potential errors.\n"
                            "2. Use the search_web tool to find relevant information about the error or related debugging techniques.\n"
                            "3. Use the python_repl tool to execute code snippets and test potential fixes.\n"
                            "4. If necessary, use read_file and write_file to modify the code.\n"
                            "5. Provide clear and concise explanations of the error and the debugging process."
                        ),
                    }
                ]
            )
        ]
    }

# Define the main agent node with configuration support
debugger_workflow.add_node("analyze_code", analyze_code)
debugger_workflow.add_node("tools", tool_node)

# Define the control flow edges:
# 1. Start at 'analyze_code'.
# 2. If any message contains tool_calls, transition from 'analyze_code' to 'tools'.
# 3. After running tools, loop back to 'analyze_code'.
# 4. If no tool_calls remain, finish.
debugger_workflow.add_edge(START, "analyze_code")

# Add conditional edge from analyze_code to either tools or END
debugger_workflow.add_conditional_edges(
    "analyze_code",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to analyze_code
debugger_workflow.add_edge("tools", "analyze_code")

# Export the workflow
__all__ = ["debugger_workflow"]
