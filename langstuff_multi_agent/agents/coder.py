# agents/coder.py
"""
Coder Agent module for writing and improving code.

This module provides a workflow for code generation, debugging,
and optimization using various development tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, python_repl, read_file, write_file
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

coder_workflow = StateGraph(MessagesState, ConfigSchema)

# Define tools for coding tasks
tools = [search_web, python_repl, read_file, write_file]
tool_node = ToolNode(tools)

def code(state, config):
    """Write and improve code with configuration support."""
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
                            "You are a Coder Agent. Your task is to write, debug, and improve code.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Find coding examples and documentation.\n"
                            "- python_repl: Execute and test Python code snippets.\n"
                            "- read_file: Retrieve code from files.\n"
                            "- write_file: Save code modifications to files.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's code or coding request.\n"
                            "2. Provide solutions, test code, and explain your reasoning.\n"
                            "3. Use the available tools to execute code and verify fixes as necessary."
                        ),
                    }
                ]
            )
        ]
    }

# Define the main node with configuration support
coder_workflow.add_node("code", code)
coder_workflow.add_node("tools", tool_node)

# Define control flow edges
coder_workflow.add_edge(START, "code")

# Add conditional edge from code to either tools or END
coder_workflow.add_conditional_edges(
    "code",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to code
coder_workflow.add_edge("tools", "code")

# Export the workflow
__all__ = ["coder_workflow"]
