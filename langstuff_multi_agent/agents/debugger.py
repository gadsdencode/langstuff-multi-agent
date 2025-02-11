# langstuff_multi_agent/agents/debugger.py
"""
Debugger Agent module for analyzing code and identifying errors.

This module provides a workflow for debugging code using various tools
and LLM-based analysis.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, python_repl, read_file, write_file, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

debugger_workflow = StateGraph(MessagesState, ConfigSchema)

# Define the tools available to the Debugger Agent
tools = [search_web, python_repl, read_file, write_file]
tool_node = ToolNode(tools)


def analyze_code(state, config):
    """Analyze code and identify errors with configuration support."""
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


def process_tool_results(state, config):
    """Process tool outputs and generate final response."""
    llm = get_llm(config.get("configurable", {}))
    tool_outputs = [tc["output"] for msg in state["messages"] for tc in getattr(msg, "tool_calls", [])]
    
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [{
                    "role": "system",
                    "content": (
                        "Process the tool outputs and provide a final response.\n\n"
                        f"Tool outputs: {tool_outputs}\n\n"
                        "Instructions:\n"
                        "1. Review the tool outputs in context of the debugging query.\n"
                        "2. Explain the identified issues and solutions clearly.\n"
                        "3. Include code fixes and verification steps."
                    )
                }]
            )
        ]
    }


debugger_workflow.add_node("analyze_code", analyze_code)
debugger_workflow.add_node("tools", tool_node)
debugger_workflow.add_node("process_results", process_tool_results)

debugger_workflow.add_edge(START, "analyze_code")

debugger_workflow.add_conditional_edges(
    "analyze_code",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

debugger_workflow.add_edge("tools", "process_results")
debugger_workflow.add_edge("process_results", END)

debugger_graph = debugger_workflow.compile()

__all__ = ["debugger_graph"]
