"""
Debugger Agent module for analyzing code and identifying errors.

This module provides a workflow for debugging code using various tools and LLM-based analysis.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, python_repl, read_file, write_file, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm

debugger_graph = StateGraph(MessagesState, ConfigSchema)


def analyze_code(state, config):
    """Analyze code and identify errors."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, python_repl, read_file, write_file, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        {
            "role": "system",
            "content": (
                "You are a Debugger Agent. Your task is to analyze code and identify errors.\n"
                "You have access to the following tools:\n"
                "- search_web: Look up debugging resources.\n"
                "- python_repl: Test code snippets.\n"
                "- read_file: Retrieve code from files.\n"
                "- write_file: Save corrected code.\n"
                "- calc_tool: Perform calculations if needed.\n\n"
                "Instructions:\n"
                "1. Analyze the user's code or error description.\n"
                "2. Use tools to test or research solutions.\n"
                "3. Provide a clear explanation and fix."
            )
        }
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}


def process_tool_results(state, config):
    """Processes tool outputs and formats final response."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, python_repl, read_file, write_file, calc_tool] if t.name == tc["name"])
        output = tool.invoke(tc["args"])
        tool_messages.append({
            "role": "tool",
            "content": output,
            "tool_call_id": tc["id"]
        })

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}


debugger_graph.add_node("analyze_code", analyze_code)
debugger_graph.add_node("tools", tool_node)
debugger_graph.add_node("process_results", process_tool_results)
debugger_graph.set_entry_point("analyze_code")
debugger_graph.add_conditional_edges(
    "analyze_code",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
debugger_graph.add_edge("tools", "process_results")
debugger_graph.add_edge("process_results", "analyze_code")
debugger_graph = debugger_graph.compile()

__all__ = ["debugger_graph"]
