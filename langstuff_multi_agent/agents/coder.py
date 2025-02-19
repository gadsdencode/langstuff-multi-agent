"""
Coder Agent module for writing and improving code.

This module provides a workflow for code generation, debugging, and optimization using various development tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, python_repl, read_file, write_file, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def code(state: MessagesState, config: dict) -> dict:
    """Write and improve code with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, python_repl, read_file, write_file, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Coder Agent. Your task is to write, debug, and improve code.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Find coding examples and docs.\n"
            "- python_repl: Execute and test Python code.\n"
            "- read_file: Retrieve code from files.\n"
            "- write_file: Save code modifications to files.\n"
            "- calc_tool: Perform calculations if needed.\n\n"
            "Instructions:\n"
            "1. Analyze the user's code or coding request.\n"
            "2. Provide solutions, test code, and explain your reasoning.\n"
            "3. Use the available tools to execute code and verify fixes as necessary."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
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

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    return tool_node(state)

# Define and compile the graph
coder_graph = StateGraph(MessagesState)
coder_graph.add_node("code", code)
coder_graph.add_node("tools", tools_node)  # Use wrapped tools_node
coder_graph.add_node("process_results", process_tool_results)
coder_graph.set_entry_point("code")
coder_graph.add_conditional_edges(
    "code",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
coder_graph.add_edge("tools", "process_results")
coder_graph.add_edge("process_results", "code")
coder_graph = coder_graph.compile()

__all__ = ["coder_graph"]