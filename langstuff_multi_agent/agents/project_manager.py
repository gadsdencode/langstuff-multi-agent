"""
Project Manager Agent module for task and timeline management.

This module provides a workflow for overseeing project schedules and coordinating tasks using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, python_repl
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def manage(state: MessagesState, config: dict) -> dict:
    """Project management agent that coordinates tasks and timelines."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, python_repl]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Project Manager Agent. Your task is to oversee project schedules and coordinate tasks.\n"
            "Use tools like search_web and python_repl to gather info or perform calculations.\n"
            "Provide actionable plans or updates."
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
        tool = next(t for t in [search_web, python_repl] if t.name == tc["name"])
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
project_manager_graph = StateGraph(MessagesState)
project_manager_graph.add_node("manage", manage)
project_manager_graph.add_node("tools", tools_node)  # Use wrapped tools_node
project_manager_graph.add_node("process_results", process_tool_results)
project_manager_graph.set_entry_point("manage")
project_manager_graph.add_conditional_edges(
    "manage",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
project_manager_graph.add_edge("tools", "process_results")
project_manager_graph.add_edge("process_results", "manage")
project_manager_graph = project_manager_graph.compile()

__all__ = ["project_manager_graph"]