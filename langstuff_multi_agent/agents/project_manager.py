"""
Project Manager Agent module for task and timeline management.

This module provides a workflow for overseeing project schedules and coordinating tasks using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, python_repl
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Helper function to convert messages to BaseMessage objects
def convert_message(msg):
    if isinstance(msg, dict):
        msg_type = msg.get("type", msg.get("role"))  # Support both "type" and "role"
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant" or msg_type == "ai":
            return AIMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls", []))
        elif msg_type == "system":
            return SystemMessage(content=msg.get("content", ""))
        elif msg_type == "tool":
            return ToolMessage(
                content=msg.get("content", ""),
                tool_call_id=msg.get("tool_call_id", ""),
                name=msg.get("name", "")
            )
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    return msg

def convert_messages(messages):
    return [convert_message(msg) for msg in messages]

def manage(state: MessagesState, config: dict) -> dict:
    """Project management agent that coordinates tasks and timelines."""
    # Convert incoming messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, python_repl]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Project Manager Agent. Your task is to oversee project schedules and coordinate tasks.\n"
            "Use tools like search_web and python_repl to gather info or perform calculations.\n"
            "Provide actionable plans or updates."
        ))
    ])
    
    # Ensure response is an AIMessage
    if isinstance(response, dict):
        content = response.get("content", "")
        raw_tool_calls = response.get("tool_calls", [])
        tool_calls = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            if isinstance(tc, dict) else tc
            for tc in raw_tool_calls
        ]
        response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs and formats final response."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, python_repl] if t.name == tc["name"])
        output = tool.invoke(tc["args"])
        tool_messages.append(ToolMessage(
            content=str(output),
            tool_call_id=tc["id"],
            name=tc["name"]
        ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    
    # Ensure final_response is an AIMessage
    if isinstance(final_response, dict):
        content = final_response.get("content", "Task completed")
        raw_tool_calls = final_response.get("tool_calls", [])
        tool_calls = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in raw_tool_calls
        ]
        final_response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(final_response, AIMessage):
        final_response = AIMessage(content=str(final_response))
    
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    state["messages"] = convert_messages(state["messages"])
    return tool_node(state)

# Define and compile the graph
project_manager_graph = StateGraph(MessagesState)
project_manager_graph.add_node("manage", manage)
project_manager_graph.add_node("tools", tools_node)
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