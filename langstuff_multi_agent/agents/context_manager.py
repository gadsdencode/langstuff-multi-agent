"""
Context Manager Agent module for tracking conversation context.

This module provides a workflow for managing conversation history and maintaining context across interactions.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, read_file, write_file, save_memory, search_memories, news_tool, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import SystemMessage, ToolMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def manage_context(state: MessagesState, config: dict) -> dict:
    """Manage conversation context using memory tools."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, read_file, write_file, save_memory, search_memories]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Context Manager Agent. Your task is to track and maintain conversation context.\n"
            "You have access to the following tools:\n"
            "- search_web: Look up additional context.\n"
            "- read_file: Retrieve stored context from files.\n"
            "- write_file: Save context to files.\n"
            "- save_memory: Save conversation history to memory.\n"
            "- search_memories: Retrieve relevant past context.\n\n"
            "Instructions:\n"
            "1. Analyze the current conversation.\n"
            "2. Use tools to save or retrieve context as needed.\n"
            "3. Provide a response that reflects the full context."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, news_tool, calc_tool] if t.name == tc["name"])
        try:
            output = tool.invoke(tc["args"])
            tool_messages.append(ToolMessage(
                content=str(output),
                tool_call_id=tc["id"],
                name=tc["name"]
            ))
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_messages.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tc["id"],
                name=tc["name"]
            ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    return tool_node(state)

# Define and compile the graph
context_manager_graph = StateGraph(MessagesState)
context_manager_graph.add_node("manage_context", manage_context)
context_manager_graph.add_node("tools", tools_node)  # Use wrapped tools_node
context_manager_graph.add_node("process_results", process_tool_results)
context_manager_graph.set_entry_point("manage_context")
context_manager_graph.add_conditional_edges(
    "manage_context",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
context_manager_graph.add_edge("tools", "process_results")
context_manager_graph.add_edge("process_results", "manage_context")
context_manager_graph = context_manager_graph.compile()

__all__ = ["context_manager_graph"]