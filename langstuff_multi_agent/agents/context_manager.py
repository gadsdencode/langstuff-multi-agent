# langstuff_multi_agent/agents/context_manager.py
"""
Context Manager Agent module for tracking conversation context.

This module provides a workflow for managing conversation history
and maintaining context across interactions.
"""
import json
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    read_file,
    write_file,
    has_tool_calls
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage

# 1. Initialize workflow FIRST
context_manager_workflow = StateGraph(MessagesState, ConfigSchema)

# Define tools for context management
tools = [search_web, read_file, write_file]
tool_node = ToolNode(tools)


def save_context(state):
    """Saves conversation history to a file"""
    with open("context.json", "w") as f:
        json.dump(state["messages"], f)


def load_context():
    """Loads previous conversation history"""
    try:
        with open("context.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def manage_context(state, config):
    """Manages conversation context with persistent storage"""
    previous_messages = load_context()
    updated_messages = previous_messages + state["messages"]

    save_context({"messages": updated_messages})  # Save merged history

    llm = get_llm(config.get("configurable", {}))
    return {
        "messages": [
            llm.invoke(updated_messages + [{
                "role": "system",
                "content": "Track and summarize conversation history."
            }])
        ]
    }


def process_tool_results(state, config):
    """Process tool outputs and generate final response."""
    # Add handoff command detection
    for msg in state["messages"]:
        if tool_calls := getattr(msg, 'tool_calls', None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {
                        "messages": [ToolMessage(
                            goto=tc['name'].replace('transfer_to_', ''),
                            graph=ToolMessage.PARENT
                        )]
                    }

    tool_outputs = []

    for msg in state["messages"]:
        if tool_calls := getattr(msg, "tool_calls", None):
            for tc in tool_calls:
                try:
                    output = f"Tool {tc['name']} result: {tc['output']}"
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "output": output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tc["id"],
                        "error": f"Tool execution failed: {str(e)}"
                    })

    return {
        "messages": state["messages"] + [
            {
                "role": "tool",
                "content": to["output"],
                "tool_call_id": to["tool_call_id"]
            } 
            for to in tool_outputs
        ]
    }


# 2. Add nodes BEFORE compiling
context_manager_workflow.add_node("manage_context", manage_context)
context_manager_workflow.add_node("tools", tool_node)
context_manager_workflow.add_node("process_results", process_tool_results)

# 3. Set entry point explicitly
context_manager_workflow.set_entry_point("manage_context")

# 4. Add edges in sequence
context_manager_workflow.add_edge(START, "manage_context")
context_manager_workflow.add_conditional_edges(
    "manage_context",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else END
    ),
    {"tools": "tools", END: END}
)
context_manager_workflow.add_edge("tools", "process_results")
context_manager_workflow.add_edge("process_results", "manage_context")

# 5. Compile ONCE at the end
context_manager_graph = context_manager_workflow.compile()

__all__ = ["context_manager_graph"]
