# langstuff_multi_agent/agents/debugger.py
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
    write_file,
    has_tool_calls
)
from langstuff_multi_agent.config import get_llm
from langchain.schema import Command

debugger_workflow = StateGraph(MessagesState)

# Define the tools available to the Debugger Agent
tools = [search_web, python_repl, read_file, write_file]
tool_node = ToolNode(tools)


def analyze_code(state):
    """Analyze code and identify errors."""
    messages = state.get("messages", [])
    config = state.get("config", {})
    
    llm = get_llm(config.get("configurable", {}))
    response = llm.invoke(messages)
    
    return {"messages": messages + [response]}


def process_tool_results(state, config):
    """Processes tool outputs and formats FINAL user response"""
    # Add handoff command detection
    for msg in state["messages"]:
        if tool_calls := getattr(msg, 'tool_calls', None):
            for tc in tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {
                        "messages": [Command(
                            goto=tc['name'].replace('transfer_to_', ''),
                            graph=Command.PARENT
                        )]
                    }

    last_message = state["messages"][-1]
    tool_outputs = []

    if tool_calls := getattr(last_message, 'tool_calls', None):
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
                } for to in tool_outputs
            ]
        }
    return state


# Initialize and configure the debugger workflow
debugger_workflow.add_node("analyze_code", analyze_code)
debugger_workflow.add_node("tools", tool_node)
debugger_workflow.add_node("process_results", process_tool_results)
debugger_workflow.set_entry_point("analyze_code")
debugger_workflow.add_edge(START, "analyze_code")

debugger_workflow.add_conditional_edges(
    "analyze_code",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

debugger_workflow.add_edge("tools", "process_results")
debugger_workflow.add_edge("process_results", "analyze_code")

debugger_graph = debugger_workflow.compile()

__all__ = ["debugger_graph"]
