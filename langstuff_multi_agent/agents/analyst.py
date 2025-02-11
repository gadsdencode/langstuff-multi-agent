# langstuff_multi_agent/agents/analyst.py
"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing
calculations using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    python_repl,
    calc_tool,
    has_tool_calls
)
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import ToolMessage

analyst_graph = StateGraph(MessagesState)

# Define tools for analysis tasks
tools = [search_web, python_repl, calc_tool]
tool_node = ToolNode(tools)


def analyze_data(state):
    """Analyze data and perform calculations."""
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
                        "messages": [ToolMessage(
                            goto=tc['name'].replace('transfer_to_', ''),
                            graph=ToolMessage.PARENT
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


# Initialize and configure the analyst graph
analyst_graph.add_node("analyze_data", analyze_data)
analyst_graph.add_node("tools", tool_node)
analyst_graph.add_node("process_results", process_tool_results)
analyst_graph.set_entry_point("analyze_data")
analyst_graph.add_edge(START, "analyze_data")

analyst_graph.add_conditional_edges(
    "analyze_data",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

analyst_graph.add_edge("tools", "process_results")
analyst_graph.add_edge("process_results", "analyze_data")

analyst_graph = analyst_graph.compile()

__all__ = ["analyst_graph"]
