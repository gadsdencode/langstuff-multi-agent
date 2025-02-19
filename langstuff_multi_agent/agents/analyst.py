"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing calculations using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, python_repl, calc_tool, news_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import AIMessage, SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def analyze_data(state: MessagesState, config: dict) -> dict:
    """Analyze data and perform calculations."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, python_repl, calc_tool, news_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are an Analyst Agent. Your task is to analyze data and perform calculations.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Gather data from the web.\n"
            "- python_repl: Execute Python code for analysis.\n"
            "- calc_tool: Perform mathematical calculations.\n"
            "- news_tool: Retrieve relevant news data.\n\n"
            "Instructions:\n"
            "1. Analyze the user's data request.\n"
            "2. Use tools to gather or compute data.\n"
            "3. Provide a clear, concise analysis."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs and formats final analysis."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, python_repl, calc_tool, news_tool] if t.name == tc["name"])
        try:
            output = tool.invoke(tc["args"])
            tool_messages.append({
                "role": "tool",
                "content": output,
                "tool_call_id": tc["id"]
            })
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_messages.append({
                "role": "tool",
                "content": f"Error: {str(e)}",
                "tool_call_id": tc["id"]
            })

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages + [
        SystemMessage(content="Analyze and interpret these results:")
    ])
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    return tool_node(state)

# Define and compile the graph
analyst_graph = StateGraph(MessagesState)
analyst_graph.add_node("analyze_data", analyze_data)
analyst_graph.add_node("tools", tools_node)  # Use wrapped tools_node
analyst_graph.add_node("process_results", process_tool_results)
analyst_graph.set_entry_point("analyze_data")
analyst_graph.add_conditional_edges(
    "analyze_data",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
analyst_graph.add_edge("tools", "process_results")
analyst_graph.add_edge("process_results", "analyze_data")
analyst_graph = analyst_graph.compile()

__all__ = ["analyst_graph"]