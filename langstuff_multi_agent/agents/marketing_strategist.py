"""
Marketing Strategist Agent module for analyzing trends, planning campaigns, and providing social media strategy insights.

This module provides a workflow for gathering market data, identifying trends, and delivering actionable marketing strategies.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, news_tool, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def marketing(state: MessagesState, config: dict) -> dict:
    """Conduct marketing strategy analysis with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, news_tool, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Marketing Strategist Agent. Your task is to analyze current trends, plan marketing campaigns, and provide social media strategy insights.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Gather market and trend information.\n"
            "- news_tool: Retrieve the latest news and social media trends.\n"
            "- calc_tool: Perform quantitative analysis if needed.\n\n"
            "Instructions:\n"
            "1. Analyze the customer's marketing query.\n"
            "2. Use tools to gather accurate market data and trend information.\n"
            "3. Provide detailed, actionable marketing strategies and social media insights."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs and formats the final marketing strategy response."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, news_tool, calc_tool] if t.name == tc["name"])
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
marketing_strategist_graph = StateGraph(MessagesState)
marketing_strategist_graph.add_node("marketing", marketing)
marketing_strategist_graph.add_node("tools", tools_node)  # Use wrapped tools_node
marketing_strategist_graph.add_node("process_results", process_tool_results)
marketing_strategist_graph.set_entry_point("marketing")
marketing_strategist_graph.add_conditional_edges(
    "marketing",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
marketing_strategist_graph.add_edge("tools", "process_results")
marketing_strategist_graph.add_edge("process_results", "marketing")
marketing_strategist_graph = marketing_strategist_graph.compile()

__all__ = ["marketing_strategist_graph"]