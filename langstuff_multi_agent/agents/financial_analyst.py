"""
Financial Analyst Agent module for analyzing market data, forecasting trends, and providing investment insights.

This module provides a workflow for gathering financial news and data, analyzing stock performance or economic indicators, and synthesizing a concise summary with actionable investment insights.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, news_tool, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

financial_analyst_graph = StateGraph(MessagesState, ConfigSchema)


def financial_analysis(state, config):
    """Conduct financial analysis with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, news_tool, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        {
            "role": "system",
            "content": (
                "You are a Financial Analyst Agent. Your task is to analyze current market data, stock performance, economic indicators, and forecast trends.\n"
                "You have access to the following tools:\n"
                "- search_web: Look up up-to-date financial news and data.\n"
                "- news_tool: Retrieve the latest financial headlines and market insights.\n"
                "- calc_tool: Perform any necessary calculations.\n\n"
                "Instructions:\n"
                "1. Analyze the user's query about market conditions or investment opportunities.\n"
                "2. Use tools to gather accurate and relevant financial information.\n"
                "3. Synthesize a clear, concise summary with actionable insights."
            )
        }
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}


def process_tool_results(state, config):
    """Process tool outputs and format the final financial analysis report."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, news_tool, calc_tool] if t.name == tc["name"])
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
        SystemMessage(content="Synthesize the financial data into a concise analysis with key insights:")
    ])
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}


financial_analyst_graph.add_node("financial_analysis", financial_analysis)
financial_analyst_graph.add_node("tools", tool_node)
financial_analyst_graph.add_node("process_results", process_tool_results)
financial_analyst_graph.set_entry_point("financial_analysis")
financial_analyst_graph.add_conditional_edges(
    "financial_analysis",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
financial_analyst_graph.add_edge("tools", "process_results")
financial_analyst_graph.add_edge("process_results", "financial_analysis")
financial_analyst_graph = financial_analyst_graph.compile()

__all__ = ["financial_analyst_graph"]
