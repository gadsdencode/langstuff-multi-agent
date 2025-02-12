# langstuff_multi_agent/agents/financial_analyst.py
"""
Financial Analyst Agent module for analyzing market data, forecasting trends,
and providing investment insights.

This module provides a workflow for gathering financial news and data,
analyzing stock performance or economic indicators, and synthesizing a concise
summary with actionable investment insights.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    news_tool,
    calc_tool,
    has_tool_calls
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Create state graph for the financial analyst agent
financial_analyst_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the financial analyst.
# We assume that search_web and news_tool can be used to retrieve market data and news,
# and calc_tool can be used for any necessary calculations.
tools = [search_web, news_tool, calc_tool]
tool_node = ToolNode(tools)


def financial_analysis(state, config):
    """Conduct financial analysis with configuration support."""
    # Merge configuration from state and the passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a system prompt tailored for financial analysis
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Financial Analyst Agent. Your task is to analyze current market data, stock "
                            "performance, economic indicators, and forecast trends. You have access to the following tools:\n"
                            "- search_web: Use this tool to look up up-to-date financial news and data.\n"
                            "- news_tool: Retrieve the latest financial headlines and market insights.\n"
                            "- calc_tool: Perform any necessary calculations to support your analysis.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's query about market conditions or investment opportunities.\n"
                            "2. Use the available tools to gather accurate and relevant financial information.\n"
                            "3. Synthesize a clear, concise summary that highlights market trends and actionable insights."
                        )
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Process tool outputs and format the final financial analysis report."""
    # Check for handoff commands (if any)
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            for tc in msg.tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {
                        "messages": [ToolMessage(
                            goto=tc['name'].replace('transfer_to_', ''),
                            graph=ToolMessage.PARENT
                        )]
                    }
    # Collect tool outputs from ToolMessages
    tool_outputs = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                # Attempt to parse JSON responses if applicable
                content = msg.content.strip()
                if content and content[0] in ('{', '['):
                    data = json.loads(content)
                    # For financial news or data, we assume a list of items with "title" and "source"
                    if isinstance(data, list):
                        for item in data:
                            title = item.get('title', 'No title')
                            source = item.get('source', {}).get('name', 'Unknown')
                            tool_outputs.append(f"{title} ({source})")
                    else:
                        title = data.get('title', 'No title')
                        source = data.get('source', {}).get('name', 'Unknown')
                        tool_outputs.append(f"{title} ({source})")
                else:
                    # Fallback: use raw text split by newline (each line as a data point)
                    for line in content.split("\n"):
                        if line.strip():
                            tool_outputs.append(line.strip())
            except Exception as e:
                tool_outputs.append(f"Error processing financial data: {str(e)}")

    # Generate final summary only if we have outputs
    if tool_outputs:
        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Synthesize the following financial data into a concise analysis with key insights:"),
            HumanMessage(content="\n".join(tool_outputs))
        ])
        return {"messages": [summary]}

    # Fallback if no results
    return {
        "messages": [AIMessage(
            content="Could not retrieve current financial data. Please try again later."
        )]
    }


# Configure the state graph for the financial analyst agent
financial_analyst_graph.add_node("financial_analysis", financial_analysis)
financial_analyst_graph.add_node("tools", tool_node)
financial_analyst_graph.add_node("process_results", process_tool_results)
financial_analyst_graph.set_entry_point("financial_analysis")
financial_analyst_graph.add_edge(START, "financial_analysis")

financial_analyst_graph.add_conditional_edges(
    "financial_analysis",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

financial_analyst_graph.add_edge("tools", "process_results")
financial_analyst_graph.add_edge("process_results", "financial_analysis")

financial_analyst_graph = financial_analyst_graph.compile()

__all__ = ["financial_analyst_graph"]
