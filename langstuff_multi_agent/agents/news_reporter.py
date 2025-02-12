# langstuff_multi_agent/agents/news_reporter.py
"""
News Reporter Agent module for gathering and summarizing news reports.

This module provides a workflow for gathering and reporting the latest news using various tools.
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
from langchain_core.messages import ToolMessage

# Create state graph for the news reporter agent
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the news reporter
tools = [search_web, news_tool, calc_tool, news_tool]
tool_node = ToolNode(tools)


def news_report(state, config):
    """Conduct news reporting with configuration support."""
    # Merge the configuration from the state and the passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a system prompt tailored for a news reporter agent
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a News Reporter Agent. Your task is to gather and report "
                            "the latest news, headlines, and summaries from reliable sources.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up recent info and data.\n"
                            "- news_tool: Retrieve the latest news articles and headlines.\n"
                            "- calc_tool: Perform calculations if necessary.\n"
                            "- news_tool: Conduct specialized news searches.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's news query.\n"
                            "2. Use the available tools to gather accurate and up-to-date news.\n"
                            "3. Provide a clear and concise summary of your findings."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Process tool outputs and format the final news report for the user."""
    # Check for handoff commands to other agents
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


# Configure the state graph for the news reporter agent
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.add_node("tools", tool_node)
news_reporter_graph.add_node("process_results", process_tool_results)
news_reporter_graph.set_entry_point("news_report")
news_reporter_graph.add_edge(START, "news_report")

news_reporter_graph.add_conditional_edges(
    "news_report",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

news_reporter_graph.add_edge("tools", "process_results")
news_reporter_graph.add_edge("process_results", "news_report")

news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]
