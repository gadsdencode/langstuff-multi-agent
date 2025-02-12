# langstuff_multi_agent/agents/researcher.py
"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research 
information using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    news_tool,
    calc_tool,
    has_tool_calls,
    news_tool
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage

researcher_graph = StateGraph(MessagesState, ConfigSchema)

# Define research tools
tools = [search_web, news_tool, calc_tool, news_tool]
tool_node = ToolNode(tools)


def research(state, config):
    """Conduct research with configuration support."""
    # Get config from state and merge with passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Researcher Agent. Your task is to gather "
                            "and summarize news and research information.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up recent info and data.\n"
                            "- news_tool: Get latest news and articles.\n"
                            "- calc_tool: Perform calculations.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's research query.\n"
                            "2. Use tools to gather accurate and relevant info.\n"
                            "3. Provide a clear summary of your findings."
                        ),
                    }
                ]
            )
        ]
    }


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


researcher_graph.add_node("research", research)
researcher_graph.add_node("tools", tool_node)
researcher_graph.add_node("process_results", process_tool_results)
researcher_graph.set_entry_point("research")
researcher_graph.add_edge(START, "research")

researcher_graph.add_conditional_edges(
    "research",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

researcher_graph.add_edge("tools", "process_results")
researcher_graph.add_edge("process_results", "research")

researcher_graph = researcher_graph.compile()

__all__ = ["researcher_graph"]
