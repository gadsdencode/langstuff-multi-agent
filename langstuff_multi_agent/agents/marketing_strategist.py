# langstuff_multi_agent/agents/marketing_strategist.py
"""
Marketing Strategist Agent module for analyzing trends, planning campaigns, and providing social media strategy insights.

This module provides a workflow for gathering market data, identifying trends, and delivering actionable marketing strategies.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, news_tool, calc_tool, has_tool_calls
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage

marketing_strategist_graph = StateGraph(MessagesState, ConfigSchema)

# Define tools for the Marketing Strategist Agent
tools = [search_web, news_tool, calc_tool]
tool_node = ToolNode(tools)


def marketing(state, config):
    """Conduct marketing strategy analysis with configuration support."""
    # Merge configuration from state and passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a tailored system prompt for marketing strategy
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Marketing Strategist Agent. Your task is to analyze current trends, plan marketing campaigns, and provide social media strategy insights.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Gather market and trend information.\n"
                            "- news_tool: Retrieve the latest news and social media trends.\n"
                            "- calc_tool: Perform quantitative analysis if needed.\n\n"
                            "Instructions:\n"
                            "1. Analyze the customer's marketing query.\n"
                            "2. Use tools to gather accurate market data and trend information.\n"
                            "3. Provide detailed, actionable marketing strategies and social media insights."
                        )
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Processes tool outputs and formats the final marketing strategy response."""
    # Check for handoff commands
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


marketing_strategist_graph.add_node("marketing", marketing)
marketing_strategist_graph.add_node("tools", tool_node)
marketing_strategist_graph.add_node("process_results", process_tool_results)
marketing_strategist_graph.set_entry_point("marketing")
marketing_strategist_graph.add_edge(START, "marketing")

marketing_strategist_graph.add_conditional_edges(
    "marketing",
    lambda state: ("tools" if has_tool_calls(state.get("messages", [])) else "END"),
    {"tools": "tools", "END": END}
)

marketing_strategist_graph.add_edge("tools", "process_results")
marketing_strategist_graph.add_edge("process_results", "marketing")

marketing_strategist_graph = marketing_strategist_graph.compile()

__all__ = ["marketing_strategist_graph"]
