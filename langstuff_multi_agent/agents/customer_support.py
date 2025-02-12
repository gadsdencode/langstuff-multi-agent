# langstuff_multi_agent/agents/customer_support.py
"""
Customer Support Agent module for handling customer inquiries, troubleshooting, and FAQs.

This module provides a workflow for addressing common customer support issues.
It uses tools to search for support documentation and perform any necessary calculations.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, calc_tool, has_tool_calls
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage

customer_support_graph = StateGraph(MessagesState, ConfigSchema)

# Define tools for the Customer Support Agent
tools = [search_web, calc_tool]
tool_node = ToolNode(tools)


def support(state, config):
    """Conduct customer support interaction with configuration support."""
    # Merge state configuration with passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a tailored system prompt for customer support
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Customer Support Agent. Your task is to address customer inquiries, provide troubleshooting steps, and answer frequently asked questions.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up support documentation and FAQs.\n"
                            "- calc_tool: Perform calculations if needed.\n\n"
                            "Instructions:\n"
                            "1. Analyze the customer's query.\n"
                            "2. Use tools to retrieve accurate support information.\n"
                            "3. Provide a clear, concise response to help the customer."
                        )
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Processes tool outputs and formats the final customer support response."""
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


customer_support_graph.add_node("support", support)
customer_support_graph.add_node("tools", tool_node)
customer_support_graph.add_node("process_results", process_tool_results)
customer_support_graph.set_entry_point("support")
customer_support_graph.add_edge(START, "support")

customer_support_graph.add_conditional_edges(
    "support",
    lambda state: ("tools" if has_tool_calls(state.get("messages", [])) else "END"),
    {"tools": "tools", "END": END}
)

customer_support_graph.add_edge("tools", "process_results")
customer_support_graph.add_edge("process_results", "support")

customer_support_graph = customer_support_graph.compile()

__all__ = ["customer_support_graph"]
