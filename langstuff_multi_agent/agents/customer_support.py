"""
Customer Support Agent module for handling customer inquiries, troubleshooting, and FAQs.

This module provides a workflow for addressing common customer support issues using tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm

customer_support_graph = StateGraph(MessagesState, ConfigSchema)


def support(state, config):
    """Conduct customer support interaction with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        {
            "role": "system",
            "content": (
                "You are a Customer Support Agent. Your task is to address customer inquiries, provide troubleshooting steps, and answer FAQs.\n\n"
                "You have access to the following tools:\n"
                "- search_web: Look up support documentation and FAQs.\n"
                "- calc_tool: Perform calculations if needed.\n\n"
                "Instructions:\n"
                "1. Analyze the customer's query.\n"
                "2. Use tools to retrieve accurate support information.\n"
                "3. Provide a clear, concise response to help the customer."
            )
        }
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}


def process_tool_results(state, config):
    """Processes tool outputs and formats the final response."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, calc_tool] if t.name == tc["name"])
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


customer_support_graph.add_node("support", support)
customer_support_graph.add_node("tools", tool_node)
customer_support_graph.add_node("process_results", process_tool_results)
customer_support_graph.set_entry_point("support")
customer_support_graph.add_conditional_edges(
    "support",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
customer_support_graph.add_edge("tools", "process_results")
customer_support_graph.add_edge("process_results", "support")
customer_support_graph = customer_support_graph.compile()

__all__ = ["customer_support_graph"]
