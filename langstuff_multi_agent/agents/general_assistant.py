# langstuff_multi_agent/agents/general_assistant.py
"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests
using a variety of tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

general_assistant_graph = StateGraph(MessagesState, ConfigSchema)

# Define general assistant tools
tools = [search_web, get_current_weather]
tool_node = ToolNode(tools)


def assist(state, config):
    """Provide general assistance with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a General Assistant Agent. Your task is to assist with a variety of general queries and tasks.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Provide general information and answer questions.\n"
                            "- get_current_weather: Retrieve current weather updates.\n\n"
                            "Instructions:\n"
                            "1. Understand the user's request.\n"
                            "2. Use the available tools to gather relevant information when needed.\n"
                            "3. Provide clear, concise, and helpful responses to assist the user."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state):
    """Processes tool outputs and formats final response"""
    last_message = state.messages[-1]

    if tool_calls := getattr(last_message, 'tool_calls', None):
        # Extract tool outputs
        tool_outputs = [tc["output"] for tc in tool_calls if "output" in tc]

        # Generate final response with context
        return {"messages": [
            get_llm().invoke(state.messages + [{
                "role": "user",
                "content": f"Tool outputs: {tool_outputs}\nFormulate final answer:"
            }])
        ]}
    return state


general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", tool_node)
general_assistant_graph.add_node("process_results", process_tool_results)
general_assistant_graph.set_entry_point("assist")
general_assistant_graph.add_edge(START, "assist")

general_assistant_graph.add_conditional_edges(
    "assist",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

general_assistant_graph.add_edge("tools", "process_results")
general_assistant_graph.add_edge("process_results", END)

general_assistant_graph = general_assistant_graph.compile()

__all__ = ["general_assistant_graph"]
