"""
Creative Content Agent module for generating creative writing, marketing copy, social media posts, or brainstorming ideas.

This module provides a workflow for generating creative content using various tools and a creative prompt.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, calc_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import AIMessage, SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def creative_content(state: MessagesState, config: dict) -> dict:
    """Generate creative content based on the user's query with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, calc_tool]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Creative Content Agent. Your task is to generate creative writing, marketing copy, "
            "social media posts, or brainstorming ideas. Use vivid, imaginative, and engaging language.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Look up trends or inspiration from online sources.\n"
            "- calc_tool: Perform calculations if needed (secondary in this role).\n\n"
            "Instructions:\n"
            "1. Analyze the user's creative query.\n"
            "2. Draw upon your creative instincts (and tool data if helpful) to generate an inspiring draft.\n"
            "3. Produce a final piece of creative content that addresses the query."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Process tool outputs and integrate them into a final creative content draft."""
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
    final_response = llm.invoke(state["messages"] + tool_messages + [
        SystemMessage(content="Synthesize the following inspirations into a creative draft:")
    ])
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    return tool_node(state)

# Define and compile the graph
creative_content_graph = StateGraph(MessagesState)
creative_content_graph.add_node("creative_content", creative_content)
creative_content_graph.add_node("tools", tools_node)  # Use wrapped tools_node
creative_content_graph.add_node("process_results", process_tool_results)
creative_content_graph.set_entry_point("creative_content")
creative_content_graph.add_conditional_edges(
    "creative_content",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
creative_content_graph.add_edge("tools", "process_results")
creative_content_graph.add_edge("process_results", "creative_content")
creative_content_graph = creative_content_graph.compile()

__all__ = ["creative_content_graph"]