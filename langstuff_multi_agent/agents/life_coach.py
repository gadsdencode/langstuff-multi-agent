"""
Life Coach Agent module for personal advice and guidance.

This module provides a workflow for offering lifestyle tips and personal development advice using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, get_current_weather, calendar_tool, save_memory, search_memories
from langstuff_multi_agent.config import ConfigSchema, get_llm

life_coach_graph = StateGraph(MessagesState, ConfigSchema)


def life_coach(state, config):
    """Provide life coaching and personal advice."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, get_current_weather, calendar_tool, save_memory, search_memories]
    llm = llm.bind_tools(tools)
    messages = state["messages"]
    if "user_id" in config.get("configurable", {}):
        preferences = search_memories.invoke("lifestyle preferences", config)
        if preferences:
            messages.append({"role": "system", "content": f"User preferences: {preferences}"})
    response = llm.invoke(messages + [
        {
            "role": "system",
            "content": (
                "You are a Life Coach Agent. Your task is to offer lifestyle tips and personal development advice.\n"
                "You have access to the following tools:\n"
                "- search_web: Find advice and resources.\n"
                "- get_current_weather: Provide weather-based suggestions.\n"
                "- calendar_tool: Schedule activities.\n"
                "- save_memory: Save user preferences.\n"
                "- search_memories: Retrieve past preferences.\n\n"
                "Instructions:\n"
                "1. Analyze the user's request.\n"
                "2. Use tools to provide tailored advice.\n"
                "3. Offer actionable suggestions."
            )
        }
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}


def process_tool_results(state, config):
    """Processes tool outputs and formats final response."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, get_current_weather, calendar_tool, save_memory, search_memories] if t.name == tc["name"])
        output = tool.invoke(tc["args"], config=config if tc["name"] in ["save_memory", "search_memories"] else None)
        tool_messages.append({
            "role": "tool",
            "content": output,
            "tool_call_id": tc["id"]
        })

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}


life_coach_graph.add_node("life_coach", life_coach)
life_coach_graph.add_node("tools", tool_node)
life_coach_graph.add_node("process_results", process_tool_results)
life_coach_graph.set_entry_point("life_coach")
life_coach_graph.add_conditional_edges(
    "life_coach",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
life_coach_graph.add_edge("tools", "process_results")
life_coach_graph.add_edge("process_results", "life_coach")
life_coach_graph = life_coach_graph.compile()

__all__ = ["life_coach_graph"]
