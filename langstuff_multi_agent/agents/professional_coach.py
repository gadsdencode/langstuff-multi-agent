"""
Professional Coach Agent module for career guidance.

This module provides a workflow for offering career advice and job search strategies using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage

professional_coach_graph = StateGraph(MessagesState, ConfigSchema)


def coach(state, config):
    """Provide professional coaching and career advice."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories]
    llm = llm.bind_tools(tools)
    messages = state["messages"]
    if "user_id" in config.get("configurable", {}):
        career_goals = search_memories.invoke("career goals", config)
        if career_goals:
            messages.append({"role": "system", "content": f"User career history: {career_goals}"})
    response = llm.invoke(messages + [
        {
            "role": "system",
            "content": (
                "You are a Professional Coach Agent. Your task is to offer career advice and job search strategies.\n"
                "Use tools to provide relevant information and schedule events as needed."
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
        tool = next(t for t in [search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories] if t.name == tc["name"])
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


professional_coach_graph.add_node("coach", coach)
professional_coach_graph.add_node("tools", tool_node)
professional_coach_graph.add_node("process_results", process_tool_results)
professional_coach_graph.set_entry_point("coach")
professional_coach_graph.add_conditional_edges(
    "coach",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
professional_coach_graph.add_edge("tools", "process_results")
professional_coach_graph.add_edge("process_results", "coach")

professional_coach_graph = professional_coach_graph.compile()

__all__ = ["professional_coach_graph"]
