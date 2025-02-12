# langstuff_multi_agent/agents/professional_coach.py
"""
Professional Coach Agent module for career guidance.

This module provides a workflow for offering career advice and
job search strategies using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    job_search_tool,
    has_tool_calls,
    get_current_weather,
    calendar_tool,
    save_memory,
    search_memories
)
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import ToolMessage

professional_coach_graph = StateGraph(MessagesState)

# Define the tools for professional coaching
tools = [search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories]
tool_node = ToolNode(tools)


def coach(state):
    """Provide professional coaching and career advice."""
    messages = state.get("messages", [])
    config = state.get("config", {})

    llm = get_llm(config.get("configurable", {}))
    response = llm.invoke(messages)

    # Add memory context
    if 'user_id' in state.get("configurable", {}):
        career_goals = search_memories.invoke(
            "career goals", 
            {"configurable": state["configurable"]}
        )
        if career_goals:
            messages.append({
                "role": "system",
                "content": f"User career history: {career_goals}"
            })

    return {"messages": messages + [response]}


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


# Initialize and configure the professional coach graph
professional_coach_graph.add_node("coach", coach)
professional_coach_graph.add_node("tools", tool_node)
professional_coach_graph.add_node("process_results", process_tool_results)
professional_coach_graph.set_entry_point("coach")
professional_coach_graph.add_edge(START, "coach")

professional_coach_graph.add_conditional_edges(
    "coach",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

professional_coach_graph.add_edge("tools", "process_results")
professional_coach_graph.add_edge("process_results", "coach")

professional_coach_graph = professional_coach_graph.compile()

__all__ = ["professional_coach_graph"]
