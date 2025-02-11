# langstuff_multi_agent/agents/life_coach.py
"""
Life Coach Agent module for personal advice and guidance.

This module provides a workflow for offering lifestyle tips and
personal development advice using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    get_current_weather,
    calendar_tool,
    has_tool_calls
)
from langstuff_multi_agent.config import get_llm

life_coach_graph = StateGraph(MessagesState)

# Define tools for life coaching
tools = [search_web, get_current_weather, calendar_tool]
tool_node = ToolNode(tools)


def life_coach(state):
    """Provide life coaching and personal advice."""
    messages = state.get("messages", [])
    config = state.get("config", {})
    
    llm = get_llm(config.get("configurable", {}))
    response = llm.invoke(messages)
    
    return {"messages": messages + [response]}


def process_tool_results(state):
    """Processes tool outputs and formats FINAL user response"""
    last_message = state.messages[-1]
    
    if tool_calls := getattr(last_message, 'tool_calls', None):
        outputs = [tc["output"] for tc in tool_calls if "output" in tc]
        
        # Generate FINAL response with tool data
        prompt = [
            {"role": "user", "content": state.messages[0].content},
            {"role": "assistant", "content": f"Tool outputs: {outputs}"},
            {
                "role": "system", 
                "content": (
                    "Formulate final answer using these results. "
                    "Focus on personal advice and actionable steps."
                )
            }
        ]
        return {"messages": [get_llm().invoke(prompt)]}
    return state


# Initialize and configure the life coach graph
life_coach_graph.add_node("life_coach", life_coach)
life_coach_graph.add_node("tools", tool_node)
life_coach_graph.add_node("process_results", process_tool_results)
life_coach_graph.set_entry_point("life_coach")
life_coach_graph.add_edge(START, "life_coach")

life_coach_graph.add_conditional_edges(
    "life_coach",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

life_coach_graph.add_edge("tools", "process_results")
life_coach_graph.add_edge("process_results", END)

life_coach_graph = life_coach_graph.compile()

__all__ = ["life_coach_graph"]
