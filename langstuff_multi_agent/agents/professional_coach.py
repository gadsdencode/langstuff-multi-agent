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
    has_tool_calls
)
from langstuff_multi_agent.config import get_llm

professional_coach_graph = StateGraph(MessagesState)

# Define the tools for professional coaching
tools = [search_web, job_search_tool]
tool_node = ToolNode(tools)


def coach(state):
    """Provide professional coaching and career advice."""
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
                    "Focus on career advice and job opportunities."
                )
            }
        ]
        return {"messages": [get_llm().invoke(prompt)]}
    return state


# Initialize and configure the professional coach graph
professional_coach_graph.add_node("coach", coach)
professional_coach_graph.add_node("tools", tool_node)
professional_coach_graph.add_node("process_results", process_tool_results)
professional_coach_graph.set_entry_point("coach")
professional_coach_graph.add_edge(START, "coach")

professional_coach_graph.add_conditional_edges(
    "coach",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

professional_coach_graph.add_edge("tools", "process_results")
professional_coach_graph.add_edge("process_results", END)

professional_coach_graph = professional_coach_graph.compile()

__all__ = ["professional_coach_graph"]
