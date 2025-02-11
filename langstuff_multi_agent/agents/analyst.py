# langstuff_multi_agent/agents/analyst.py
"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing
calculations using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    python_repl,
    calc_tool,
    has_tool_calls
)
from langstuff_multi_agent.config import get_llm

analyst_graph = StateGraph(MessagesState)

# Define tools for analysis tasks
tools = [search_web, python_repl, calc_tool]
tool_node = ToolNode(tools)


def analyze_data(state):
    """Analyze data and perform calculations."""
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
                    "Focus on data analysis insights and calculations."
                )
            }
        ]
        return {"messages": [get_llm().invoke(prompt)]}
    return state


# Initialize and configure the analyst graph
analyst_graph.add_node("analyze_data", analyze_data)
analyst_graph.add_node("tools", tool_node)
analyst_graph.add_node("process_results", process_tool_results)
analyst_graph.set_entry_point("analyze_data")
analyst_graph.add_edge(START, "analyze_data")

analyst_graph.add_conditional_edges(
    "analyze_data",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

analyst_graph.add_edge("tools", "process_results")
analyst_graph.add_edge("process_results", END)

analyst_graph = analyst_graph.compile()

__all__ = ["analyst_graph"]
