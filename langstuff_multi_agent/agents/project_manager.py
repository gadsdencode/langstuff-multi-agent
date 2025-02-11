# langstuff_multi_agent/agents/project_manager.py
"""
Project Manager Agent module for task and timeline management.

This module provides a workflow for overseeing project schedules
and coordinating tasks using various tools.
"""

from langgraph.graph import END, START, StateGraph
from typing import TypedDict, Dict, Any

from langstuff_multi_agent.utils.tools import get_tool_node, search_web, python_repl
from langstuff_multi_agent.config import get_llm
from langstuff_multi_agent.utils.tools import has_tool_calls


def manage(state):
    """Project management agent that coordinates tasks and timelines."""
    messages = state.get("messages", [])
    config = state.get("configurable", {})

    llm = get_llm(config)
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
                    "Focus on project management insights and next steps."
                )
            }
        ]
        return {"messages": [get_llm().invoke(prompt)]}
    return state


# Define state schema properly
class ProjectState(TypedDict):
    tasks: Dict[str, Any]
    current_step: str
    artifacts: Dict[str, Any]


def planning_node(state: ProjectState) -> dict:
    """Generates initial project plan"""
    return {"tasks": ["research", "prototype"], "current_step": "planning"}


def execution_node(state: ProjectState) -> dict:
    """Executes planned tasks"""
    return {"current_step": "executing", "artifacts": {"result": "prototype_v1"}}


# Correct initialization pattern from @Web examples
project_manager_graph = StateGraph(ProjectState)  # Pass state schema class

# Add ALL required nodes first
project_manager_graph.add_node("planning", planning_node)
project_manager_graph.add_node("execution", execution_node)
project_manager_graph.add_node("manage", manage)
project_manager_graph.add_node("tools", get_tool_node([search_web, python_repl]))
project_manager_graph.add_node("process_results", process_tool_results)

# Then define edges
project_manager_graph.add_edge("planning", "execution")
project_manager_graph.add_edge("execution", "manage")

# Conditional edges must point to REGISTERED nodes
project_manager_graph.add_conditional_edges(
    "manage",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else END,
    {"tools": "tools", "END": END}
)

project_manager_graph.add_edge("tools", "process_results")
project_manager_graph.add_edge("process_results", END)

# Set entry point AFTER all nodes exist
project_manager_graph.set_entry_point("planning")

project_manager_graph = project_manager_graph.compile()


__all__ = ["project_manager_graph"]
