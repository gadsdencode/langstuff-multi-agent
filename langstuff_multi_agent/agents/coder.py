# langstuff_multi_agent/agents/coder.py
"""
Coder Agent module for writing and improving code.

This module provides a workflow for code generation, debugging,
and optimization using various development tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, python_repl, read_file, write_file, calc_tool, has_tool_calls
from langstuff_multi_agent.config import ConfigSchema, get_llm

coder_graph = StateGraph(MessagesState, ConfigSchema)

# Define tools for coding tasks.
tools = [search_web, python_repl, read_file, write_file, calc_tool]
tool_node = ToolNode(tools)


def code(state, config):
    """Write and improve code with configuration support."""
    # Get config from state and merge with passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)

    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Coder Agent. Your task is to write, debug, and improve code.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Find coding examples and documentation.\n"
                            "- python_repl: Execute and test Python code snippets.\n"
                            "- read_file: Retrieve code from files.\n"
                            "- write_file: Save code modifications to files.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's code or coding request.\n"
                            "2. Provide solutions, test code, and explain your reasoning.\n"
                            "3. Use the available tools to execute code and verify fixes as necessary."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state):
    """Processes tool outputs and formats FINAL user response"""
    last_message = state.messages[-1]
    
    if tool_calls := getattr(last_message, 'tool_calls', None):
        tool_outputs = [tc["output"] for tc in tool_calls if "output" in tc]
        
        # Generate FINAL response with tool data
        return {"messages": [
            get_llm().invoke([
                {"role": "user", "content": state.messages[0].content},
                {"role": "assistant", "content": f"Tool outputs: {tool_outputs}"},
                {"role": "system", "content": (
                    "Formulate final answer using these results. "
                    "Include code explanations and next steps if relevant."
                )}
            ])
        ]}
    return state


coder_graph.add_node("code", code)
coder_graph.add_node("tools", tool_node)
coder_graph.add_node("process_results", process_tool_results)
coder_graph.set_entry_point("code")
coder_graph.add_edge(START, "code")

coder_graph.add_conditional_edges(
    "code",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

coder_graph.add_edge("tools", "process_results")
coder_graph.add_edge("process_results", END)

coder_graph = coder_graph.compile()

__all__ = ["coder_graph"]
