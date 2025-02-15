# langstuff_multi_agent/agents/general_assistant.py
"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests
using a variety of tools. In this revised version, we mark the final assistant 
message with a 'final_answer' flag and (critically) remove any unconditional edge 
that forces the loop to continue, and we explicitly set the finish point to END.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather, has_tool_calls, news_tool
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langgraph.types import Command  # Use Command to control flow

# Initialize the graph with the MessagesState and configuration schema
general_assistant_graph = StateGraph(MessagesState, ConfigSchema)

# Define the available tools and create the ToolNode
tools = [search_web, get_current_weather, news_tool]
tool_node = ToolNode(tools)

def assist(state, config):
    """Provide general assistance with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    system_message = {
        "role": "system",
        "content": (
            "You are a General Assistant Agent. Your task is to assist with a variety of general queries and tasks.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Provide general information and answer questions.\n"
            "- get_current_weather: Retrieve current weather updates.\n"
            "- news_tool: Retrieve news headlines and articles.\n\n"
            "Instructions:\n"
            "1. Understand the user's request.\n"
            "2. Use the available tools to gather relevant information when needed.\n"
            "3. Provide clear, concise, and helpful responses."
        )
    }
    response = llm.invoke(state["messages"] + [system_message])
    return {"messages": [response]}

def process_tool_results(state, config):
    """Process tool outputs and return a final assistant response.

    If tool calls exist, this node aggregates their outputs, calls the LLM,
    marks the resulting assistant message as final, and returns a Command directing
    execution to END—thus stopping the loop.
    """
    last_message = state["messages"][-1]
    tool_outputs = []
    if tool_calls := getattr(last_message, 'tool_calls', None):
        for tc in tool_calls:
            try:
                output = f"Tool {tc['name']} result: {tc.get('output', '')}"
                tool_outputs.append({
                    "tool_call_id": tc["id"],
                    "output": output
                })
            except Exception as e:
                tool_outputs.append({
                    "tool_call_id": tc["id"],
                    "output": f"Tool execution failed: {str(e)}"
                })
        updated_messages = state["messages"] + [
            {
                "role": "tool",
                "content": to["output"],
                "tool_call_id": to["tool_call_id"]
            } for to in tool_outputs
        ]
        llm = get_llm(config.get("configurable", {}))
        final_response = llm.invoke(updated_messages)
        final_assistant_message = {
            "role": "assistant",
            "content": final_response.content,
            "additional_kwargs": {"final_answer": True}
        }
        # Return a Command that directs execution to END.
        return Command(goto=END, update={"messages": updated_messages + [final_assistant_message]})
    return state

# Add nodes to the graph
general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", tool_node)
general_assistant_graph.add_node("process_results", process_tool_results)
general_assistant_graph.set_entry_point("assist")
general_assistant_graph.add_edge(START, "assist")

def assist_edge_condition(state):
    msgs = state.get("messages", [])
    if not msgs:
        return "tools"
    last_msg = msgs[-1]
    if getattr(last_msg, "additional_kwargs", {}).get("final_answer", False):
        return END
    return "tools" if has_tool_calls(msgs) else END

general_assistant_graph.add_conditional_edges(
    "assist",
    assist_edge_condition,
    {"tools": "tools", END: END}
)

# IMPORTANT: Do NOT add any unconditional edge from "tools" (or process_results) back to "assist".

# Explicitly set the finish point to END so that when END is returned, the graph stops.
general_assistant_graph.set_finish_point(END)

general_assistant_graph = general_assistant_graph.compile()

__all__ = ["general_assistant_graph"]
