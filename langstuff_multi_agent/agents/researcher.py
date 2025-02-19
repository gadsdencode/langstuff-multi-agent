"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research information using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, news_tool, calc_tool, save_memory, search_memories
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Helper function to convert messages to BaseMessage objects
def convert_message(msg):
    if isinstance(msg, dict):
        msg_type = msg.get("type", msg.get("role"))  # Support both "type" and "role"
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant" or msg_type == "ai":
            return AIMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls", []))
        elif msg_type == "system":
            return SystemMessage(content=msg.get("content", ""))
        elif msg_type == "tool":
            return ToolMessage(
                content=msg.get("content", ""),
                tool_call_id=msg.get("tool_call_id", ""),
                name=msg.get("name", "")
            )
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    return msg

def convert_messages(messages):
    return [convert_message(msg) for msg in messages]

def research(state: MessagesState, config: dict) -> dict:
    """Conduct research with configuration support."""
    # Convert incoming messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, news_tool, calc_tool, save_memory, search_memories]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Researcher Agent. Your task is to gather and summarize news and research information.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Look up recent info and data.\n"
            "- news_tool: Get latest news and articles.\n"
            "- calc_tool: Perform calculations.\n"
            "- save_memory: Save information to memory.\n"
            "- search_memories: Search for information in memory.\n\n"
            "Instructions:\n"
            "1. Analyze the user's research query.\n"
            "2. Use tools to gather accurate and relevant info.\n"
            "3. Provide a clear summary of your findings."
        ))
    ])
    
    # Ensure response is an AIMessage
    if isinstance(response, dict):
        content = response.get("content", "")
        raw_tool_calls = response.get("tool_calls", [])
        tool_calls = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            if isinstance(tc, dict) else tc
            for tc in raw_tool_calls
        ]
        response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs with enhanced error handling."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []  # Changed from tool_outputs to align with standard naming
    try:
        for tc in last_message.tool_calls:
            tool = next(t for t in [search_web, news_tool, calc_tool, save_memory, search_memories] if t.name == tc["name"])
            result = tool.invoke(tc["args"], config=config if tc["name"] in ["save_memory", "search_memories"] else None)
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
                name=tc["name"]
            ))

        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke(state["messages"] + tool_messages + [
            SystemMessage(content="Synthesize these research findings:")
        ])
        
        # Ensure summary is an AIMessage
        if isinstance(summary, dict):
            content = summary.get("content", "Task completed")
            raw_tool_calls = summary.get("tool_calls", [])
            tool_calls = [
                {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in raw_tool_calls
            ]
            summary = AIMessage(content=content, tool_calls=tool_calls)
        elif not isinstance(summary, AIMessage):
            summary = AIMessage(content=str(summary))
        
        summary.additional_kwargs["final_answer"] = True
        return {"messages": state["messages"] + tool_messages + [summary]}

    except Exception as e:
        error_message = AIMessage(content=f"Error processing research: {str(e)}", additional_kwargs={"final_answer": True})
        return {"messages": state["messages"] + [error_message]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    state["messages"] = convert_messages(state["messages"])
    return tool_node(state)

# Define and compile the graph
researcher_graph = StateGraph(MessagesState)
researcher_graph.add_node("research", research)
researcher_graph.add_node("tools", tools_node)
researcher_graph.add_node("process_results", process_tool_results)
researcher_graph.set_entry_point("research")
researcher_graph.add_conditional_edges(
    "research",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
researcher_graph.add_edge("tools", "process_results")
researcher_graph.add_edge("process_results", "research")
researcher_graph = researcher_graph.compile()

__all__ = ["researcher_graph"]