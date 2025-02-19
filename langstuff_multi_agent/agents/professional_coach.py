"""
Professional Coach Agent module for career guidance.

This module provides a workflow for offering career advice and job search strategies using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

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

def coach(state: MessagesState, config: dict) -> dict:
    """Provide professional coaching and career advice."""
    # Convert incoming messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    messages = state["messages"]
    
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories]
    llm_with_tools = llm.bind_tools(tools)
    
    if "user_id" in config.get("configurable", {}):
        career_goals = search_memories.invoke("career goals", config)
        if career_goals:
            messages.append(SystemMessage(content=f"User career history: {career_goals}"))
    
    response = llm_with_tools.invoke(messages + [
        SystemMessage(content=(
            "You are a Professional Coach Agent. Your task is to offer career advice and job search strategies.\n"
            "Use tools to provide relevant information and schedule events as needed."
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
    """Processes tool outputs and formats final response."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, job_search_tool, get_current_weather, calendar_tool, save_memory, search_memories] if t.name == tc["name"])
        output = tool.invoke(tc["args"], config=config if tc["name"] in ["save_memory", "search_memories"] else None)
        tool_messages.append(ToolMessage(
            content=str(output),
            tool_call_id=tc["id"],
            name=tc["name"]
        ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    
    # Ensure final_response is an AIMessage
    if isinstance(final_response, dict):
        content = final_response.get("content", "Task completed")
        raw_tool_calls = final_response.get("tool_calls", [])
        tool_calls = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in raw_tool_calls
        ]
        final_response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(final_response, AIMessage):
        final_response = AIMessage(content=str(final_response))
    
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    state["messages"] = convert_messages(state["messages"])
    return tool_node(state)

# Define and compile the graph
professional_coach_graph = StateGraph(MessagesState)
professional_coach_graph.add_node("coach", coach)
professional_coach_graph.add_node("tools", tools_node)
professional_coach_graph.add_node("process_results", process_tool_results)
professional_coach_graph.set_entry_point("coach")
professional_coach_graph.add_conditional_edges(
    "coach",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
professional_coach_graph.add_edge("tools", "process_results")
professional_coach_graph.add_edge("process_results", "coach")
professional_coach_graph = professional_coach_graph.compile()

__all__ = ["professional_coach_graph"]