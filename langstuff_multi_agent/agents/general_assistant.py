"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests using a variety of tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, get_current_weather, news_tool
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

general_assistant_graph = StateGraph(MessagesState, ConfigSchema)

def assist(state, config):
    """Provide general assistance with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, get_current_weather, news_tool]
    llm = llm.bind_tools(tools)
    
    # Convert raw dict messages to BaseMessage objects
    messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            role = msg.get("type") or msg.get("role", "human")  # Handle 'type' from Studio
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "tool":
                messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", "")
                ))
            else:
                messages.append(HumanMessage(content=content))  # Fallback
        elif isinstance(msg, BaseMessage):
            messages.append(msg)
        else:
            messages.append(HumanMessage(content=str(msg)))  # Fallback
    
    system_prompt = SystemMessage(content=(
        "You are a General Assistant Agent. Your task is to assist with a variety of general queries and tasks.\n\n"
        "You have access to the following tools:\n"
        "- search_web: Provide general information and answer questions.\n"
        "- get_current_weather: Retrieve current weather updates.\n"
        "- news_tool: Retrieve news headlines and articles.\n\n"
        "Instructions:\n"
        "1. Understand the user's request.\n"
        "2. Use the available tools to gather relevant information when needed.\n"
        "3. Provide clear, concise, and helpful responses to assist the user."
    ))
    
    response = llm.invoke(messages + [system_prompt])
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
        tool = next(t for t in [search_web, get_current_weather, news_tool] if t.name == tc["name"])
        output = tool.invoke(tc["args"])
        tool_messages.append({
            "role": "tool",
            "content": output,
            "tool_call_id": tc["id"]
        })
    
    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", tool_node)
general_assistant_graph.add_node("process_results", process_tool_results)
general_assistant_graph.set_entry_point("assist")
general_assistant_graph.add_conditional_edges(
    "assist",
    lambda state: "tools" if has_tool_calls(state["messages"]) else "END",
    {"tools": "tools", "END": END}
)
general_assistant_graph.add_edge("tools", "process_results")
general_assistant_graph.add_edge("process_results", "assist")
general_assistant_graph = general_assistant_graph.compile()

__all__ = ["general_assistant_graph"]