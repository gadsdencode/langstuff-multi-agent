"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests using a variety of tools.
"""

from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, get_current_weather, news_tool
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage, ToolMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Helper function to convert dictionary messages to BaseMessage objects
def convert_message(msg):
    """Convert a single message to a BaseMessage object if it’s a dictionary."""
    if isinstance(msg, dict):
        msg_type = msg.get("type")
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant":
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
    return msg  # Return unchanged if already a BaseMessage

def convert_messages(messages):
    """Convert a list of messages to BaseMessage objects."""
    return [convert_message(msg) for msg in messages]

def assist(state: MessagesState, config: dict) -> dict:
    """Handle general assistance queries with tool support."""
    # Convert messages to BaseMessage objects at the start
    state["messages"] = convert_messages(state["messages"])
    logger.info(f"Assist input messages: {[type(m) for m in state['messages']]}")
    
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, get_current_weather, news_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
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
    
    # Invoke LLM and convert response to AIMessage
    response = llm_with_tools.invoke([system_prompt] + messages)
    if isinstance(response, dict):
        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])  # Keep as list of dictionaries
        response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    
    # Mark as final if no tool calls
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    
    logger.info(f"Assist returning: type={type(response)}, tool_calls={response.tool_calls}")
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Process tool outputs and generate a final response."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    logger.info(f"Process tool results input messages: {[type(m) for m in state['messages']]}")
    last_message = state["messages"][-1]
    
    # Extract tool calls (list of dicts) from the last message
    tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []
    if not tool_calls:
        return state
    
    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]
        tool = next(t for t in [search_web, get_current_weather, news_tool] if t.name == tool_name)
        output = tool.invoke(tool_args)
        tool_messages.append(ToolMessage(
            content=str(output),
            tool_call_id=tool_id,
            name=tool_name
        ))
    
    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    if isinstance(final_response, dict):
        content = final_response.get("content", "Task completed")
        tool_calls = final_response.get("tool_calls", [])  # Keep as list of dicts
        final_response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(final_response, AIMessage):
        final_response = AIMessage(content=str(final_response))
    
    final_response.additional_kwargs["final_answer"] = True
    logger.info(f"Process tool results returning messages: {len(state['messages'] + tool_messages + [final_response])}")
    return {"messages": state["messages"] + tool_messages + [final_response]}

def tools_node(state: MessagesState, config: dict) -> dict:
    """Wrapper for tool execution node."""
    state["messages"] = convert_messages(state["messages"])
    logger.info(f"Tools node input messages: {[type(m) for m in state['messages']]}")
    return tool_node(state)

# Build and compile the graph
general_assistant_graph = StateGraph(MessagesState)
general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", tools_node)
general_assistant_graph.add_node("process_results", process_tool_results)
general_assistant_graph.set_entry_point("assist")
general_assistant_graph.add_conditional_edges(
    "assist",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
general_assistant_graph.add_edge("tools", "process_results")
general_assistant_graph.add_edge("process_results", "assist")
general_assistant_graph = general_assistant_graph.compile()

__all__ = ["general_assistant_graph"]
