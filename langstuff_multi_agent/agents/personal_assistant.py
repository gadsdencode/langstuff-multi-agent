"""
Personal Assistant Agent module for handling personal tasks and queries.

This module provides a workflow for assisting with scheduling, reminders, and general information using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, SearchWebInput, GetWeatherInput, CalendarInput, search_web, get_current_weather, calendar_tool
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage, ToolCall
from langchain_core.pydantic_v1 import BaseModel, ValidationError

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

system_prompt = SystemMessage(content=(
    "You are a Personal Assistant Agent. Your task is to help with a variety of personal tasks and queries, such as scheduling, reminders, and general information.\n\n"
    "You have access to the following tools:\n"
    "- search_web: Provide general information and answer questions.\n"
    "- get_current_weather: Retrieve current weather updates.\n"
    "- calendar_tool: Schedule events and manage your calendar.\n\n"
    "Instructions:\n"
    "1. Understand the user's request.\n"
    "2. Use the available tools to gather relevant information when needed.\n"
    "3. Provide clear, concise, and helpful responses to assist the user."
))

def assist(state: MessagesState, config: dict) -> dict:
    """Provide personal assistance with configuration support."""
    state["messages"] = convert_messages(state["messages"])
    logger.info(f"Assist input state: {state}")
    
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, get_current_weather, calendar_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
    response = llm_with_tools.invoke(messages + [system_prompt])
    
    if isinstance(response, dict):
        content = response.get("content", "")
        raw_tool_calls = response.get("tool_calls", [])  # List of dicts
        tool_calls = [ToolCall(**tc) for tc in raw_tool_calls]
        response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    
    logger.info(f"Returning response: {response}")
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    state["messages"] = convert_messages(state["messages"])
    last_message = state["messages"][-1]
    
    tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []
    if not tool_calls:
        return state
    
    tool_messages = []
    for tc in tool_calls:
        tool_name = tc.name
        if tool_name == "search_web":
            input_model = SearchWebInput
            tool_func = search_web
        elif tool_name == "get_current_weather":
            input_model = GetWeatherInput
            tool_func = get_current_weather
        elif tool_name == "calendar_tool":
            input_model = CalendarInput
            tool_func = calendar_tool
        else:
            logger.error(f"Unknown tool: {tool_name}")
            tool_messages.append(ToolMessage(
                content=f"Error: Tool '{tool_name}' not found",
                tool_call_id=tc.id,
                name=tool_name
            ))
            continue

        try:
            # Parse tc.args into the input model
            input_obj = input_model(**tc.args)
            # Call the tool function with the input model
            output = tool_func(input=input_obj)
            tool_messages.append(ToolMessage(
                content=str(output),
                tool_call_id=tc.id,
                name=tool_name
            ))
        except ValidationError as e:
            logger.error(f"Validation error for tool {tool_name}: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error: Invalid arguments for tool '{tool_name}': {str(e)}",
                tool_call_id=tc.id,
                name=tool_name
            ))
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tc.id,
                name=tool_name
            ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    
    if isinstance(final_response, dict):
        content = final_response.get("content", "Task completed")
        raw_tool_calls = final_response.get("tool_calls", [])  # List of dicts
        tool_calls = [ToolCall(**tc) for tc in raw_tool_calls]
        final_response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(final_response, AIMessage):
        final_response = AIMessage(content=str(final_response))
    
    final_response.additional_kwargs["final_answer"] = True
    return {"messages": state["messages"] + tool_messages + [final_response]}

def tools_node(state: MessagesState) -> dict:
    state["messages"] = convert_messages(state["messages"])
    return tool_node(state)

personal_assistant_graph = StateGraph(MessagesState)
personal_assistant_graph.add_node("assist", assist)
personal_assistant_graph.add_node("tools", tools_node)
personal_assistant_graph.add_node("process_results", process_tool_results)
personal_assistant_graph.set_entry_point("assist")
personal_assistant_graph.add_conditional_edges(
    "assist",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
personal_assistant_graph.add_edge("tools", "process_results")
personal_assistant_graph.add_edge("process_results", "assist")
personal_assistant_graph = personal_assistant_graph.compile()

__all__ = ["personal_assistant_graph"]
