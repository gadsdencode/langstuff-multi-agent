"""
Personal Assistant Agent module for handling personal tasks and queries.

This module provides a workflow for assisting with scheduling, reminders, and general information using various tools.
"""

import logging
import json
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    BaseMessage
)
from langstuff_multi_agent.utils.tools import (
    tool_node,
    has_tool_calls,
    SearchWebInput,
    GetWeatherInput,
    CalendarInput,
    search_web,
    get_current_weather,
    calendar_tool
)
from langstuff_multi_agent.config import get_llm
from langchain_core.pydantic_v1 import BaseModel, ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_message(msg):
    """Convert a message dict or object to a proper BaseMessage object."""
    if isinstance(msg, BaseMessage):
        # If it's already a BaseMessage, ensure it has a valid type
        msg_type = getattr(msg, "type", None)
        if msg_type in ("human", "user", "ai", "assistant", "function", "tool", "system", "developer"):
            return msg
        # If type is invalid, convert to appropriate type based on content
        content = str(getattr(msg, "content", ""))
        kwargs = getattr(msg, "additional_kwargs", {})
        return HumanMessage(content=content, additional_kwargs=kwargs)
        
    if isinstance(msg, dict):
        # Get message type, defaulting to "human" for user messages
        msg_type = str(msg.get("type", msg.get("role", "human")))
        content = str(msg.get("content", ""))
        kwargs = msg.get("additional_kwargs", {})
        
        # Map message types to LangChain types
        if msg_type in ("human", "user"):
            return HumanMessage(content=content, additional_kwargs=kwargs)
        elif msg_type in ("ai", "assistant"):
            # Ensure tool_calls is a list
            tool_calls = list(msg.get("tool_calls", []))
            return AIMessage(content=content, tool_calls=tool_calls, additional_kwargs=kwargs)
        elif msg_type == "system":
            return SystemMessage(content=content, additional_kwargs=kwargs)
        elif msg_type in ("function", "tool"):
            return ToolMessage(
                content=content,
                tool_call_id=str(msg.get("tool_call_id", "")),
                name=str(msg.get("name", "")),
                additional_kwargs=kwargs
            )
        elif msg_type == "developer":
            return SystemMessage(content=content, additional_kwargs=kwargs)
        else:
            # Default to HumanMessage for unknown types
            return HumanMessage(content=content, additional_kwargs=kwargs)
    
    # If it's a string or other type, treat as human message content
    return HumanMessage(content=str(msg))


def convert_messages(messages):
    """Convert a list of messages to proper BaseMessage objects."""
    if not messages:
        return []
    # Ensure each message is properly converted
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
    """Process user input and provide assistance."""
    try:
        # Convert state to list of proper message objects
        messages = convert_messages(state.get("messages", []))
        
        # Convert config to serializable form
        config_dict = dict(config) if hasattr(config, "dict") else {}
        configurable = config_dict.get("configurable", {})
        
        # Get LLM with config
        llm = get_llm(configurable)
        tools = [search_web, get_current_weather, calendar_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        # Create input state
        input_messages = messages + [system_prompt]
        
        # Get response
        response = llm_with_tools.invoke(input_messages)
        
        # Ensure response is properly formatted
        if isinstance(response, dict):
            content = str(response.get("content", ""))
            response = AIMessage(
                content=content,
                tool_calls=[],
                additional_kwargs={"final_answer": True},
                type="assistant"  # Explicitly set type
            )
        elif not isinstance(response, AIMessage):
            response = AIMessage(
                content=str(response),
                tool_calls=[],
                additional_kwargs={"final_answer": True},
                type="assistant"  # Explicitly set type
            )
        else:
            response.type = "assistant"  # Ensure type is set
            response.tool_calls = []
            response.additional_kwargs = {"final_answer": True}
        
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"Error in assist: {str(e)}"
        logger.error(error_msg)
        return {
            "messages": [
                SystemMessage(
                    content=error_msg,
                    additional_kwargs={"final_answer": True}
                )
            ]
        }


def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Process the results of tool calls."""
    try:
        # Convert state to list of proper message objects
        messages = convert_messages(state.get("messages", []))
        if not messages:
            return {"messages": []}
        
        # Get tool calls from last message
        last_message = messages[-1]
        tool_calls = []
        if isinstance(last_message, AIMessage):
            # Convert tool calls to proper format
            tool_calls = list(getattr(last_message, "tool_calls", []))
        
        if not tool_calls:
            return {"messages": messages}

        # Process tool calls
        tool_messages = []
        for tc in tool_calls:
            try:
                tool_name = str(tc.name)
                tool_id = str(tc.id)
                tool_args = dict(tc.args)
                
                # Get appropriate tool
                if tool_name == "search_web":
                    input_model, tool_func = SearchWebInput, search_web
                elif tool_name == "get_current_weather":
                    input_model, tool_func = GetWeatherInput, get_current_weather
                elif tool_name == "calendar_tool":
                    input_model, tool_func = CalendarInput, calendar_tool
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    logger.error(error_msg)
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id,
                            name=tool_name,
                            additional_kwargs={"error": True},
                            type="tool"  # Explicitly set type
                        )
                    )
                    continue

                # Execute tool
                input_obj = input_model(**tool_args)
                output = tool_func(input=input_obj)
                
                # Create tool message
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tool_id,
                        name=tool_name,
                        additional_kwargs={},
                        type="tool"  # Explicitly set type
                    )
                )
            except Exception as e:
                error_msg = f"Error with tool {tool_name}: {str(e)}"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_id,
                        name=tool_name,
                        additional_kwargs={"error": True},
                        type="tool"  # Explicitly set type
                    )
                )

        # Convert config to serializable form
        config_dict = dict(config) if hasattr(config, "dict") else {}
        configurable = config_dict.get("configurable", {})
        
        # Get LLM with config
        llm = get_llm(configurable)
        
        # Create input state
        input_messages = messages + tool_messages
        
        # Get final response
        final_response = llm.invoke(input_messages)
        
        # Ensure response is properly formatted
        if isinstance(final_response, dict):
            content = str(final_response.get("content", "Task completed"))
            final_response = AIMessage(
                content=content,
                tool_calls=[],
                additional_kwargs={"final_answer": True},
                type="assistant"  # Explicitly set type
            )
        elif not isinstance(final_response, AIMessage):
            final_response = AIMessage(
                content=str(final_response),
                tool_calls=[],
                additional_kwargs={"final_answer": True},
                type="assistant"  # Explicitly set type
            )
        else:
            final_response.type = "assistant"  # Ensure type is set
            final_response.tool_calls = []
            final_response.additional_kwargs = {"final_answer": True}
        
        return {"messages": messages + tool_messages + [final_response]}
        
    except Exception as e:
        error_msg = f"Error processing tool results: {str(e)}"
        logger.error(error_msg)
        return {
            "messages": messages + [
                SystemMessage(
                    content=error_msg,
                    additional_kwargs={"final_answer": True},
                    type="system"  # Explicitly set type
                )
            ]
        }


def tools_node(state: MessagesState) -> dict:
    """Process tool calls in state."""
    messages = tuple(convert_messages(state.get("messages", [])))
    return tool_node({"messages": list(messages)})


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
