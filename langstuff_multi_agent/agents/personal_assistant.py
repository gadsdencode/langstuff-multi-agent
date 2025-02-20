"""
Personal Assistant Agent module for handling personal tasks and queries.

This module provides a workflow for assisting with scheduling, reminders, and general information using various tools.
"""

import logging
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
        return msg
        
    if isinstance(msg, dict):
        # Convert dict to hashable form
        msg_type = str(msg.get("type", msg.get("role", "")))
        content = str(msg.get("content", ""))
        kwargs = frozenset((k, str(v)) for k, v in msg.get("additional_kwargs", {}).items())
        
        if msg_type == "human":
            return HumanMessage(content=content)
        elif msg_type in ("assistant", "ai"):
            # Ensure tool_calls is immutable
            tool_calls = []  # Always start with empty tool calls
            return AIMessage(content=content, tool_calls=tool_calls)
        elif msg_type == "system":
            return SystemMessage(content=content)
        elif msg_type == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=str(msg.get("tool_call_id", "")),
                name=str(msg.get("name", "")),
                additional_kwargs=dict(kwargs) if kwargs else {}
            )
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    return msg


def convert_messages(messages):
    """Convert a list of messages to proper BaseMessage objects with immutable state."""
    if not messages:
        return []
    return tuple(convert_message(msg) for msg in messages)


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
        # Convert state to immutable form using frozenset
        messages = convert_messages(state.get("messages", []))
        
        # Convert config to immutable form
        config_dict = dict(config) if hasattr(config, "dict") else {}
        configurable = frozenset((k, str(v)) for k, v in config_dict.get("configurable", {}).items())
        
        # Get LLM with immutable config
        llm = get_llm(dict(configurable))
        tools = (search_web, get_current_weather, calendar_tool)
        llm_with_tools = llm.bind_tools(tools)
        
        # Create immutable input state
        input_messages = list(messages) + [system_prompt]
        
        # Get response
        response = llm_with_tools.invoke(input_messages)
        
        # Ensure response is properly formatted with immutable components
        if isinstance(response, dict):
            content = str(response.get("content", ""))
            response = AIMessage(
                content=content,
                tool_calls=[],
                additional_kwargs={"final_answer": True}
            )
        elif not isinstance(response, AIMessage):
            response = AIMessage(
                content=str(response),
                tool_calls=[],
                additional_kwargs={"final_answer": True}
            )
        else:
            # Ensure immutable components
            response.tool_calls = []
            response.additional_kwargs = {"final_answer": True}
        
        # Return result with immutable message
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
        # Convert state to immutable form using frozenset
        messages = convert_messages(state.get("messages", []))
        if not messages:
            return {"messages": []}
        
        # Get tool calls in immutable form
        last_message = messages[-1]
        tool_calls = []
        if isinstance(last_message, AIMessage):
            tool_calls = tuple(
                (str(tc.name), str(tc.id), frozenset(tc.args.items()))
                for tc in getattr(last_message, "tool_calls", [])
            )
        
        if not tool_calls:
            return {"messages": list(messages)}

        # Process tool calls with immutable handling
        tool_messages = []
        for tool_name, tool_id, tool_args in tool_calls:
            try:
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
                            additional_kwargs={"error": True}
                        )
                    )
                    continue

                # Execute tool with immutable args
                input_obj = input_model(**dict(tool_args))
                output = tool_func(input=input_obj)
                
                # Create immutable tool message
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tool_id,
                        name=tool_name,
                        additional_kwargs={}
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
                        additional_kwargs={"error": True}
                    )
                )

        # Convert config to immutable form
        config_dict = dict(config) if hasattr(config, "dict") else {}
        configurable = frozenset((k, str(v)) for k, v in config_dict.get("configurable", {}).items())
        
        # Get LLM with immutable config
        llm = get_llm(dict(configurable))
        
        # Create immutable input state
        input_messages = list(messages) + tool_messages
        
        # Get final response
        final_response = llm.invoke(input_messages)
        
        # Ensure response is properly formatted with immutable components
        if isinstance(final_response, dict):
            content = str(final_response.get("content", "Task completed"))
            final_response = AIMessage(
                content=content,
                tool_calls=[],
                additional_kwargs={"final_answer": True}
            )
        elif not isinstance(final_response, AIMessage):
            final_response = AIMessage(
                content=str(final_response),
                tool_calls=[],
                additional_kwargs={"final_answer": True}
            )
        else:
            final_response.tool_calls = []
            final_response.additional_kwargs = {"final_answer": True}
        
        # Return result with immutable messages
        return {"messages": list(messages) + tool_messages + [final_response]}
        
    except Exception as e:
        error_msg = f"Error processing tool results: {str(e)}"
        logger.error(error_msg)
        return {
            "messages": list(messages) + [
                SystemMessage(
                    content=error_msg,
                    additional_kwargs={"final_answer": True}
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
