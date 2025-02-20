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
        msg_type = msg.get("type", msg.get("role"))
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant" or msg_type == "ai":
            # Force tool_calls to empty list to avoid dictionaries
            return AIMessage(content=msg.get("content", ""), tool_calls=[])
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
    """Process user input and provide assistance."""
    # Convert messages to proper format and freeze
    messages = tuple(convert_messages(state.get("messages", [])))
    logger.info(f"Assist input state: {state}")

    try:
        llm = get_llm(config.get("configurable", {}))
        tools = tuple([search_web, get_current_weather, calendar_tool])
        llm_with_tools = llm.bind_tools(tools)

        response = llm_with_tools.invoke(list(messages) + [system_prompt])
        
        # Ensure response is properly formatted
        if isinstance(response, dict):
            content = response.get("content", "")
            response = AIMessage(content=content, tool_calls=[], additional_kwargs={})
        elif not isinstance(response, AIMessage):
            response = AIMessage(content=str(response), tool_calls=[], additional_kwargs={})
        
        # Ensure tool_calls is always a list and additional_kwargs is a new dict
        response.tool_calls = list(response.tool_calls or [])
        response.additional_kwargs = dict(response.additional_kwargs or {})
        
        # Mark as final answer if no tool calls
        if not response.tool_calls:
            response.additional_kwargs["final_answer"] = True
        
        logger.info(f"Returning response: {response}")
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
    # Convert messages to proper format and freeze
    messages = tuple(convert_messages(state.get("messages", [])))
    if not messages:
        return {"messages": []}
    
    last_message = messages[-1]
    tool_calls = tuple(last_message.tool_calls) if isinstance(last_message, AIMessage) else ()
    
    if not tool_calls:
        return {"messages": list(messages)}

    tool_messages = []
    for tc in tool_calls:
        try:
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
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tc.id,
                        name=tool_name,
                        additional_kwargs={}
                    )
                )
                continue

            # Convert args to immutable form
            args = dict(tc.args)
            input_obj = input_model(**args)
            output = tool_func(input=input_obj)
            tool_messages.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=tc.id,
                    name=tool_name,
                    additional_kwargs={}
                )
            )
        except Exception as e:
            error_msg = f"Error with tool {tc.name}: {str(e)}"
            logger.error(error_msg)
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tc.id,
                    name=tc.name,
                    additional_kwargs={}
                )
            )

    try:
        llm = get_llm(config.get("configurable", {}))
        final_response = llm.invoke(list(messages) + tool_messages)
        
        # Ensure response is properly formatted with immutable components
        if isinstance(final_response, dict):
            content = final_response.get("content", "Task completed")
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
        
        return {"messages": list(messages) + tool_messages + [final_response]}
        
    except Exception as e:
        error_msg = f"Error processing tool results: {str(e)}"
        logger.error(error_msg)
        return {
            "messages": list(messages) + tool_messages + [
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
