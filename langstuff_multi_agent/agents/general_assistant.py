"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests using a variety of tools.
"""

from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, get_current_weather, news_tool
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage, ToolMessage, ToolCall
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def assist(state: MessagesState, config: dict) -> dict:
    """Provide general assistance with configuration support."""
    logger.info(f"Assist config received: type={type(config)}, value={config}")
    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, get_current_weather, news_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    # Log incoming state
    logger.info(f"Assist input state: {state}")
    
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
    
    # Invoke LLM and ensure response is an AIMessage
    response = llm_with_tools.invoke([system_prompt] + messages)
    logger.info(f"LLM response: type={type(response)}, value={response}")
    
    if isinstance(response, dict):
        content = response.get("content", "I'm your General Assistant—how can I help?")
        raw_tool_calls = response.get("tool_calls", [])
        tool_calls = [ToolCall(**tc) for tc in raw_tool_calls]
        response = AIMessage(content=content, tool_calls=tool_calls)
    else:
        if not response.tool_calls:
            response = AIMessage(
                content=response.content,
                additional_kwargs={**response.additional_kwargs, "final_answer": True}
            )

    logger.info(f"Returning response: {response}")
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, get_current_weather, news_tool] if t.name == tc.name)
        output = tool.invoke(tc.args)
        tool_messages.append(ToolMessage(
            content=output,
            tool_call_id=tc.id,
            name=tc.name
        ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)

    if not isinstance(final_response, AIMessage):
        if isinstance(final_response, dict):
            content = final_response.get("content", "Task completed")
            raw_tool_calls = final_response.get("tool_calls", [])
            tool_calls = [ToolCall(**tc) for tc in raw_tool_calls]
            final_response = AIMessage(content=content, tool_calls=tool_calls)
        else:
            final_response = AIMessage(content=str(final_response))

    final_response.additional_kwargs["final_answer"] = True

    return {"messages": state["messages"] + tool_messages + [final_response]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState, config: dict) -> dict:
    return tool_node(state)

# Define and compile the graph
general_assistant_graph = StateGraph(MessagesState)
general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", lambda state, config: tools_node(state, config))
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
