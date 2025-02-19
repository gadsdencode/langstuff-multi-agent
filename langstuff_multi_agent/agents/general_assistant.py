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
    
    # Ensure response is an AIMessage
    if isinstance(response, dict):
        content = response.get("content", "")
        raw_tool_calls = response.get("tool_calls", [])
        tool_calls = [
            ToolCall(id=tc.get("id", ""), name=tc.get("name", ""), args=tc.get("args", {}))
            if isinstance(tc, dict) else tc
            for tc in raw_tool_calls
        ]
        response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    # Add final_answer if no tool calls
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True

    logger.info(f"Returning response: {response}")
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs and formats final response."""
    logger.info(f"Process tool results config: type={type(config)}, value={config}")
    last_message = state["messages"][-1]
    
    tool_calls = getattr(last_message, 'tool_calls', [])
    if not tool_calls:
        return state
    
    logger.info(f"Processing tool calls: {tool_calls}")
    tool_messages = []
    for tc in tool_calls:
        tool_name = tc.name if isinstance(tc, ToolCall) else tc.get("name")
        tool_args = tc.args if isinstance(tc, ToolCall) else tc.get("args")
        tool_id = tc.id if isinstance(tc, ToolCall) else tc.get("id")
        
        tool = next(t for t in [search_web, get_current_weather, news_tool] if t.name == tool_name)
        output = tool.invoke(tool_args)
        tool_messages.append(ToolMessage(
            content=str(output),
            tool_call_id=tool_id,
            name=tool_name
        ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages)
    
    # Ensure final_response is an AIMessage
    if isinstance(final_response, dict):
        content = final_response.get("content", "Task completed")
        raw_tool_calls = final_response.get("tool_calls", [])
        tool_calls = [
            ToolCall(id=tc.get("id", ""), name=tc.get("name", ""), args=tc.get("args", {}))
            for tc in raw_tool_calls
        ]
        final_response = AIMessage(content=content, tool_calls=tool_calls)
    elif not isinstance(final_response, AIMessage):
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
