"""
Enhanced News Reporter Agent for LangGraph.

This module provides a workflow for fetching and summarizing news based on user queries.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, news_tool, calc_tool
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Helper function to convert messages to BaseMessage objects
def convert_message(msg):
    if isinstance(msg, dict):
        msg_type = msg.get("type")
        if msg_type == "human":
            return HumanMessage(content=msg.get("content", ""))
        elif msg_type == "assistant":
            return AIMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls", []))
        elif msg_type == "system":
            return SystemMessage(content=msg.get("content", ""))
        elif msg_type == "tool":
            return ToolMessage(content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", ""), name=msg.get("name", ""))
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    return msg

def convert_messages(messages):
    return [convert_message(msg) for msg in messages]

def news_report(state: MessagesState, config: dict) -> dict:
    """Main node: fetch news based on user query and update the state."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state.get("messages", []))
    messages = state["messages"]
    user_query = next((msg.content for msg in messages if isinstance(msg, HumanMessage)), "")

    if not user_query:
        response = AIMessage(content="No query provided to search for news.", additional_kwargs={"final_answer": True})
        return {"messages": messages + [response]}

    llm = get_llm(config.get("configurable", {}))
    tools = [search_web, news_tool, calc_tool]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages + [
        SystemMessage(content=(
            "You are a News Reporter Agent. Use the available tools to gather and summarize news.\n"
            "Always use news_tool first, then search_web if needed for additional context."
        ))
    ])
    
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
    return {"messages": messages + [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Process tool outputs and format final response."""
    # Convert messages to BaseMessage objects
    state["messages"] = convert_messages(state["messages"])
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_messages = []
    for tc in last_message.tool_calls:
        tool = next(t for t in [search_web, news_tool, calc_tool] if t.name == tc["name"])
        try:
            output = tool.invoke(tc["args"])
            tool_messages.append(ToolMessage(
                content=str(output),
                tool_call_id=tc["id"],
                name=tc["name"]
            ))
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_messages.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tc["id"],
                name=tc["name"]
            ))

    llm = get_llm(config.get("configurable", {}))
    final_response = llm.invoke(state["messages"] + tool_messages + [
        SystemMessage(content="Create a clear and concise summary of the news articles.")
    ])
    
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
news_reporter_graph = StateGraph(MessagesState)
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.add_node("tools", tools_node)
news_reporter_graph.add_node("process_results", process_tool_results)
news_reporter_graph.set_entry_point("news_report")
news_reporter_graph.add_conditional_edges(
    "news_report",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
news_reporter_graph.add_edge("tools", "process_results")
news_reporter_graph.add_edge("process_results", "news_report")
news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]