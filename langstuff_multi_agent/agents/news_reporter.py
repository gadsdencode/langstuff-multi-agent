"""
Enhanced News Reporter Agent for LangGraph.
"""

import json
import logging
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    news_tool,
    calc_tool,
    has_tool_calls
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Define tools
tools = [search_web, news_tool, calc_tool]
tool_node = ToolNode(tools)

def news_report(state, config):
    """Main node: fetch news based on user query and update the state."""
    messages = state.get("messages", [])
    
    # Get the user's query
    user_query = next(
        (msg.content for msg in messages if isinstance(msg, HumanMessage)),
        ""
    )

    if not user_query:
        return {
            "messages": messages + [
                AIMessage(content="No query provided to search for news.")
            ]
        }

    # Get LLM with config
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    
    # Generate response with tools
    response = llm.invoke(
        messages + [
            SystemMessage(content=(
                "You are a news reporter. Use the available tools to gather and summarize news. "
                "Always use news_tool first, then search_web if needed for additional context."
            ))
        ]
    )
    
    return {"messages": messages + [response]}

def process_tool_results(state, config):
    """Process tool outputs and format final response."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if not last_message or not getattr(last_message, "tool_calls", None):
        return state
        
    tool_outputs = []
    for tc in last_message.tool_calls:
        try:
            if tc["name"] == "news_tool":
                result = news_tool.invoke(tc["args"].get("query", ""))
                tool_outputs.append({
                    "tool_call_id": tc["id"],
                    "output": result
                })
            elif tc["name"] == "search_web":
                result = search_web.invoke(tc["args"].get("query", ""))
                tool_outputs.append({
                    "tool_call_id": tc["id"],
                    "output": result
                })
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_outputs.append({
                "tool_call_id": tc["id"],
                "error": f"Tool execution failed: {str(e)}"
            })

    # Add tool messages to state
    updated_messages = messages[:-1] + [
        ToolMessage(
            content=to["output"],
            tool_call_id=to["tool_call_id"]
        ) for to in tool_outputs
    ]

    # Generate final summary
    llm = get_llm(config.get("configurable", {}))
    summary = llm.invoke(
        updated_messages + [
            SystemMessage(content=(
                "Create a clear and concise summary of the news articles. "
                "Include the most important points and insights."
            ))
        ]
    )
    
    # Mark as final answer
    summary.additional_kwargs["final_answer"] = True
    
    return {
        "messages": updated_messages + [summary]
    }

# Configure graph
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Add nodes
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.add_node("tools", tool_node)
news_reporter_graph.add_node("process_results", process_tool_results)

# Set entry point
news_reporter_graph.set_entry_point("news_report")

# Add edges
news_reporter_graph.add_edge(START, "news_report")

# Add conditional edges
news_reporter_graph.add_conditional_edges(
    "news_report",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])[-1]) else "END",
    {"tools": "tools", "END": END}
)

news_reporter_graph.add_edge("tools", "process_results")
news_reporter_graph.add_edge("process_results", "news_report")

# Compile graph
news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]
