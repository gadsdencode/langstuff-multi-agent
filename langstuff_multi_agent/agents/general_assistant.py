# langstuff_multi_agent/agents/general_assistant.py
"""
General Assistant Agent module for handling diverse queries.

This module provides a workflow for addressing general user requests
using a variety of tools. In this revised version, we mark the final assistant 
message with a 'final_answer' flag and (critically) remove any unconditional edge 
that forces the loop to continue, and we explicitly set the finish point to END.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, get_current_weather, has_tool_calls, news_tool
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langgraph.types import Command  # Use Command to control flow
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Initialize the graph with the MessagesState and configuration schema
general_assistant_graph = StateGraph(MessagesState, ConfigSchema)

# Define the available tools and create the ToolNode
tools = [search_web, get_current_weather, news_tool]
tool_node = ToolNode(tools)

def assist(state, config):
    """Provide general assistance with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        messages[-1].additional_kwargs["final_answer"] = True
        return {"messages": messages}
    
    system_message = SystemMessage(
        content=(
            "You are a General Assistant Agent. Your task is to assist with a variety of general queries and tasks.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Provide general information and answer questions.\n"
            "- get_current_weather: Retrieve current weather updates.\n"
            "- news_tool: Retrieve news headlines and articles.\n\n"
            "Instructions:\n"
            "1. Understand the user's request.\n"
            "2. Use the available tools to gather relevant information when needed.\n"
            "3. Provide clear, concise, and helpful responses."
        )
    )
    
    response = llm.invoke(messages + [system_message])
    
    # Critical fix: Always mark final answer in new responses
    if isinstance(response, AIMessage) and not response.tool_calls:
        response.additional_kwargs["final_answer"] = True
    
    return {"messages": messages + [response]}

def process_tool_results(state, config):
    """Process tool outputs and return a final assistant response."""
    messages = state.get("messages", [])
    if not messages:
        return state
        
    last_message = messages[-1]
    tool_outputs = []
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
        tool_calls = last_message.tool_calls
        if tool_calls:
            for tc in tool_calls:
                try:
                    output = f"Tool {tc.name} result: {tc.get('output', '')}"
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": f"Tool execution failed: {str(e)}"
                    })
            
            updated_messages = messages + [
                ToolMessage(
                    content=to["output"],
                    tool_call_id=to["tool_call_id"]
                ) for to in tool_outputs
            ]
            
            llm = get_llm(config.get("configurable", {}))
            final_response = llm.invoke(updated_messages)
            if isinstance(final_response, AIMessage):
                final_response.additional_kwargs["final_answer"] = True
            
            return {"messages": updated_messages + [final_response]}
    return state

def assist_edge_condition(state):
    """Enhanced edge condition handling"""
    msgs = state.get("messages", [])
    if not msgs:
        return "tools"
    
    last_msg = msgs[-1]
    
    # Improved final answer detection
    if isinstance(last_msg, AIMessage):
        if last_msg.additional_kwargs.get("final_answer", False):
            return END
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
    
    # Handle tool message transfers
    if isinstance(last_msg, ToolMessage) and hasattr(last_msg, 'goto'):
        return END
    
    # Final fallback to end conversation
    return END

# Add nodes to the graph
general_assistant_graph.add_node("assist", assist)
general_assistant_graph.add_node("tools", tool_node)
general_assistant_graph.add_node("process_results", process_tool_results)

# Set the entry point
general_assistant_graph.set_entry_point("assist")

# Add the initial edge
general_assistant_graph.add_edge(START, "assist")

# Add conditional edges from assist
general_assistant_graph.add_conditional_edges(
    "assist",
    assist_edge_condition,
    {"tools": "tools", END: END}
)

# Add edge from tools to process_results
general_assistant_graph.add_edge("tools", "process_results")

# Add edge from process_results to END
general_assistant_graph.add_edge("process_results", END)

general_assistant_graph = general_assistant_graph.compile()

__all__ = ["general_assistant_graph"]
