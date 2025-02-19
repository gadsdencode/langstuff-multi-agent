"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research 
information using various tools.
"""

import logging
from langgraph.graph import StateGraph, MessagesState, END
from langstuff_multi_agent.utils.tools import tool_node, has_tool_calls, search_web, news_tool, calc_tool, save_memory, search_memories
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def research(state: MessagesState, config: dict) -> dict:
    """Conduct research with configuration support."""
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    tools = [search_web, news_tool, calc_tool, save_memory, search_memories]
    llm = llm.bind_tools(tools)
    response = llm.invoke(state["messages"] + [
        SystemMessage(content=(
            "You are a Researcher Agent. Your task is to gather and summarize news and research information.\n\n"
            "You have access to the following tools:\n"
            "- search_web: Look up recent info and data.\n"
            "- news_tool: Get latest news and articles.\n"
            "- calc_tool: Perform calculations.\n"
            "- save_memory: Save information to memory.\n"
            "- search_memories: Search for information in memory.\n\n"
            "Instructions:\n"
            "1. Analyze the user's research query.\n"
            "2. Use tools to gather accurate and relevant info.\n"
            "3. Provide a clear summary of your findings."
        ))
    ])
    if not response.tool_calls:
        response.additional_kwargs["final_answer"] = True  # Signal completion
    return {"messages": [response]}

def process_tool_results(state: MessagesState, config: dict) -> dict:
    """Processes tool outputs with enhanced error handling."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    try:
        tool_outputs = []
        for tc in last_message.tool_calls:
            tool = next(t for t in [search_web, news_tool, calc_tool, save_memory, search_memories] if t.name == tc["name"])
            result = tool.invoke(tc["args"], config=config if tc["name"] in ["save_memory", "search_memories"] else None)
            tool_outputs.append({"tool_call_id": tc["id"], "output": result})

        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Synthesize these research findings:"),
            HumanMessage(content="\n".join([to["output"] if isinstance(to["output"], str) else json.dumps(to["output"]) for to in tool_outputs]))
        ])
        summary.additional_kwargs["final_answer"] = True
        return {"messages": state["messages"] + [ToolMessage(
            content=summary.content,
            tool_call_id=last_message.tool_calls[0]["id"]  # Use first tool call ID for simplicity
        )]}

    except Exception as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error processing research: {str(e)}", additional_kwargs={"final_answer": True})]}

# Define a wrapper for the tools node to avoid passing config
def tools_node(state: MessagesState) -> dict:
    return tool_node(state)

# Define and compile the graph
researcher_graph = StateGraph(MessagesState)
researcher_graph.add_node("research", research)
researcher_graph.add_node("tools", tools_node)  # Use wrapped tools_node
researcher_graph.add_node("process_results", process_tool_results)
researcher_graph.set_entry_point("research")
researcher_graph.add_conditional_edges(
    "research",
    lambda state: "tools" if has_tool_calls(state["messages"]) else END,
    {"tools": "tools", END: END}
)
researcher_graph.add_edge("tools", "process_results")
researcher_graph.add_edge("process_results", "research")
researcher_graph = researcher_graph.compile()

__all__ = ["researcher_graph"]