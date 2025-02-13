# langstuff_multi_agent/agents/analyst.py
"""
Analyst Agent module for data analysis and interpretation.

This module provides a workflow for analyzing data and performing
calculations using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    python_repl,
    calc_tool,
    has_tool_calls,
    news_tool
)
from langstuff_multi_agent.config import get_llm
from langchain_core.messages import ToolMessage
import json
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging

analyst_graph = StateGraph(MessagesState)

# Define tools for analysis tasks
tools = [search_web, python_repl, calc_tool, news_tool]
tool_node = ToolNode(tools)

logger = logging.getLogger(__name__)


def analyze_data(state):
    """Analyze data and perform calculations."""
    messages = state.get("messages", [])
    config = state.get("config", {})

    llm = get_llm(config.get("configurable", {}))
    response = llm.invoke(messages)

    return {"messages": messages + [response]}


def process_tool_results(state, config):
    """Processes tool outputs with robust data validation"""
    # Clean previous error messages
    state["messages"] = [msg for msg in state["messages"]
                        if not (isinstance(msg, ToolMessage) and "⚠️" in msg.content)]

    try:
        # Get last tool message with content validation
        last_tool_msg = next(msg for msg in reversed(state["messages"])
                            if isinstance(msg, ToolMessage))

        # Clean and validate raw content
        raw_content = last_tool_msg.content
        clean_content = raw_content.replace('\0', '').replace('\ufeff', '').strip()
        if not clean_content:
            raise ValueError("Empty tool response after cleaning")

        # Hybrid data parsing
        if clean_content[0] in ('{', '['):
            results = json.loads(clean_content, strict=False)
        else:
            results = [{"content": line} for line in clean_content.split("\n") if line.strip()]

        # Validate and process results
        if not isinstance(results, list):
            results = [results]

        valid_results = [
            res for res in results[:5]
            if validate_analysis_result(res)
        ]

        if not valid_results:
            raise ValueError("No valid analysis results")

        # Generate analytical summary
        tool_outputs = []
        for res in valid_results:
            output = f"{res.get('metric', 'Result')}: {res['value']}" if 'value' in res else res['content']
            tool_outputs.append(output[:200])

        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Analyze and interpret these results:"),
            HumanMessage(content="\n".join(tool_outputs))
        ])

        return {"messages": [summary]}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Analysis Error: {str(e)}")
        return {"messages": [AIMessage(
            content=f"Analysis summary:\n{clean_content[:500]}",
            additional_kwargs={"raw_data": True}
        )]}


def validate_analysis_result(result: dict) -> bool:
    """Validate analysis result structure"""
    return isinstance(result, dict) and any(key in result for key in ['content', 'value'])


# Initialize and configure the analyst graph
analyst_graph.add_node("analyze_data", analyze_data)
analyst_graph.add_node("tools", tool_node)
analyst_graph.add_node("process_results", process_tool_results)
analyst_graph.set_entry_point("analyze_data")
analyst_graph.add_edge(START, "analyze_data")

analyst_graph.add_conditional_edges(
    "analyze_data",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

analyst_graph.add_edge("tools", "process_results")
analyst_graph.add_edge("process_results", "analyze_data")

analyst_graph = analyst_graph.compile()

__all__ = ["analyst_graph"]
