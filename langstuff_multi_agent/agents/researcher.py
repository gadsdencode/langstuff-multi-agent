# agents/researcher.py
"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research 
information using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, news_tool
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

researcher_workflow = StateGraph(MessagesState, ConfigSchema)

# Define research tools
tools = [search_web, news_tool]
tool_node = ToolNode(tools)

def research(state, config):
    """Conduct research with configuration support."""
    # Get LLM with configuration
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Researcher Agent. Your task is to gather and summarize news and research information.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up recent information and background data.\n"
                            "- news_tool: Retrieve the latest news headlines and articles.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's research query.\n"
                            "2. Use the available tools to gather accurate and relevant information.\n"
                            "3. Provide a clear summary of your findings."
                        ),
                    }
                ]
            )
        ]
    }

# Define the main node with configuration support
researcher_workflow.add_node("research", research)
researcher_workflow.add_node("tools", tool_node)

# Define control flow edges
researcher_workflow.add_edge(START, "research")

# Add conditional edge from research to either tools or END
researcher_workflow.add_conditional_edges(
    "research",
    lambda state: "tools" if any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]) else "END",
    {
        "tools": "tools",
        "END": END
    }
)

# Add edge from tools back to research
researcher_workflow.add_edge("tools", "research")

# Export the workflow
__all__ = ["researcher_workflow"]
