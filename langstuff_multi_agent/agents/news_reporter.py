# langstuff_multi_agent/agents/news_reporter.py
"""
News Reporter Agent module for gathering and summarizing news reports.

This module provides a workflow for gathering and reporting the latest news using various tools.
"""

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
import json
import logging

# Create state graph for the news reporter agent
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the news reporter
tools = [search_web, news_tool, calc_tool]
tool_node = ToolNode(tools)

# Configure logger
logger = logging.getLogger(__name__)

def final_response(state, config):
    """Directly return last ToolMessage for immediate responses"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            return {"messages": [msg]}
    return {"messages": state["messages"]}

def news_should_continue(state):
    """Enhanced conditional routing with direct return check"""
    messages = state.get("messages", [])
    if not messages:
        return "END"
        
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", []):
        return "END"

    # Check first tool call for return_direct flag
    args = last_message.tool_calls[0].get("args", {})
    return "final" if args.get("return_direct", False) else "tools"

def news_report(state, config):
    """Conduct news reporting with configuration support."""
    # Merge the configuration from the state and the passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a system prompt tailored for a news reporter agent
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a News Reporter Agent. Your task is to gather and report "
                            "the latest news, headlines, and summaries from reliable sources.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up recent info and data.\n"
                            "- news_tool: Retrieve the latest news articles and headlines.\n"
                            "- calc_tool: Perform calculations if necessary.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's news query.\n"
                            "2. Use the available tools to gather accurate and up-to-date news.\n"
                            "3. Provide a clear and concise summary of your findings."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Process tool outputs with enhanced error handling"""
    # NEW: Prune previous error messages first
    state["messages"] = [msg for msg in state["messages"] 
                        if not (isinstance(msg, ToolMessage) and "⚠️ Error" in msg.content)]
    
    # Existing handoff check remains
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['name'].startswith('transfer_to_'):
                    return {
                        "messages": [ToolMessage(
                            goto=tc['name'].replace('transfer_to_', ''),
                            graph=ToolMessage.PARENT
                        )]
                    }

    # NEW: Add recursion counter
    state.setdefault("recursion_count", 0)
    if state["recursion_count"] > 3:
        return {"messages": [AIMessage(content="Maximum processing attempts reached")]}
    state["recursion_count"] += 1

    # Modified processing with strict validation
    try:
        last_tool_msg = next(msg for msg in reversed(state["messages"]) 
                            if isinstance(msg, ToolMessage))
        
        # NEW: Validate content before parsing
        raw_content = last_tool_msg.content
        if not raw_content or not isinstance(raw_content, str):
            raise ValueError("Empty or non-string tool response")
            
        # Clean content: remove null bytes and whitespace
        clean_content = raw_content.replace('\0', '').strip()
        if not clean_content:
            raise ValueError("Empty content after cleaning")

        articles = json.loads(clean_content)
        
        if not isinstance(articles, list):
            articles = [articles]

        valid_articles = [
            art for art in articles[:5]  # Hard limit
            if validate_article(art)  # NEW validation function
        ]
        
        if not valid_articles:
            raise ValueError("No valid articles after filtering")

        # Rest of processing remains...
        # ... existing summary generation code ...

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed. Raw content: {raw_content[:200]}")
        return {"messages": [AIMessage(
            content=f"News format error: {str(e)}",
            additional_kwargs={"error": True}
        )]}
    except ValueError as e:
        logger.error(f"Content validation failed: {str(e)}")
        return {"messages": [AIMessage(
            content=f"Invalid news data: {str(e)}",
            additional_kwargs={"error": True}
        )]}

def validate_article(article: dict) -> bool:
    """Strict validation for news article structure"""
    return all(
        key in article and isinstance(article[key], str)
        for key in ['title', 'source']
    ) and len(article.get('title', '')) >= 10


# Configure the state graph for the news reporter agent
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.add_node("tools", tool_node)
news_reporter_graph.add_node("process_results", process_tool_results)
news_reporter_graph.add_node("final", final_response)
news_reporter_graph.set_entry_point("news_report")
news_reporter_graph.add_edge(START, "news_report")

news_reporter_graph.add_conditional_edges(
    "news_report",
    news_should_continue,
    {"tools": "tools", "final": "final", "END": END}
)

news_reporter_graph.add_edge("final", END)
news_reporter_graph.add_edge("tools", "process_results")
news_reporter_graph.add_edge("process_results", "news_report")

news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]
