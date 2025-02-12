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

# Create state graph for the news reporter agent
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the news reporter
tools = [search_web, news_tool, calc_tool]
tool_node = ToolNode(tools)


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
    """Process tool outputs and format the final news report with validation."""
    # Check for handoff commands first
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

    # Process only the most recent tool call results
    tool_outputs = []
    max_articles = 5  # Limit number of articles processed
    
    # Look for ToolMessages from the last tool call
    last_tool_call_id = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            last_tool_call_id = msg.tool_call_id
            break
            
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id == last_tool_call_id:
            try:
                # Safely parse tool output
                articles = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                
                # Validate article structure
                if not isinstance(articles, list):
                    articles = [articles]
                    
                for article in articles[:max_articles]:
                    if not all(key in article for key in ['title', 'source']):
                        continue
                        
                    title = article['title'].strip()[:100]  # Limit title length
                    source = article['source']['name'].strip() if isinstance(article['source'], dict) else str(article['source'])
                    tool_outputs.append(f"{title} ({source})")
                    
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                tool_outputs.append(f"⚠️ Error processing article: {str(e)}")
                continue

    # Generate summary only if valid outputs exist
    if tool_outputs:
        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Synthesize these news articles into a concise bulleted list. Include source in parentheses."),
            HumanMessage(content="\n".join(tool_outputs[:10]))  # Limit input size
        ])
        return {"messages": [summary]}
    
    # Fallback with error diagnosis
    error_info = next((msg.content for msg in state["messages"] if isinstance(msg, ToolMessage)), "No valid news data found")
    return {
        "messages": [AIMessage(
            content=f"News update failed. Last error: {error_info[:200]}"
        )]
    }


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
