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
    has_tool_calls,
    save_memory,
    search_memories
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage
import json
import logging

# Create state graph for the news reporter agent
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the news reporter
tools = [search_web, news_tool, calc_tool, save_memory, search_memories]
tool_node = ToolNode(tools)

# Configure logger
logger = logging.getLogger(__name__)


def final_response(state, config):
    """Add completion marker to final response"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            return {
                "messages": [
                    AIMessage(
                        content=msg.content,
                        additional_kwargs={"final_answer": True}  # Completion marker
                    )
                ]
            }
    return {"messages": state["messages"]}


def news_should_continue(state):
    """Fixed termination condition with explicit completion check"""
    messages = state.get("messages", [])
    if not messages:
        return "END"

    last_message = messages[-1]

    # Explicit completion marker check
    if isinstance(last_message, AIMessage) and "final_answer" in last_message.additional_kwargs:
        return "END"

    # Existing tool call check
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
                            "- calc_tool: Perform calculations if necessary.\n"
                            "- save_memory: Save information for future reference.\n"
                            "- search_memories: Retrieve saved information.\n\n"
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
    """Process tool outputs with hybrid JSON/text parsing"""
    # Clean previous error messages
    state["messages"] = [msg for msg in state["messages"]
                        if not (isinstance(msg, ToolMessage) and "⚠️" in msg.content)]

    try:
        # Get last tool message with content validation
        last_tool_msg = next(msg for msg in reversed(state["messages"])
                            if isinstance(msg, ToolMessage))

        # Null byte removal and encoding cleanup
        raw_content = last_tool_msg.content
        if not isinstance(raw_content, str):
            raise ValueError("Non-string tool response")

        clean_content = raw_content.replace('\0', '').replace('\ufeff', '').strip()
        if not clean_content:
            raise ValueError("Empty content after cleaning")

        # Attempt JSON parsing first
        if clean_content[0] in ('{', '['):
            articles = json.loads(clean_content, strict=False)
        else:
            # NEW: Handle non-JSON responses using text parsing
            articles = [
                {"title": line.split(" (")[0], "source": line.split("(")[-1].rstrip(")")}
                for line in clean_content.split("\n") if " (" in line and ")" in line
            ]
            logger.info(f"Converted {len(articles)} text entries to structured format")

        # Convert single article to list
        if not isinstance(articles, list):
            articles = [articles]

        # Process articles with validation
        valid_articles = [
            art for art in articles[:5]
            if validate_article(art)
        ]

        if not valid_articles:
            raise ValueError("No valid articles after filtering")

        # Add memory context to articles
        if 'user_id' in config.get("configurable", {}):
            memories = search_memories.invoke(
                "news preferences",
                {"configurable": config["configurable"]}
            )
            if memories:
                state["messages"].append(AIMessage(
                    content=f"User preferences context: {memories}"
                ))

        # Generate summary
        tool_outputs = []
        for art in valid_articles:
            title = art.get('title', 'Untitled')[:100]
            # Handle both string and dict source formats
            source = (
                art['source'].get('name', 'Unknown')
                if isinstance(art.get('source'), dict)
                else str(art.get('source', 'Unknown'))
            )[:50]
            tool_outputs.append(f"{title} ({source})")

        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Create concise bullet points from these articles:"),
            HumanMessage(content="\n".join(tool_outputs))
        ])

        return {"messages": [summary]}

    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}\nFirst 200 chars: {clean_content[:200]}")
        # NEW: Attempt text fallback
        if "\n" in clean_content:
            return handle_text_fallback(clean_content, config)
        return {"messages": [AIMessage(
            content=f"⚠️ News format error: {str(e)[:100]}",
            additional_kwargs={"error": True, "raw_content": clean_content[:200]}
        )]}

    except ValueError as e:
        logger.error(f"Validation Error: {str(e)}")
        return {"messages": [AIMessage(
            content=f"⚠️ Invalid news data: {str(e)[:100]}",
            additional_kwargs={"error": True}
        )]}


def handle_text_fallback(content: str, config: dict) -> dict:
    """Process text-based news format with source validation"""
    articles = []
    for line in content.split("\n"):
        if " (" in line and line.endswith(")"):
            title, source = line.rsplit(" (", 1)
            articles.append({
                "title": title.strip(),
                "source": source.rstrip(")").strip()  # Store source as string
            })

    # Validate at least 1 article has both fields
    if not any(validate_article(art) for art in articles):
        raise ValueError("No valid articles in text fallback")

    # Generate summary from parsed text
    tool_outputs = [f"{art['title']} ({art['source']})" for art in articles[:5]]
    llm = get_llm(config.get("configurable", {}))
    summary = llm.invoke([
        SystemMessage(content="Create concise bullet points from these articles:"),
        HumanMessage(content="\n".join(tool_outputs))
    ])
    return {"messages": [summary]}


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
