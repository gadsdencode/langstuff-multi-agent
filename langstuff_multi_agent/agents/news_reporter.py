# langstuff_multi_agent/agents/news_reporter.py
"""
News Reporter Agent module for gathering and summarizing news reports.

This revised module enhances the processing of fetched news:
1. It accumulates fetched articles in state["articles"] without reprocessing duplicate tool responses.
2. It tracks processed tool messages using state["processed_tool_msg_count"].
3. It robustly parses tool responses using both JSON and regex-based extraction.
4. It terminates the processing cycle immediately once a final summary is produced.
"""

import json
import logging
import re
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    news_tool,
    calc_tool,
    save_memory,
    search_memories
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage

# Helper function to safely get an attribute from a message object or dictionary
def msg_get(msg, key, default=None):
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)

# Create state graph for the news reporter agent
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the news reporter
tools = [search_web, news_tool, calc_tool, save_memory, search_memories]
tool_node = ToolNode(tools)

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_ARTICLES = 3
MAX_ITERATIONS = 5  # Force finalization after a set number of cycles

def final_response(state, config):
    """Return the final summary message with proper tool call IDs if present."""
    final_msg = state.get("final_summary")
    if final_msg:
        return {"messages": [final_msg]}
    responses = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                responses.append(
                    ToolMessage(
                        content=msg.content,
                        name=tool_call.get("name"),
                        tool_call_id=tool_call.get("id")
                    )
                )
    return {"messages": responses}

def news_should_continue(state):
    """Decide whether to continue gathering news or finalize the report."""
    # Immediately finalize if final summary already exists
    if state.get("final_summary"):
        logger.info("Final summary already exists; terminating cycle.")
        return "final"

    articles = state.get("articles", [])
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        logger.info(f"Max iterations reached: {state.get('iteration_count', 0)}")
        return "final"
    if len(articles) >= MAX_ARTICLES:
        logger.info(f"Enough articles gathered: {len(articles)}")
        return "final"
    last_msg = state.get("messages", [])[-1] if state.get("messages") else None
    if last_msg and getattr(last_msg, "additional_kwargs", {}).get("report_complete"):
        return "END"
    if last_msg and getattr(last_msg, "additional_kwargs", {}).get("final_answer"):
        return "END"
    return "tools"

def news_report(state, config):
    """Initial news report node: prompt the LLM to fetch news using available tools."""
    # If final summary exists, bypass further processing.
    if state.get("final_summary"):
        logger.info("Final summary exists; bypassing news_report invocation.")
        return final_response(state, config)

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    logger.info(f"Iteration count: {state['iteration_count']}")
    if "articles" not in state:
        state["articles"] = []
    # Initialize processed tool message count if not present
    if "processed_tool_msg_count" not in state:
        state["processed_tool_msg_count"] = 0
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools, parallel_tool_calls=False)

    user_query = ""
    for msg in state.get("messages", []):
        if msg_get(msg, "role") == "user":
            user_query = msg_get(msg, "content")
            break

    prompt = [
        SystemMessage(content=(
            "You are a News Reporter Agent. Your task is to gather and report "
            "the latest news, headlines, and summaries from reliable sources.\n\n"
            "Tools available:\n"
            "- search_web: To look up recent information.\n"
            "- news_tool: To retrieve the latest news articles and headlines.\n"
            "- calc_tool: For any necessary calculations.\n"
            "- save_memory: To save gathered information for future reference.\n"
            "- search_memories: To retrieve saved information.\n\n"
            "Instructions:\n"
            "1. Analyze the user's query and fetch news articles accordingly.\n"
            "2. Return a structured JSON list of articles. Each article must be an object "
            "with 'title' and 'source' keys. For example: "
            '[{"title": "Example News Title", "source": "Example Source"}].\n'
            "3. If unable to produce JSON, output each article as: 'Article Title (Source)'.\n"
            "4. Accumulate articles until you have at least three valid ones, or until "
            "the iteration limit is reached."
        )),
        HumanMessage(content=f"User query: {user_query}")
    ]

    response = llm.invoke(state.get("messages", []) + prompt)
    state.setdefault("messages", []).append(response)
    return {"messages": [response]}

def extract_articles(raw_content):
    """
    Extract valid articles from raw content.
    Attempts JSON parsing first; if that fails, falls back to line parsing with regex.
    """
    articles = []
    try:
        articles = json.loads(raw_content)
        if isinstance(articles, dict):
            articles = [articles]
    except json.JSONDecodeError:
        # Fallback: use regex to extract lines like "Some Title (Some Source)"
        pattern = re.compile(r"(.+?)\s*\((.+?)\)\s*$")
        for line in raw_content.splitlines():
            match = pattern.search(line.strip())
            if match:
                articles.append({
                    "title": match.group(1).strip(),
                    "source": match.group(2).strip()
                })
    return articles

def process_tool_results(state, config):
    """
    Process unprocessed tool outputs:
      - Iterate over new ToolMessages (tracked via state["processed_tool_msg_count"])
      - Extract and accumulate valid articles (avoiding duplicates)
      - Update processed message count to prevent reprocessing
      - If enough articles or max iterations reached, generate final summary
    """
    try:
        tool_msgs = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
        processed_count = state.get("processed_tool_msg_count", 0)
        new_tool_msgs = tool_msgs[processed_count:]
        logger.info(f"Processing {len(new_tool_msgs)} new tool message(s).")

        new_articles = []
        for msg in new_tool_msgs:
            raw_content = msg_get(msg, "content", "")
            if not isinstance(raw_content, str):
                logger.warning("Tool message content is not a string; skipping.")
                continue
            clean_content = raw_content.replace('\0', '').replace('\ufeff', '').strip()
            if not clean_content:
                logger.warning("Empty tool response after cleaning; skipping.")
                continue
            extracted = extract_articles(clean_content)
            valid_articles = [art for art in extracted if validate_article(art)]
            if valid_articles:
                new_articles.extend(valid_articles)
            else:
                logger.warning("No valid articles found in this tool message.")

        # Update processed tool message count
        state["processed_tool_msg_count"] = len(tool_msgs)

        # Accumulate new articles, avoiding duplicates
        state.setdefault("articles", [])
        existing_titles = {art["title"] for art in state["articles"]}
        for art in new_articles:
            if art["title"] not in existing_titles:
                state["articles"].append(art)
                logger.info(f"Added article: {art['title']} from {art['source']}")

        logger.info(f"Total accumulated articles: {len(state['articles'])}")

        # Finalize if enough articles or max iterations reached
        if len(state.get("articles", [])) >= MAX_ARTICLES or state.get("iteration_count", 0) >= MAX_ITERATIONS:
            llm = get_llm(config.get("configurable", {}))
            if state.get("articles"):
                content = "\n".join(
                    f"{art['title']} ({art['source']})" for art in state["articles"][:MAX_ARTICLES]
                )
            else:
                content = "No valid news articles were found for the query."
            summary_response = llm.invoke([
                SystemMessage(content=(
                    "Create a FINAL news summary that includes:\n"
                    "1. Clear section headers\n"
                    "2. Bullet-point summaries for the news (or a note that no articles were found)\n"
                    "3. Concluding remarks"
                )),
                HumanMessage(content=content)
            ])
            final_msg = AIMessage(
                content=summary_response.content,
                additional_kwargs={"final_answer": True, "report_complete": True}
            )
            state["final_summary"] = final_msg
            logger.info("Final summary generated; terminating cycle.")
            return {"messages": [final_msg]}

        status_msg = AIMessage(
            content=f"Accumulated {len(state.get('articles', []))} valid article(s).",
            additional_kwargs={"article_count": len(state.get("articles", []))}
        )
        return {"messages": [status_msg]}

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        error_msg = AIMessage(
            content=f"Error processing news: {str(e)}",
            additional_kwargs={"error": True}
        )
        return {"messages": [error_msg]}

def validate_article(article: dict) -> bool:
    """Ensure the article has a valid title and source."""
    return (
        isinstance(article, dict) and
        all(key in article and isinstance(article[key], str) for key in ['title', 'source']) and
        len(article.get('title', '')) >= 10
    )

def handle_text_fallback(content: str, config: dict) -> dict:
    """Fallback processing if tool output is plain text."""
    articles = extract_articles(content)
    if not any(validate_article(art) for art in articles):
        raise ValueError("No valid articles in text fallback")
    llm = get_llm(config.get("configurable", {}))
    summary = llm.invoke([
        SystemMessage(content="Create concise bullet points for these articles:"),
        HumanMessage(content="\n".join(f"{art['title']} ({art['source']})" for art in articles[:5]))
    ])
    return {"messages": [summary]}

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
