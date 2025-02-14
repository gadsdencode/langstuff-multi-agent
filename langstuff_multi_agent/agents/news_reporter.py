# news_reporter.py
"""
Enhanced News Reporter Agent for LangGraph.

This version fixes:
- Infinite loop by breaking the cycle when a final summary exists.
- Ensures the agent terminates by routing to the final node instead of looping back.
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

logger = logging.getLogger(__name__)

# Constants
MAX_ARTICLES = 3
MAX_ITERATIONS = 5  # Prevents infinite loops

# Helper function to safely extract message attributes.
def msg_get(msg, key, default=None):
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)

def final_response(state, config):
    """Ensures final response is correctly sent to the user."""
    final_msg = state.get("final_summary")
    if final_msg:
        logger.info("Returning final summary to user.")
        return {"messages": [final_msg]}
    # Fallback: if no final summary exists, create a default final message.
    logger.info("No final summary found; creating default final response to break loop.")
    default_final = AIMessage(
        content="No sufficient news articles could be summarized.",
        additional_kwargs={"final_answer": True}
    )
    state["final_summary"] = default_final
    return {"messages": [default_final]}

def news_should_continue(state):
    """Determines whether to continue or finalize the agent's workflow."""
    if state.get("final_summary"):
        logger.info("Final summary exists; terminating.")
        return "final"
    last_msg = state.get("messages", [])[-1] if state.get("messages") else None
    if isinstance(last_msg, ToolMessage) and last_msg.content.strip():
        logger.info("ToolMessage received; ending process.")
        return "final"
    if len(state.get("articles", [])) >= MAX_ARTICLES or state.get("iteration_count", 0) >= MAX_ITERATIONS:
        logger.info("Sufficient articles gathered or max iterations reached; finalizing.")
        return "final"
    return "tools"

def news_report(state, config):
    """Fetches and processes news, ensuring termination when enough data is collected."""
    if state.get("final_summary"):
        logger.info("Final summary exists; returning it instead of running news_report.")
        return final_response(state, config)

    # Increment iteration count and update state persistently.
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    logger.info(f"Iteration count: {state['iteration_count']}")

    state.setdefault("articles", [])
    state.setdefault("processed_tool_msg_count", 0)
    state.setdefault("messages", [])

    llm = get_llm(state.get("configurable", {}))
    llm = llm.bind_tools([search_web, news_tool, calc_tool, save_memory, search_memories])

    # Preserve the original user query for consistency.
    if "user_query" not in state:
        state["user_query"] = next(
            (msg_get(msg, "content") for msg in state.get("messages", []) if msg_get(msg, "role") == "user"),
            ""
        )

    prompt = [
        SystemMessage(content="You are a News Reporter Agent. Fetch and summarize news articles."),
        HumanMessage(content=f"User query: {state['user_query']}")
    ]

    response = llm.invoke(state.get("messages", []) + prompt)
    state["messages"].append(response)
    return state

def process_tool_results(state, config):
    """Processes tool outputs, accumulates news, and finalizes when aggregation is complete."""
    try:
        tool_msgs = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
        processed_count = state.get("processed_tool_msg_count", 0)
        new_tool_msgs = tool_msgs[processed_count:]
        logger.info(f"Processing {len(new_tool_msgs)} new tool message(s).")

        new_articles = []
        for msg in new_tool_msgs:
            raw_content = msg_get(msg, "content", "").strip()
            if not raw_content:
                continue
            extracted = extract_articles(raw_content)
            valid_articles = [art for art in extracted if validate_article(art)]
            if valid_articles:
                new_articles.extend(valid_articles)

        state["processed_tool_msg_count"] = len(tool_msgs)
        state.setdefault("articles", [])
        existing_titles = {art["title"] for art in state["articles"]}
        for art in new_articles:
            if art["title"] not in existing_titles:
                state["articles"].append(art)

        if len(state["articles"]) >= MAX_ARTICLES:
            llm = get_llm(config.get("configurable", {}))
            summary_response = llm.invoke([
                SystemMessage(content="Create a final news summary."),
                HumanMessage(content="\n".join(f"{art['title']} ({art['source']})" for art in state["articles"][:MAX_ARTICLES]))
            ])
            state["final_summary"] = AIMessage(
                content=summary_response.content,
                additional_kwargs={"final_answer": True}
            )
            state["messages"].append(state["final_summary"])
            return state

        state["messages"].append(AIMessage(content=f"Accumulated {len(state['articles'])} articles."))
        return state

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error processing news: {str(e)}", additional_kwargs={"error": True}))
        return state

def extract_articles(raw_content):
    """Extracts articles from tool responses."""
    articles = []
    try:
        articles = json.loads(raw_content)
        if isinstance(articles, dict):
            articles = [articles]
    except json.JSONDecodeError:
        pattern = re.compile(r"(.+?)\s*\((.+?)\)\s*$")
        for line in raw_content.splitlines():
            match = pattern.search(line.strip())
            if match:
                articles.append({"title": match.group(1).strip(), "source": match.group(2).strip()})
    return articles

def validate_article(article):
    """Ensures an article has both a title and a source, and that the title is sufficiently descriptive."""
    return isinstance(article, dict) and "title" in article and "source" in article and len(article["title"]) > 10

# LangGraph setup with corrected edge to break the infinite loop.
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.add_node("tools", ToolNode([search_web, news_tool, calc_tool, save_memory, search_memories]))
news_reporter_graph.add_node("process_results", process_tool_results)
news_reporter_graph.add_node("final", final_response)

news_reporter_graph.set_entry_point("news_report")
news_reporter_graph.add_edge(START, "news_report")
news_reporter_graph.add_conditional_edges("news_report", news_should_continue, {"tools": "tools", "final": "final", "END": END})
# --- CORRECTED EDGE: Instead of an unconditional loop from process_results back to news_report,
# we now conditionally route: if a final summary exists, go to "final"; otherwise, loop back to "news_report".
news_reporter_graph.add_conditional_edges(
    "process_results",
    lambda state: "final" if state.get("final_summary") else "news_report",
    {"final": "final", "news_report": "news_report"}
)
news_reporter_graph.add_edge("tools", "process_results")
news_reporter_graph.add_edge("final", END)

news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]
