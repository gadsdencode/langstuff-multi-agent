# langstuff_multi_agent/agents/news_reporter.py
"""
News Reporter Agent module for gathering and summarizing news reports.

This revised module fixes issues in the original implementation:
1. It accumulates fetched articles in state["articles"] rather than relying solely on a numeric count.
2. It uses explicit, well-formed message objects for LLM invocations.
3. It robustly parses tool responses (JSON or text fallback) and generates a final summary when enough articles are gathered.
4. It introduces a helper function to safely access message attributes, fixing the AttributeError.
"""

import json
import logging
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


def final_response(state, config):
    """Return the final summary message with proper tool call IDs if present."""
    # If the final summary is already generated in state["final_summary"], use it.
    final_msg = state.get("final_summary")
    if final_msg:
        return {"messages": [final_msg]}
    # Fallback: return all messages with tool call references
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
    articles = state.get("articles", [])
    if len(articles) >= MAX_ARTICLES:
        return "final"
    # Check if last message flags completion
    last_msg = state.get("messages", [])[-1] if state.get("messages") else None
    if last_msg and getattr(last_msg, "additional_kwargs", {}).get("report_complete"):
        return "END"
    if last_msg and getattr(last_msg, "additional_kwargs", {}).get("final_answer"):
        return "END"
    return "tools"


def news_report(state, config):
    """Initial news report node: prompt the LLM to fetch news using available tools."""
    # Ensure the state has an articles accumulator list
    if "articles" not in state:
        state["articles"] = []
    # Merge configuration
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    # Bind tools with sequential (non-parallel) calls
    llm = llm.bind_tools(tools, parallel_tool_calls=False)

    # Prepare the prompt – include user query if available
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
            "2. Return a structured list of articles in JSON format with 'title' and 'source' keys.\n"
            "3. If unable to produce JSON, format each article as: 'Article Title (Source)'.\n"
            "4. Accumulate articles until you have at least three valid ones."
        )),
        HumanMessage(content=f"User query: {user_query}")
    ]

    response = llm.invoke(state.get("messages", []) + prompt)
    # Append LLM response to state
    state.setdefault("messages", []).append(response)
    return {"messages": [response]}


def process_tool_results(state, config):
    """Process tool outputs: parse fetched articles, accumulate them, and generate a final summary if ready."""
    try:
        # Attempt to extract tool response from the latest messages
        tool_msgs = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
        raw_content = None
        if tool_msgs:
            raw_content = tool_msgs[-1].content
        else:
            # Fallback: if no ToolMessage exists, use the latest AIMessage content
            raw_content = msg_get(state.get("messages", [])[-1], "content", "")

        if not isinstance(raw_content, str):
            raise ValueError("Tool response is not a string")

        clean_content = raw_content.replace('\0', '').replace('\ufeff', '').strip()
        if not clean_content:
            raise ValueError("Empty tool response after cleaning")

        # Parse articles: try JSON first, then fallback to line parsing
        if clean_content[0] in ('{', '['):
            articles = json.loads(clean_content)
            if isinstance(articles, dict):
                articles = [articles]
        else:
            articles = []
            for line in clean_content.split("\n"):
                if " (" in line and line.endswith(")"):
                    title, source = line.rsplit(" (", 1)
                    articles.append({
                        "title": title.strip(),
                        "source": source.rstrip(")").strip()
                    })
        # Validate articles using the helper below and accumulate valid ones
        valid_articles = [art for art in articles if validate_article(art)]
        if not valid_articles:
            raise ValueError("No valid articles found in the tool response")

        # Accumulate articles in state
        state.setdefault("articles", [])
        # Avoid duplicates by checking titles
        existing_titles = {art["title"] for art in state["articles"]}
        for art in valid_articles:
            if art["title"] not in existing_titles:
                state["articles"].append(art)

        # If we have enough articles, generate the final summary
        if len(state["articles"]) >= MAX_ARTICLES:
            llm = get_llm(config.get("configurable", {}))
            summary_response = llm.invoke([
                SystemMessage(content=(
                    "Create a FINAL news summary that includes:\n"
                    "1. Clear section headers\n"
                    "2. Bullet-point summaries for each article\n"
                    "3. Source attribution\n"
                    "4. Concluding remarks"
                )),
                HumanMessage(content="\n".join(
                    f"{art['title']} ({art['source']})" for art in state["articles"][:MAX_ARTICLES]
                ))
            ])
            final_msg = AIMessage(
                content=summary_response.content,
                additional_kwargs={"final_answer": True, "report_complete": True}
            )
            state["final_summary"] = final_msg
            return {"messages": [final_msg]}

        # Otherwise, report the count of accumulated articles and continue
        status_msg = AIMessage(
            content=f"Accumulated {len(state['articles'])} valid article(s).",
            additional_kwargs={"article_count": len(state["articles"])}
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
    articles = []
    for line in content.split("\n"):
        if " (" in line and line.endswith(")"):
            title, source = line.rsplit(" (", 1)
            articles.append({
                "title": title.strip(),
                "source": source.rstrip(")").strip()
            })
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
