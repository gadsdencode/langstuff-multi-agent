# news_reporter.py
"""
Enhanced News Reporter Agent for LangGraph.
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

MAX_ARTICLES = 3
MAX_ITERATIONS = 3


def news_report(state, config):
    """Main node: fetch news based on user query and update the state."""
    messages = state.get("messages", [])

    # Get the user's query
    user_query = next(
        (msg.content for msg in messages if isinstance(msg, HumanMessage)),
        ""
    )

    if not user_query:
        return {
            "messages": messages + [
                AIMessage(content="No query provided to search for news.")
            ]
        }

    # Directly use the news_tool to fetch articles
    try:
        news_results = news_tool.invoke(user_query)
        if not news_results:
            return {
                "messages": messages + [
                    AIMessage(content="No news articles found for your query.")
                ]
            }

        # Parse the results
        articles = []
        for line in news_results.split('\n'):
            if line.strip():
                articles.append({"content": line.strip()})

        # Generate summary using LLM
        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="You are a news reporter. Analyze the following articles and provide a comprehensive summary including key points and insights."),
            HumanMessage(content="\n".join(article["content"] for article in articles))
        ])
        final = AIMessage(
            content=summary.content,
            additional_kwargs={"final_answer": True}
        )
        return {"messages": messages + [final]}
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return {"messages": messages + [AIMessage(content=f"Error fetching news: {str(e)}")]}


# Simplified graph structure: one node that runs news_report and then ends.
news_reporter_graph = StateGraph(MessagesState, ConfigSchema)
news_reporter_graph.add_node("news_report", news_report)
news_reporter_graph.set_entry_point("news_report")
news_reporter_graph.add_edge(START, "news_report")
news_reporter_graph.add_edge("news_report", END)
news_reporter_graph = news_reporter_graph.compile()

__all__ = ["news_reporter_graph"]
