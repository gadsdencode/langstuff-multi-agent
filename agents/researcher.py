# agents/researcher.py
# This file defines the Researcher Agent workflow.
# The Researcher Agent gathers and summarizes news and research information.
# It uses tools such as search_web and news_tool.
# The agent is powered by ChatAnthropic (Claude‑2) and uses a ToolNode for research tasks.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from my_agent.utils.tools import search_web, news_tool
from langchain_anthropic import ChatAnthropic

researcher_workflow = StateGraph(MessagesState)

# Define research tools
tools = [search_web, news_tool]
tool_node = ToolNode(tools)

# Bind the LLM with research tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node with instructions for conducting research
researcher_workflow.add_node(
    "research",
    lambda state: {
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
    },
)
researcher_workflow.add_node("tools", tool_node)

# Define control flow edges
researcher_workflow.add_edge(START, "research")
researcher_workflow.add_edge(
    "research",
    "tools",
    condition=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
researcher_workflow.add_edge("tools", "research")
researcher_workflow.add_edge(
    "research",
    END,
    condition=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
