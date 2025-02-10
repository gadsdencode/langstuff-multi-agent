# agents/context_manager.py
# This file defines the Context Manager Agent workflow.
# The Context Manager Agent tracks conversation context, summarizes important details,
# and can use tools such as search_web, read_file, and write_file.
# It uses ChatAnthropic (Claude‑2) and a ToolNode to perform these tasks.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, read_file, write_file
from langchain_anthropic import ChatAnthropic

context_manager_workflow = StateGraph(MessagesState)

# Define tools for context management
tools = [search_web, read_file, write_file]
tool_node = ToolNode(tools)

# Bind the LLM with tools
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main node that manages context with a system prompt
context_manager_workflow.add_node(
    "manage_context",
    lambda state: {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Context Manager Agent. Your task is to track and manage conversation context.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search the web for general information and recent content.\n"
                            "- read_file: Read the contents of a file.\n"
                            "- write_file: Write content to a file.\n\n"
                            "Instructions:\n"
                            "1. Keep track of key information and topics discussed in the conversation.\n"
                            "2. Summarize important points and decisions made.\n"
                            "3. Use read_file and write_file to store and retrieve context information.\n"
                            "4. If necessary, use search_web to gather additional context.\n"
                            "5. Ensure that the conversation stays focused and relevant."
                        ),
                    }
                ]
            )
        ]
    },
)
context_manager_workflow.add_node("tools", tool_node)

# Define the control flow edges (mirroring the debugger workflow structure)
context_manager_workflow.add_edge(START, "manage_context")
context_manager_workflow.add_edge(
    "manage_context",
    "tools",
    condition=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
context_manager_workflow.add_edge("tools", "manage_context")
context_manager_workflow.add_edge(
    "manage_context",
    END,
    condition=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
