# debugger.py
# This file defines the Debugger Agent workflow for LangGraph.
# The Debugger Agent is responsible for analyzing code, identifying errors,
# and optionally using tools such as search_web, python_repl, read_file, and write_file.
# It uses ChatAnthropic (Claude‑2) as the underlying LLM and leverages a ToolNode
# to invoke tools when necessary.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, python_repl, read_file, write_file
from langchain_anthropic import ChatAnthropic

debugger_workflow = StateGraph(MessagesState)

# Define the tools available to the Debugger Agent
tools = [search_web, python_repl, read_file, write_file]
tool_node = ToolNode(tools)

# Define the LLM and bind the tools to it
llm = ChatAnthropic(model="claude-2", temperature=0).bind_tools(tools)

# Define the main agent node with a system prompt detailing the agent's role and instructions
debugger_workflow.add_node(
    "analyze_code",
    lambda state: {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Debugger Agent. Your task is to identify and analyze code errors.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Search the web for general information and recent content.\n"
                            "- python_repl: Execute Python code.\n"
                            "- read_file: Read the contents of a file.\n"
                            "- write_file: Write content to a file.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's code and identify potential errors.\n"
                            "2. Use the search_web tool to find relevant information about the error or related debugging techniques.\n"
                            "3. Use the python_repl tool to execute code snippets and test potential fixes.\n"
                            "4. If necessary, use read_file and write_file to modify the code.\n"
                            "5. Provide clear and concise explanations of the error and the debugging process."
                        ),
                    }
                ]
            )
        ]
    },
)
debugger_workflow.add_node("tools", tool_node)

# Define the control flow edges:
# 1. Start at 'analyze_code'.
# 2. If any message contains tool_calls, transition from 'analyze_code' to 'tools'.
# 3. After running tools, loop back to 'analyze_code'.
# 4. If no tool_calls remain, finish.
debugger_workflow.add_edge(START, "analyze_code")
debugger_workflow.add_edge(
    "analyze_code",
    "tools",
    condition=lambda state: any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)
debugger_workflow.add_edge("tools", "analyze_code")
debugger_workflow.add_edge(
    "analyze_code",
    END,
    condition=lambda state: not any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in state["messages"]),
)

# The debugger_workflow is now complete.
# :contentReference[oaicite:0]{index=0}
