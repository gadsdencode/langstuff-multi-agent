# langstuff_multi_agent/agents/researcher.py
"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research 
information using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, news_tool, has_tool_calls
from langchain_anthropic import ChatAnthropic
from langstuff_multi_agent.config import ConfigSchema, get_llm

researcher_graph = StateGraph(MessagesState, ConfigSchema)

# Define research tools
tools = [search_web, news_tool]
tool_node = ToolNode(tools)


def research(state, config):
    """Conduct research with configuration support."""
    llm = get_llm(config.get("configurable", {}))
    llm = llm.bind_tools(tools)
    return {
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
    }


def process_tool_results(state):
    """Processes tool outputs and formats FINAL user response"""
    last_message = state.messages[-1]
    
    if tool_calls := getattr(last_message, 'tool_calls', None):
        tool_outputs = [tc["output"] for tc in tool_calls if "output" in tc]
        
        # Generate FINAL response with tool data
        return {"messages": [
            get_llm().invoke([
                {"role": "user", "content": state.messages[0].content},
                {"role": "assistant", "content": f"Tool outputs: {tool_outputs}"},
                {"role": "system", "content": (
                    "Formulate final answer using these results. "
                    "Synthesize the research findings into a clear summary."
                )}
            ])
        ]}
    return state


researcher_graph.add_node("research", research)
researcher_graph.add_node("tools", tool_node)
researcher_graph.add_node("process_results", process_tool_results)
researcher_graph.set_entry_point("research")
researcher_graph.add_edge(START, "research")

researcher_graph.add_conditional_edges(
    "research",
    lambda state: "tools" if has_tool_calls(state.get("messages", [])) else "END",
    {"tools": "tools", "END": END}
)

researcher_graph.add_edge("tools", "process_results")
researcher_graph.add_edge("process_results", END)

researcher_graph = researcher_graph.compile()

__all__ = ["researcher_graph"]
