# langstuff_multi_agent/agents/creative_content.py
"""
Creative Content Agent module for generating creative writing, marketing copy, social media posts, or brainstorming ideas.

This module provides a workflow for generating creative content using various tools and a creative prompt.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import search_web, calc_tool, has_tool_calls
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage

# Create state graph for the Creative Content Agent
creative_content_graph = StateGraph(MessagesState, ConfigSchema)

# Define the tools available for the creative content agent.
# Here we include search_web (to gather inspiration) and calc_tool (if needed).
tools = [search_web, calc_tool]
tool_node = ToolNode(tools)


def creative_content(state, config):
    """Generate creative content based on the user's query with configuration support."""
    # Merge configuration from state and passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    # Invoke the LLM with a creative system prompt
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Creative Content Agent. Your task is to generate creative writing, marketing copy, "
                            "social media posts, or brainstorming ideas. Use vivid, imaginative, and engaging language to "
                            "craft content that inspires and captivates.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Use this tool to look up trends or inspiration from online sources.\n"
                            "- calc_tool: Use this for any quick calculations if needed (though it is secondary in this role).\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's creative query.\n"
                            "2. Draw upon your creative instincts (and any tool data if helpful) to generate an inspiring draft.\n"
                            "3. Produce a final piece of creative content that directly addresses the query."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Process tool outputs and integrate them into a final creative content draft."""
    # Check for handoff commands first (if any)
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
    # Collect outputs from ToolMessages, if any
    tool_outputs = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                output = f"Tool {msg.tool_call_id} result: {msg.content}"
                tool_outputs.append(output)
            except Exception as e:
                tool_outputs.append(f"Error processing tool result: {str(e)}")
    # If we have tool outputs, use the LLM to synthesize them into a creative draft
    if tool_outputs:
        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Synthesize the following inspirations into a creative draft:"),
            HumanMessage(content="\n".join(tool_outputs))
        ])
        return {"messages": [summary]}

    # If no tool outputs were collected, just return the current state
    return state

# Configure the state graph for the creative content agent
creative_content_graph.add_node("creative_content", creative_content)
creative_content_graph.add_node("tools", tool_node)
creative_content_graph.add_node("process_results", process_tool_results)
creative_content_graph.set_entry_point("creative_content")
creative_content_graph.add_edge(START, "creative_content")

creative_content_graph.add_conditional_edges(
    "creative_content",
    lambda state: ("tools" if has_tool_calls(state.get("messages", [])) else "END"),
    {"tools": "tools", "END": END}
)

creative_content_graph.add_edge("tools", "process_results")
creative_content_graph.add_edge("process_results", "creative_content")

creative_content_graph = creative_content_graph.compile()

__all__ = ["creative_content_graph"]
