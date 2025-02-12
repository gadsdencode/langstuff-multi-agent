# langstuff_multi_agent/agents/researcher.py
"""
Researcher Agent module for gathering and summarizing information.

This module provides a workflow for gathering and summarizing news and research 
information using various tools.
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langstuff_multi_agent.utils.tools import (
    search_web,
    news_tool,
    calc_tool,
    has_tool_calls,
    news_tool,
    save_memory,
    search_memories
)
from langstuff_multi_agent.config import ConfigSchema, get_llm
from langchain_core.messages import ToolMessage
import json
from langchain.schema import SystemMessage, HumanMessage, AIMessage

researcher_graph = StateGraph(MessagesState, ConfigSchema)

# Define research tools
tools = [search_web, news_tool, calc_tool, news_tool, save_memory, search_memories]
tool_node = ToolNode(tools)


def research(state, config):
    """Conduct research with configuration support."""
    # Get config from state and merge with passed config
    state_config = state.get("configurable", {})
    if config:
        state_config.update(config.get("configurable", {}))
    llm = get_llm(state_config)
    llm = llm.bind_tools(tools)
    return {
        "messages": [
            llm.invoke(
                state["messages"] + [
                    {
                        "role": "system",
                        "content": (
                            "You are a Researcher Agent. Your task is to gather "
                            "and summarize news and research information.\n\n"
                            "You have access to the following tools:\n"
                            "- search_web: Look up recent info and data.\n"
                            "- news_tool: Get latest news and articles.\n"
                            "- calc_tool: Perform calculations.\n"
                            "- save_memory: Save information to memory.\n"
                            "- search_memories: Search for information in memory.\n\n"
                            "Instructions:\n"
                            "1. Analyze the user's research query.\n"
                            "2. Use tools to gather accurate and relevant info.\n"
                            "3. Provide a clear summary of your findings."
                        ),
                    }
                ]
            )
        ]
    }


def process_tool_results(state, config):
    """Processes tool outputs with enhanced error handling"""
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

        # Hybrid JSON/text parsing
        if clean_content[0] in ('{', '['):
            results = json.loads(clean_content, strict=False)
        else:
            results = [{"content": line} for line in clean_content.split("\n") if line.strip()]
        
        # Validate results structure
        if not isinstance(results, list):
            results = [results]
            
        valid_results = [
            res for res in results[:5]
            if isinstance(res, dict) and res.get("content")
        ]
        
        if not valid_results:
            raise ValueError("No valid research results")

        # Generate summary
        tool_outputs = [f"{res.get('title', 'Result')}: {res['content'][:200]}" for res in valid_results]
        llm = get_llm(config.get("configurable", {}))
        summary = llm.invoke([
            SystemMessage(content="Synthesize these research findings:"),
            HumanMessage(content="\n".join(tool_outputs))
        ])
        
        # Add memory handling
        tool_calls = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        memory_operations = [tc for tc in tool_calls if tc['name'] in ('save_memory', 'search_memories')]
        if memory_operations:
            return handle_memory_operations(state, memory_operations, config)
        
        return {"messages": [summary]}

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to raw content display
        return {"messages": [AIMessage(
            content=f"Research summary:\n{clean_content[:500]}",
            additional_kwargs={"raw_data": True}
        )]}


def validate_result(result: dict) -> bool:
    """Ensure research result has minimum required fields"""
    return isinstance(result, dict) and "content" in result


# Add new memory handling function
def handle_memory_operations(state, tool_calls, config):
    outputs = []
    for tc in tool_calls:
        try:
            if tc['name'] == 'save_memory':
                result = save_memory.invoke(
                    tc['args'], 
                    {"configurable": config.get("configurable", {})}
                )
            elif tc['name'] == 'search_memories':
                result = search_memories.invoke(
                    tc['args'], 
                    {"configurable": config.get("configurable", {})}
                )
            outputs.append({
                "tool_call_id": tc["id"],
                "output": result
            })
        except Exception as e:
            outputs.append({
                "tool_call_id": tc["id"],
                "error": str(e)
            })
    
    return {"messages": [ToolMessage(
        content=json.dumps([o["output"] for o in outputs]),
        tool_call_id=[o["tool_call_id"] for o in outputs]
    )]}


researcher_graph.add_node("research", research)
researcher_graph.add_node("tools", tool_node)
researcher_graph.add_node("process_results", process_tool_results)
researcher_graph.set_entry_point("research")
researcher_graph.add_edge(START, "research")

researcher_graph.add_conditional_edges(
    "research",
    lambda state: (
        "tools" if has_tool_calls(state.get("messages", [])) else "END"
    ),
    {"tools": "tools", "END": END}
)

researcher_graph.add_edge("tools", "process_results")
researcher_graph.add_edge("process_results", "research")

researcher_graph = researcher_graph.compile()

__all__ = ["researcher_graph"]
