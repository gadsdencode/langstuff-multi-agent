# agent.py
"""
Main agent module that exports the graph for LangGraph Studio.
"""

from langstuff_multi_agent.agents.supervisor import supervisor_workflow

# Export the compiled graph for LangGraph Studio
graph = supervisor_workflow.compile()
