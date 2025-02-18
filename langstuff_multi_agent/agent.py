"""
Entry point for the LangGraph multi-agent system.

This module initializes the supervisor graph, importing the supervisor and its member graphs,
and configures the system with a persistent checkpointer for state management.
"""

import logging
from langstuff_multi_agent.agents.supervisor import create_supervisor, member_graphs
from langstuff_multi_agent.config import Config, get_llm
from langstuff_multi_agent.utils.memory import SupervisorState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing primary supervisor workflow...")

# Create and compile the supervisor graph with MemorySaver
try:
    supervisor_graph = create_supervisor(
        llm=get_llm(),
        members=list(member_graphs.keys()),
        member_graphs=member_graphs,
        state_type=SupervisorState
    ).compile(checkpointer=MemorySaver())
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Graph compilation failed: {str(e)}", exc_info=True)
    raise

# Alias for entry point (Studio uses 'graph')
graph = supervisor_graph

__all__ = ["graph", "supervisor_graph"] + list(member_graphs.keys())

logger.info("Primary supervisor workflow successfully initialized.")

# Optional: Test locally
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    test_input = {"messages": [HumanMessage(content="Test query")]}
    result = graph.invoke(test_input)
    print(result)