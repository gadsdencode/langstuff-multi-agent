"""
Entry point for the LangGraph multi-agent system.
"""

import logging
from langstuff_multi_agent.agents.supervisor import create_supervisor
from langstuff_multi_agent.config import get_llm
from langstuff_multi_agent.utils.memory import SupervisorState
from langgraph.checkpoint.memory import MemorySaver

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing minimal supervisor workflow...")

# Create and compile aminimal supervisor graph
try:
    supervisor_graph = create_supervisor(llm=get_llm()).compile(checkpointer=MemorySaver())
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Graph compilation failed: {str(e)}", exc_info=True)
    raise

# Alias for entry point
graph = supervisor_graph

__all__ = ["graph", "supervisor_graph"]

logger.info("Minimal supervisor workflow initialized.")

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    test_input = {"messages": [HumanMessage(content="Test query")]}
    result = graph.invoke(test_input)
    print(result)