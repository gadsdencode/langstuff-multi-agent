"""
Entry point for the LangGraph multi-agent system.
"""

import logging
from langstuff_multi_agent.agents.supervisor import create_supervisor
from langstuff_multi_agent.config import get_llm
from langstuff_multi_agent.utils.memory import LangGraphMemoryCheckpointer, memory_manager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing supervisor workflow with preprocessing...")

# Create and compile the supervisor graph
try:
    supervisor_graph = create_supervisor(llm=get_llm()).compile(checkpointer=LangGraphMemoryCheckpointer(memory_manager))
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Graph compilation failed: {str(e)}", exc_info=True)
    raise

# Alias for entry point
graph = supervisor_graph

__all__ = ["graph", "supervisor_graph"]

logger.info("Supervisor workflow initialized.")

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    test_input = {"messages": [HumanMessage(content="Test query")]}
    result = graph.invoke(test_input)
    print(result)
