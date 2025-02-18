"""
Memory management for the LangGraph multi-agent system.
Stores and retrieves conversation memories using a simple in-memory store.
"""

from typing import List, TypedDict, Dict
from datetime import datetime, timedelta
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint

class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str
    timestamp: str

class MemoryManager:
    def __init__(self, persist_path: str = "memory_store"):
        self.memories: Dict[str, List[MemoryTriple]] = {}
        self.persist_path = persist_path

    def save_memory(self, user_id: str, memories: List[Dict[str, str]]) -> None:
        """Save memories for a user with timestamps."""
        if user_id not in self.memories:
            self.memories[user_id] = []
        for memory in memories:
            memory["timestamp"] = datetime.now().isoformat()
            self.memories[user_id].append(memory)

    def search_memories(self, user_id: str, query: str, k: int = 3) -> List[MemoryTriple]:
        """Return k most recent memories for a user."""
        if user_id not in self.memories:
            return []
        # Simple timestamp-based search (enhance with semantic search later)
        return sorted(
            self.memories[user_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )[:k]

    def delete_old_memories(self, user_id: str, days: int = 30) -> None:
        """Delete memories older than specified days."""
        if user_id not in self.memories:
            return
        cutoff = datetime.now() - timedelta(days=days)
        self.memories[user_id] = [
            m for m in self.memories[user_id]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

class LangGraphMemoryCheckpointer(BaseCheckpointSaver):
    def __init__(self, memory_manager: MemoryManager, max_history: int = 5):
        self.mm = memory_manager
        self.max_history = max_history

    def get(self, config: RunnableConfig) -> Checkpoint:
        """Synchronous get method for checkpoint retrieval."""
        user_id = config["configurable"].get("user_id", "global")
        memories = self.mm.search_memories(user_id, "recent", self.max_history)
        return Checkpoint(
            v={"memory_triples": memories, "timestamp": datetime.now().isoformat()}
        )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        """Synchronous put method for checkpoint storage."""
        user_id = config["configurable"].get("user_id", "global")
        self.mm.delete_old_memories(user_id)
        memories = [
            MemoryTriple(
                subject=item["subject"],
                predicate=item["predicate"],
                object_=item["object_"],
                timestamp=item.get("timestamp", datetime.now().isoformat())
            ) for item in checkpoint["v"]["memory_triples"]
        ]
        self.mm.save_memory(user_id, memories)

# Singleton instance
memory_manager = MemoryManager()
checkpointer = LangGraphMemoryCheckpointer(memory_manager)

__all__ = ["MemoryManager", "LangGraphMemoryCheckpointer", "memory_manager", "checkpointer"]