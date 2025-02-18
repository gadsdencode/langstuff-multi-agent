"""
Memory management for the LangGraph multi-agent system.
Stores and retrieves conversation memories with persistence options.
"""

import json
import os
from typing import List, TypedDict, Dict, Optional
from datetime import datetime, timedelta
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str
    timestamp: str


class SupervisorState(TypedDict):
    messages: List[dict]
    next: str
    error_count: int
    reasoning: Optional[str]


class MemoryManager:
    def __init__(self, persist_path: str = "memory_store.json"):
        self.memories: Dict[str, List[MemoryTriple]] = {}
        self.persist_path = persist_path
        self.load_from_disk()

    def save_memory(self, user_id: str, memories: List[Dict[str, str]]) -> None:
        """Save memories for a user with timestamps."""
        if user_id not in self.memories:
            self.memories[user_id] = []
        for memory in memories:
            memory["timestamp"] = datetime.now().isoformat()
            self.memories[user_id].append(memory)
        self.save_to_disk()

    def search_memories(self, user_id: str, query: str, k: int = 3) -> List[MemoryTriple]:
        """Return k most recent memories matching the query."""
        if user_id not in self.memories:
            return []
        # Simple content filter (enhance with LLM-based semantic search later)
        relevant = [m for m in self.memories[user_id] if query.lower() in f"{m['subject']} {m['predicate']} {m['object_']}".lower()]
        return sorted(relevant, key=lambda x: x["timestamp"], reverse=True)[:k]

    def delete_old_memories(self, user_id: str, days: int = 30) -> None:
        """Delete memories older than specified days."""
        if user_id not in self.memories:
            return
        cutoff = datetime.now() - timedelta(days=days)
        self.memories[user_id] = [
            m for m in self.memories[user_id]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        self.save_to_disk()

    def save_to_disk(self) -> None:
        """Persist memories to disk."""
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f)

    def load_from_disk(self) -> None:
        """Load memories from disk if available."""
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self.memories = json.load(f)


class LangGraphMemoryCheckpointer(BaseCheckpointSaver):
    def __init__(self, memory_manager: MemoryManager, max_history: int = 5):
        self.mm = memory_manager
        self.max_history = max_history

    def get(self, config: RunnableConfig) -> Checkpoint:
        """Retrieve checkpoint for the given config."""
        user_id = config["configurable"].get("user_id", "global")
        memories = self.mm.search_memories(user_id, "recent", self.max_history)
        # For now, return memories as part of state; expand to full SupervisorState later
        return Checkpoint(v={
            "memory_triples": memories,
            "timestamp": datetime.now().isoformat()
        })

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        """Store checkpoint for the given config."""
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


memory_manager = MemoryManager()
checkpointer = LangGraphMemoryCheckpointer(memory_manager)

__all__ = ["MemoryManager", "LangGraphMemoryCheckpointer", "memory_manager", "checkpointer"]