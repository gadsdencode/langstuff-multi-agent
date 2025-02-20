"""
Memory management for the LangGraph multi-agent system.
Stores and retrieves conversation memories and full SupervisorState with persistence.
"""

import json
import os
from typing import List, TypedDict, Optional, Annotated, Dict
from datetime import datetime, timedelta
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
import operator


class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str
    timestamp: str


class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add, add_messages]
    next: str
    error_count: Annotated[int, operator.add]  # Updated to match supervisor.py
    reasoning: Optional[str]
    memory_triples: List[MemoryTriple]


class MemoryManager:
    def __init__(self, memory_path: str = "memory_store.json", state_dir: str = "state_store"):
        self.memories: Dict[str, List[MemoryTriple]] = {}
        self.memory_path = memory_path
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.load_memories_from_disk()

    def save_memory(self, user_id: str, memories: List[Dict[str, str]]) -> None:
        """Save memories for a user with timestamps."""
        if user_id not in self.memories:
            self.memories[user_id] = []
        for memory in memories:
            memory["timestamp"] = datetime.now().isoformat()
            self.memories[user_id].append(memory)
        self.save_memories_to_disk()

    def search_memories(self, user_id: str, query: str, k: int = 3) -> List[MemoryTriple]:
        """Return k most recent memories matching the query."""
        if user_id not in self.memories:
            return []
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
        self.save_memories_to_disk()

    def save_state(self, user_id: str, state: SupervisorState) -> None:
        """Save full SupervisorState to disk."""
        state_path = os.path.join(self.state_dir, f"{user_id}.json")
        serializable_state = {
            "messages": [msg.to_json() for msg in state["messages"]],
            "next": state["next"],
            "error_count": state["error_count"],
            "reasoning": state["reasoning"],
            "memory_triples": state["memory_triples"]
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f)

    def load_state(self, user_id: str) -> Optional[SupervisorState]:
        """Load full SupervisorState from disk."""
        state_path = os.path.join(self.state_dir, f"{user_id}.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                serializable_state = json.load(f)
                messages = []
                for msg_dict in serializable_state["messages"]:
                    role = msg_dict.get("kwargs", {}).get("role") or msg_dict.get("type")
                    content = msg_dict.get("kwargs", {}).get("content", "")
                    if role == "human":
                        messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "tool":
                        messages.append(ToolMessage(
                            content=content,
                            tool_call_id=msg_dict.get("kwargs", {}).get("tool_call_id", ""),
                            name=msg_dict.get("kwargs", {}).get("name", "")
                        ))
                    else:
                        messages.append(BaseMessage(type=role, content=content))
                state = {
                    "messages": messages,
                    "next": serializable_state["next"],
                    "error_count": serializable_state["error_count"],
                    "reasoning": serializable_state["reasoning"],
                    "memory_triples": serializable_state["memory_triples"]
                }
                return state
        return None

    def save_memories_to_disk(self) -> None:
        """Persist memories to disk."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f)

    def load_memories_from_disk(self) -> None:
        """Load memories from disk if available."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.memories = json.load(f)


class LangGraphMemoryCheckpointer(BaseCheckpointSaver):
    def __init__(self, memory_manager: 'MemoryManager', max_history: int = 5):
        self.mm = memory_manager
        self.max_history = max_history

    def get(self, config: RunnableConfig) -> Checkpoint:
        """Retrieve checkpoint with serialized SupervisorState."""
        user_id = config["configurable"].get("user_id", "global")
        state = self.mm.load_state(user_id)
        if state is None:
            state = {
                "messages": [],
                "next": "supervisor",
                "error_count": 0,
                "reasoning": None,
                "memory_triples": self.mm.search_memories(user_id, "recent", self.max_history)
            }
        # Serialize the state to a JSON string
        serialized_state = json.dumps(state, default=self._serialize_message)
        return Checkpoint(v={"serialized_state": serialized_state})

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        """Store checkpoint by deserializing the SupervisorState."""
        user_id = config["configurable"].get("user_id", "global")
        serialized_state = checkpoint["v"]["serialized_state"]
        # Deserialize the JSON string back to the state
        state = json.loads(serialized_state, object_hook=self._deserialize_message)
        if "memory_triples" in state:
            self.mm.save_memory(user_id, state["memory_triples"])
        state["memory_triples"] = self.mm.search_memories(user_id, "recent", self.max_history)
        self.mm.save_state(user_id, state)

    def _serialize_message(self, obj):
        """Helper to serialize BaseMessage objects."""
        if isinstance(obj, BaseMessage):
            return obj.to_json()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _deserialize_message(self, dct):
        """Helper to deserialize messages back to BaseMessage objects."""
        if "type" in dct and "content" in dct:
            message_type = dct["type"]
            if message_type == "human":
                return HumanMessage(content=dct["content"])
            elif message_type == "ai":
                return AIMessage(content=dct["content"], tool_calls=dct.get("tool_calls", []))
            elif message_type == "system":
                return SystemMessage(content=dct["content"])
            elif message_type == "tool":
                return ToolMessage(
                    content=dct["content"],
                    tool_call_id=dct.get("tool_call_id", ""),
                    name=dct.get("name", "")
                )
        return dct


memory_manager = MemoryManager()
checkpointer = LangGraphMemoryCheckpointer(memory_manager)

__all__ = ["MemoryManager", "LangGraphMemoryCheckpointer", "memory_manager", "checkpointer"]