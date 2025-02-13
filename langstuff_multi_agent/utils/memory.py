from typing import List, TypedDict, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
import uuid
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from datetime import datetime, timedelta
from langchain_core.runnables.config import RunnableConfig


class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


class LangGraphMemoryCheckpointer(BaseCheckpointSaver):
    def __init__(self, memory_manager, max_history=5):
        self.mm = memory_manager
        self.max_history = max_history

    async def aget(self, config: RunnableConfig, *, thread_id: Optional[str] = None) -> Optional[Checkpoint]:
        user_id = config["configurable"].get("user_id")
        if not user_id:
            return None

        memories = self.mm.vector_store.similarity_search(
            query="recent",
            k=self.max_history,
            filter={"user_id": user_id}
        )
        return Checkpoint(
            {
                "memory_triples": [doc.metadata for doc in memories],
                "timestamp": datetime.now().isoformat()
            }
        )

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        user_id = config["configurable"].get("user_id")
        if not user_id:
            return

        # Trim old memories before adding new
        self._expire_old_memories(user_id)

        memories = [
            MemoryTriple(
                subject=item["subject"],
                predicate=item["predicate"],
                object_=item["object_"]
            ) for item in checkpoint["memory_triples"]
        ]
        self.mm.save_memory(user_id, memories)

    def _expire_old_memories(self, user_id: str, days=30):
        cutoff = datetime.now() - timedelta(days=days)
        self.mm.vector_store.delete(
            filter={
                "user_id": user_id,
                "timestamp": {"$lt": cutoff.isoformat()}
            }
        )


class Config:
    # ... existing config ...
    @classmethod
    def init_checkpointer(cls, memory_manager: MemoryManager):
        cls.PERSISTENT_CHECKPOINTER = LangGraphMemoryCheckpointer(memory_manager)
