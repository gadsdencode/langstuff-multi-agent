from typing import List, TypedDict, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from datetime import datetime, timedelta
from langchain_core.runnables.config import RunnableConfig
from langchain_community.vectorstores import Chroma


class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


class MemoryManager:
    def __init__(self, persist_path: str = "memory_store"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="agent_memories",
            embedding_function=self.embeddings,
            persist_directory=persist_path
        )

    def save_memory(self, user_id: str, memories: List[MemoryTriple]) -> None:
        docs = [
            Document(
                page_content=f"{m['subject']} {m['predicate']} {m['object_']}",
                metadata={
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    **m
                }
            ) for m in memories
        ]
        self.vector_store.add_documents(docs)

    def search_memories(
        self, user_id: str, query: str, k: int = 3
    ) -> List[Document]:
        return self.vector_store.similarity_search(
            query, k=k,
            filter={"user_id": user_id}
        )

    def delete_old_memories(self, user_id: str, days: int = 30) -> None:
        cutoff = datetime.now() - timedelta(days=days)
        self.vector_store.delete(
            filter={
                "user_id": user_id,
                "timestamp": {"$lt": cutoff.isoformat()}
            }
        )


class LangGraphMemoryCheckpointer(BaseCheckpointSaver):
    def __init__(self, memory_manager: MemoryManager, max_history: int = 5):
        self.mm = memory_manager
        self.max_history = max_history

    async def aget(
        self,
        config: RunnableConfig,
        *,
        thread_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        user_id = config["configurable"].get("user_id")
        if not user_id:
            return None

        memories = self.mm.search_memories(
            user_id=user_id,
            query="recent",
            k=self.max_history
        )
        return Checkpoint(
            {
                "memory_triples": [doc.metadata for doc in memories],
                "timestamp": datetime.now().isoformat()
            }
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint
    ) -> None:
        user_id = config["configurable"].get("user_id")
        if not user_id:
            return

        # Trim old memories before adding new
        self.mm.delete_old_memories(user_id)

        memories = [
            MemoryTriple(
                subject=item["subject"],
                predicate=item["predicate"],
                object_=item["object_"]
            ) for item in checkpoint["memory_triples"]
        ]
        self.mm.save_memory(user_id, memories)


class Config:
    PERSISTENT_CHECKPOINTER: Optional[LangGraphMemoryCheckpointer] = None

    @classmethod
    def init_checkpointer(cls, memory_manager: MemoryManager) -> None:
        cls.PERSISTENT_CHECKPOINTER = LangGraphMemoryCheckpointer(memory_manager)
