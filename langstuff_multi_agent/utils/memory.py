from typing import List, TypedDict
import os
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import uuid


class MemoryTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


class MemoryManager:
    def __init__(self, persist_path="memory_store"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="agent_memories",
            embedding_function=self.embeddings,
            persist_directory=persist_path
        )

    def save_memory(self, user_id: str, memories: List[MemoryTriple]):
        docs = [
            Document(
                page_content=f"{m['subject']} {m['predicate']} {m['object_']}",
                metadata={"user_id": user_id, **m}
            ) for m in memories
        ]
        self.vector_store.add_documents(docs)

    def search_memories(self, user_id: str, query: str, k=3) -> List[str]:
        results = self.vector_store.similarity_search(
            query, k=k,
            filter={"user_id": user_id}
        )
        return [f"{doc.metadata['subject']} {doc.metadata['predicate']} {doc.metadata['object_']}" 
                for doc in results]
