from typing import List, TypedDict
import os
from langchain_community.vectorstores import FAISS
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
        self.persist_path = persist_path
        
        # Modified FAISS initialization with version compatibility
        if os.path.exists(os.path.join(persist_path, "index.faiss")):
            self.vector_store = FAISS.load_local(persist_path, self.embeddings)
        else:
            # Create new index with empty documents
            self.vector_store = FAISS.from_texts(
                texts=[""], 
                embedding=self.embeddings
            )
            self.vector_store.save_local(persist_path)
            
    def save_memory(self, user_id: str, memories: List[MemoryTriple]):
        docs = [
            Document(
                page_content=f"{m['subject']} {m['predicate']} {m['object_']}",
                metadata={"user_id": user_id, **m}
            ) for m in memories
        ]
        self.vector_store.add_documents(docs)
        # Ensure directory exists before saving
        os.makedirs(self.persist_path, exist_ok=True)
        self.vector_store.save_local(self.persist_path)

    def search_memories(self, user_id: str, query: str, k=3) -> List[str]:
        results = self.vector_store.similarity_search(
            query, k=k,
            filter=lambda doc: doc.metadata.get("user_id") == user_id
        )
        return [f"{doc.metadata['subject']} {doc.metadata['predicate']} {doc.metadata['object_']}" 
                for doc in results]