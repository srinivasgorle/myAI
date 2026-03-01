"""Embed documents and persist them in ChromaDB."""
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import settings


class Embedder:
    """Wraps ChromaDB + OpenAI embeddings for storage and retrieval."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    def embed_and_store(self, chunks: list[Document]) -> int:
        """Embed chunks and add them to the vector store. Returns count stored."""
        if not chunks:
            return 0
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def get_vector_store(self) -> Chroma:
        return self.vector_store

    def get_all_documents(self) -> list[Document]:
        """Load every document from ChromaDB (used to build the BM25 index)."""
        result = self.vector_store._collection.get(
            include=["documents", "metadatas"]
        )
        return [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(result["documents"], result["metadatas"])
        ]
