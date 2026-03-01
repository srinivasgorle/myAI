"""Split documents into overlapping chunks for embedding."""
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents using recursive character splitting.

    Separators are tried in order: paragraph → line → sentence → word → char.
    Each chunk keeps metadata from its source document plus a chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
