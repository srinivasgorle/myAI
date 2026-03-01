"""Document loaders for PDF, URL, raw text, and directories."""
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader


def load_pdf(path: str) -> list[Document]:
    """Load a PDF file and return a list of Documents (one per page)."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["source"] = str(path)
    return docs


def load_url(url: str) -> list[Document]:
    """Fetch a web page and return its text as a Document."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "url"
        doc.metadata["source"] = url
    return docs


def load_text(
    text: str,
    source: str = "manual",
    metadata: dict | None = None,
) -> list[Document]:
    """Wrap raw text as a Document."""
    meta = {"source_type": "text", "source": source}
    if metadata:
        meta.update(metadata)
    return [Document(page_content=text, metadata=meta)]


def load_directory(
    dir_path: str,
    extensions: list[str] | None = None,
) -> list[Document]:
    """Recursively load all supported files from a directory."""
    if extensions is None:
        extensions = [".pdf", ".txt", ".md"]

    docs: list[Document] = []
    for file in Path(dir_path).rglob("*"):
        if file.suffix not in extensions:
            continue
        if file.suffix == ".pdf":
            docs.extend(load_pdf(str(file)))
        else:
            text = file.read_text(encoding="utf-8", errors="ignore")
            docs.extend(load_text(text, source=str(file)))
    return docs
