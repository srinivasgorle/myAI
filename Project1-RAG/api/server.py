"""
FastAPI server exposing the RAG agent over HTTP.

Endpoints
---------
GET  /health          — liveness probe
POST /chat            — send a message, get a response
POST /ingest/text     — ingest raw text
POST /ingest/url      — ingest a web page
POST /ingest/pdf      — upload and ingest a PDF file
"""
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.memory import create_memory
from agent.orchestrator import create_rag_agent
from agent.tools import get_all_tools, set_retriever
from ingestion.chunker import chunk_documents
from ingestion.embedder import Embedder
from ingestion.loader import load_pdf, load_text, load_url
from retrieval.reranker import Reranker
from retrieval.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Global singletons (initialised in lifespan)
# ---------------------------------------------------------------------------
embedder: Embedder | None = None
retriever: HybridRetriever | None = None
reranker: Reranker | None = None
agent = None
memory = None


def _init_agent() -> None:
    global embedder, retriever, reranker, agent, memory
    embedder = Embedder()
    all_docs = embedder.get_all_documents()
    reranker = Reranker()
    retriever = HybridRetriever(embedder.get_vector_store(), all_docs)
    set_retriever(retriever, reranker)
    memory = create_memory()
    agent = create_rag_agent(get_all_tools(), memory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_agent()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Agent API",
    version="1.0.0",
    description="Retrieval-Augmented Generation agent backed by Claude + ChromaDB",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"
    metadata: dict[str, Any] = {}


class IngestUrlRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    chunks_stored: int
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebuild_bm25() -> None:
    """Refresh the BM25 index after new content is ingested."""
    all_docs = embedder.get_all_documents()
    retriever.update_documents(all_docs)


def _run_agent(message: str, session_id: str) -> str:
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [("user", message)]}, config=config)
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    if request.stream:
        # SSE-style streaming
        config = {"configurable": {"thread_id": request.session_id}}

        async def event_stream():
            for chunk in agent.stream(
                {"messages": [("user", request.message)]}, config=config
            ):
                if "agent" in chunk:
                    msgs = chunk["agent"].get("messages", [])
                    for m in msgs:
                        if hasattr(m, "content") and m.content:
                            yield m.content

        return StreamingResponse(event_stream(), media_type="text/plain")

    response = _run_agent(request.message, request.session_id)
    return ChatResponse(response=response, session_id=request.session_id)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    docs = load_text(request.text, source=request.source, metadata=request.metadata)
    chunks = chunk_documents(docs)
    n = embedder.embed_and_store(chunks)
    _rebuild_bm25()
    return IngestResponse(chunks_stored=n, message=f"Stored {n} chunks from text.")


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestUrlRequest):
    docs = load_url(request.url)
    chunks = chunk_documents(docs)
    n = embedder.embed_and_store(chunks)
    _rebuild_bm25()
    return IngestResponse(chunks_stored=n, message=f"Stored {n} chunks from {request.url}.")


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = load_pdf(tmp_path)
        chunks = chunk_documents(docs)
        n = embedder.embed_and_store(chunks)
        _rebuild_bm25()
        return IngestResponse(
            chunks_stored=n,
            message=f"Stored {n} chunks from '{file.filename}'.",
        )
    finally:
        os.unlink(tmp_path)
