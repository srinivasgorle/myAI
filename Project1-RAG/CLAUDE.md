# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

Two API keys are required — `ANTHROPIC_API_KEY` drives agent reasoning (Claude), `OPENAI_API_KEY` drives embeddings (`text-embedding-3-small`). Everything else in `.env` has sensible defaults.

The cross-encoder re-ranker (`sentence-transformers`) downloads ~90 MB on first run and caches to `~/.cache/huggingface/`.

## Common Commands

```bash
# Ingest documents (auto-detects pdf / url / directory)
python main.py ingest ./docs/
python main.py ingest report.pdf
python main.py ingest https://example.com

# Interactive chat (multi-turn, persisted in memory per session)
python main.py chat --session my-session

# Start HTTP API server
python main.py serve --port 8000 --reload   # --reload for dev hot-reload

# Verify what is currently in the vector DB
python -c "from ingestion.embedder import Embedder; e=Embedder(); docs=e.get_all_documents(); print(len(docs), 'chunks')"
```

API docs available at `http://localhost:8000/docs` when the server is running.

## Architecture

The system has four distinct layers that are deliberately decoupled:

### 1. Ingestion Pipeline (`ingestion/`)
`loader.py → chunker.py → embedder.py`

Documents are loaded from any source type, split into 512-token overlapping chunks, then embedded with OpenAI and persisted in ChromaDB at `./data/chroma`. This pipeline is one-way — ingestion is separate from retrieval.

### 2. Retrieval Pipeline (`retrieval/`)
`retriever.py → reranker.py`

`HybridRetriever` runs two parallel searches — ChromaDB vector search (dense) and BM25 keyword search (sparse) — then merges them with **Reciprocal Rank Fusion**. The merged top-20 candidates are passed to `Reranker`, which uses a CrossEncoder model to score each `(query, passage)` pair and returns the top-5. The BM25 index is held in memory and rebuilt by calling `retriever.update_documents()` after any new ingestion.

### 3. Agent Layer (`agent/`)
`tools.py → orchestrator.py` wired together by `memory.py`

Tools are module-level functions decorated with `@tool` and hold `_retriever` / `_reranker` as injected globals set at startup via `set_retriever()`. The agent is a LangGraph `create_react_agent` ReAct loop (Thought → Action → Observation) backed by `ChatAnthropic`. Conversation state is checkpointed per `thread_id` using `MemorySaver` — swap this for `PostgresSaver`/`RedisSaver` for production.

### 4. Entry Points (`main.py`, `api/server.py`)
- **CLI** (`main.py`): Typer commands that initialise all layers inline before running.
- **API** (`api/server.py`): FastAPI app that initialises all layers once in the `lifespan` context and holds them as module-level globals. After any `/ingest/*` call, `_rebuild_bm25()` refreshes the BM25 index. Streaming is supported on `POST /chat` via `stream=true`.

### Key Wiring Detail
The tools in `agent/tools.py` depend on live retriever instances. The call order at startup must always be:
1. `Embedder()` → `get_vector_store()`, `get_all_documents()`
2. `HybridRetriever(vector_store, all_docs)` + `Reranker()`
3. `set_retriever(retriever, reranker)`  ← injects into tool module globals
4. `create_rag_agent(get_all_tools(), memory)`

Breaking this order means tools will return "Knowledge base not initialised."

## Configuration Reference

All settings live in `config.py` via `pydantic-settings` and are overridable via `.env`:

| Setting | Default | Effect |
|---|---|---|
| `RETRIEVAL_TOP_K` | 20 | Candidates from hybrid search before re-ranking |
| `RERANK_TOP_K` | 5 | Chunks actually passed to the LLM |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection — change to isolate datasets |

## Adding a New Tool

1. Define a function in `agent/tools.py` decorated with `@tool`.
2. Add it to the list returned by `get_all_tools()`.
3. Add a one-line description to the system prompt in `agent/orchestrator.py`.

No other files need changing.
