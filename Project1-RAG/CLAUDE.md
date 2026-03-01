# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Getting Started (First-Time Setup)

### Step 1 — Get Your API Keys

**Anthropic (Claude) key:**
1. Go to [console.anthropic.com](https://console.anthropic.com) and sign in
2. Click **API Keys** in the left sidebar → **Create Key**
3. Copy the key (starts with `sk-ant-...`)

**OpenAI key** (used only for embeddings, not the chat model):
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and sign in
2. Click **Create new secret key** → copy it (starts with `sk-...`)

> Why two keys? Claude does the reasoning. OpenAI's `text-embedding-3-small` converts documents into searchable vectors (~$0.0001 per 1000 words).

---

### Step 2 — Configure Environment

```bash
cd /Users/srinivasgorle/Git/myAI/Project1-RAG
cp .env.example .env
```

Open `.env` and fill in your two keys:

```
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

Leave everything else at the defaults.

---

### Step 3 — Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all packages (takes 3–5 min)
pip install -r requirements.txt
```

You'll see `(.venv)` in your terminal prompt when the environment is active. The cross-encoder re-ranker (~90 MB) downloads automatically on first use and caches to `~/.cache/huggingface/`.

---

## Feeding TikTok Shop API Documentation

### Step 4 — Ingest TikTok Doc Pages by URL

Run the ingest command once per page you want the agent to know about:

```bash
python main.py ingest "https://partner.tiktokshop.com/docv2/page/6550b13e43f3cb02e98e6400"
python main.py ingest "https://partner.tiktokshop.com/docv2/page/650a3a7aef547802b9e0021f"
python main.py ingest "https://partner.tiktokshop.com/docv2/page/6502f45e4a0bb702c7ddfe14"
```

Each run prints a confirmation:
```
Ingesting: https://partner.tiktokshop.com/...
✓ Stored 18 chunks from 1 document(s).
```

**If a page returns empty** (TikTok docs are JavaScript-rendered):
1. Open the page in your browser
2. Select all (`Cmd+A`), copy, paste into a file e.g. `tiktok_auth.txt`
3. Ingest the file instead:

```bash
python main.py ingest tiktok_auth.txt --type text
```

You can also ingest PDFs or entire folders:

```bash
python main.py ingest ./tiktok_docs/          # folder of files
python main.py ingest tiktok_reference.pdf    # PDF file
```

### Step 5 — Verify the Database

```bash
python -c "
from ingestion.embedder import Embedder
e = Embedder()
docs = e.get_all_documents()
print(f'Total chunks stored: {len(docs)}')
print('---')
print(docs[0].page_content[:300])
"
```

You should see a count greater than 0 and a text preview. If it shows 0, nothing has been ingested yet.

---

## Testing

### Step 6 — Chat via CLI (quickest test)

```bash
python main.py chat
```

The prompt looks like this:
```
Agent ready! Session: cli-session  Type quit to exit.

You:
```

Try these to confirm the TikTok docs were ingested correctly:
```
You: What authentication method does TikTok Shop API use?
You: How do I create an order using the API?
You: What are the required fields for the product listing endpoint?
You: What HTTP methods does the TikTok Shop API support?
```

The agent searches your stored docs first, falls back to a live web search if needed, and always cites the source.

Type `quit` to exit.

### Step 7 — Test via API Server

Open two terminals:

**Terminal 1 — start the server:**
```bash
python main.py serve
```

**Terminal 2 — send requests:**
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the TikTok Shop API base URL?", "session_id": "test-1"}'

# Ingest a new URL without stopping the server
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://partner.tiktokshop.com/docv2/page/6550b13e43f3cb02e98e6400"}'

# Upload a PDF
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@tiktok_reference.pdf"
```

Or open `http://localhost:8000/docs` in your browser for the interactive Swagger UI.

---

## Quick Reference

| Command | What it does |
|---|---|
| `python main.py ingest <url or path>` | Feeds a document into the local database |
| `python main.py chat` | Opens an interactive Q&A session |
| `python main.py serve` | Starts the HTTP API server on port 8000 |
| `python main.py serve --reload` | Same, with hot-reload for development |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ANTHROPIC_API_KEY missing` | Check that `.env` exists and the key is filled in (not the example value) |
| `Total chunks stored: 0` | Run the ingest command before chatting |
| TikTok page ingests but answers are wrong/empty | Page is JS-rendered — copy-paste text into a `.txt` file and ingest that |
| `ModuleNotFoundError` | Run `source .venv/bin/activate` first, then retry |
| First chat response is slow | Re-ranker model is downloading (~90 MB) — wait once, it caches after |
| `BM25Okapi` error on startup | Database is empty — ingest at least one document first |

---

## Architecture

The system has four distinct layers that are deliberately decoupled:

### 1. Ingestion Pipeline (`ingestion/`)
`loader.py → chunker.py → embedder.py`

Documents are loaded from any source type, split into 512-token overlapping chunks, embedded with OpenAI, and persisted in ChromaDB at `./data/chroma`. This pipeline is one-way — ingestion is separate from retrieval.

### 2. Retrieval Pipeline (`retrieval/`)
`retriever.py → reranker.py`

`HybridRetriever` runs two parallel searches — ChromaDB vector search (dense) and BM25 keyword search (sparse) — then merges them with Reciprocal Rank Fusion. The merged top-20 candidates go to `Reranker`, which uses a CrossEncoder to score each `(query, passage)` pair and returns the top-5. The BM25 index is held in memory and rebuilt by calling `retriever.update_documents()` after any new ingestion.

### 3. Agent Layer (`agent/`)
`tools.py → orchestrator.py` wired together by `memory.py`

Tools are module-level functions decorated with `@tool` and hold `_retriever` / `_reranker` as injected globals set at startup via `set_retriever()`. The agent is a LangGraph `create_react_agent` ReAct loop (Thought → Action → Observation) backed by `ChatAnthropic`. Conversation state is checkpointed per `thread_id` using `MemorySaver` — swap for `PostgresSaver`/`RedisSaver` in production.

### 4. Entry Points (`main.py`, `api/server.py`)
- **CLI** (`main.py`): Typer commands that initialise all layers inline before running.
- **API** (`api/server.py`): FastAPI app that initialises all layers once in the `lifespan` context. After any `/ingest/*` call, `_rebuild_bm25()` refreshes the BM25 index. Streaming is supported on `POST /chat` via `stream=true`.

### Key Wiring Detail
The tools in `agent/tools.py` depend on live retriever instances. Startup order must always be:
1. `Embedder()` → `get_vector_store()`, `get_all_documents()`
2. `HybridRetriever(vector_store, all_docs)` + `Reranker()`
3. `set_retriever(retriever, reranker)` — injects into tool module globals
4. `create_rag_agent(get_all_tools(), memory)`

Breaking this order causes tools to return "Knowledge base not initialised."

---

## Configuration Reference

All settings live in `config.py` and are overridable via `.env`:

| Setting | Default | Effect |
|---|---|---|
| `RETRIEVAL_TOP_K` | 20 | Candidates from hybrid search before re-ranking |
| `RERANK_TOP_K` | 5 | Final chunks passed to the LLM |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection — change to isolate datasets |

---

## Adding a New Tool

1. Define a function in `agent/tools.py` decorated with `@tool`.
2. Add it to the list returned by `get_all_tools()`.
3. Add a one-line description to the system prompt in `agent/orchestrator.py`.

No other files need changing.
