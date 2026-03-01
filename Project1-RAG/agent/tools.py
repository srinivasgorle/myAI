"""
LangChain tools exposed to the RAG agent.

Tools are created with `@tool` and injected with live retriever/reranker
instances at startup via `set_retriever()`.
"""
from langchain.schema import Document
from langchain.tools import tool

# Module-level singletons injected at startup
_retriever = None
_reranker = None


def set_retriever(retriever, reranker) -> None:
    """Inject retriever + reranker after they are initialised."""
    global _retriever, _reranker
    _retriever = retriever
    _reranker = reranker


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

@tool
def retrieve_docs(query: str) -> str:
    """
    Search the private knowledge base for relevant information.
    Always call this first for domain-specific or document-related questions.
    """
    if _retriever is None:
        return "Knowledge base not initialised. Ingest documents first."

    docs: list[Document] = _retriever.retrieve(query)
    if _reranker:
        docs = _reranker.rerank(query, docs)

    if not docs:
        return "No relevant documents found in the knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[Doc {i}] {source}" + (f" p.{page}" if page else "")
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent or real-time information not found in the knowledge base.
    Use when retrieve_docs returns insufficient results.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(
                    f"**{r['title']}**\n{r['body']}\nSource: {r['href']}"
                )

        return "\n\n---\n\n".join(results) if results else "No web results found."
    except Exception as exc:
        return f"Web search error: {exc}"


@tool
def summarize_text(text: str) -> str:
    """
    Condense a long passage to its key points.
    Use when a retrieved chunk is very long and only a summary is needed.
    """
    words = text.split()
    if len(words) <= 200:
        return text  # short enough — return as-is

    # Heuristic: first 10 sentences as a quick summary
    sentences = text.replace("\n", " ").split(". ")
    return ". ".join(sentences[:10]).strip() + "…"


def get_all_tools() -> list:
    return [retrieve_docs, web_search, summarize_text]
