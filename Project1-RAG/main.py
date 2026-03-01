"""
RAG Agent CLI entry point.

Commands
--------
  ingest  – ingest a file, directory, or URL into the knowledge base
  chat    – start an interactive chat session
  serve   – launch the FastAPI HTTP server
"""
import typer
from rich.console import Console
from rich.prompt import Prompt

app_cli = typer.Typer(help="RAG Agent — ingest documents and chat with Claude.")
console = Console()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app_cli.command()
def ingest(
    source: str = typer.Argument(..., help="File path, directory path, or URL to ingest."),
    source_type: str = typer.Option(
        "auto",
        "--type", "-t",
        help="Source type: auto | pdf | url | directory | text",
    ),
):
    """Ingest documents from a file, directory, or URL into the knowledge base."""
    from ingestion.chunker import chunk_documents
    from ingestion.embedder import Embedder
    from ingestion.loader import load_directory, load_pdf, load_text, load_url

    console.print(f"[bold blue]Ingesting:[/] {source}")

    # Auto-detect source type
    if source_type == "auto":
        if source.startswith("http://") or source.startswith("https://"):
            source_type = "url"
        elif source.endswith(".pdf"):
            source_type = "pdf"
        else:
            source_type = "directory"

    if source_type == "pdf":
        docs = load_pdf(source)
    elif source_type == "url":
        docs = load_url(source)
    elif source_type == "directory":
        docs = load_directory(source)
    else:
        docs = load_text(source, source="cli-input")

    chunks = chunk_documents(docs)
    emb = Embedder()
    n = emb.embed_and_store(chunks)

    console.print(
        f"[bold green]✓[/] Stored [bold]{n}[/] chunks from [bold]{len(docs)}[/] document(s)."
    )


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app_cli.command()
def chat(
    session_id: str = typer.Option(
        "cli-session",
        "--session", "-s",
        help="Session ID for multi-turn memory.",
    ),
):
    """Start an interactive chat session with the RAG agent."""
    from agent.memory import create_memory
    from agent.orchestrator import create_rag_agent
    from agent.tools import get_all_tools, set_retriever
    from ingestion.embedder import Embedder
    from retrieval.reranker import Reranker
    from retrieval.retriever import HybridRetriever

    console.print("[bold blue]Initialising RAG Agent…[/]")

    emb = Embedder()
    all_docs = emb.get_all_documents()
    reranker = Reranker()
    retriever = HybridRetriever(emb.get_vector_store(), all_docs)
    set_retriever(retriever, reranker)

    memory = create_memory()
    agent = create_rag_agent(get_all_tools(), memory)

    console.print(
        f"[bold green]Agent ready![/] Session: [cyan]{session_id}[/]  "
        "Type [bold]quit[/] to exit.\n"
    )

    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = Prompt.ask("[bold cyan]You[/]")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        result = agent.invoke(
            {"messages": [("user", user_input)]},
            config=config,
        )
        response = result["messages"][-1].content
        console.print(f"\n[bold green]Agent:[/] {response}\n")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app_cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable hot-reload (dev only)."),
):
    """Launch the FastAPI HTTP server."""
    import uvicorn

    console.print(f"[bold blue]Starting server on http://{host}:{port}[/]")
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app_cli()
