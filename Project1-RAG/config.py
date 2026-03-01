from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096

    # --- Embeddings ---
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Vector DB ---
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "rag_docs"

    # --- Retrieval ---
    retrieval_top_k: int = 20   # candidates from hybrid search
    rerank_top_k: int = 5       # final docs passed to LLM

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Re-ranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
