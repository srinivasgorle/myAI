"""Hybrid retriever: dense (vector) + sparse (BM25) with Reciprocal Rank Fusion."""
from langchain.schema import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from config import settings


class HybridRetriever:
    """
    Combines ChromaDB vector search (dense) with BM25 keyword search (sparse).
    Results from both are merged via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store: Chroma, all_docs: list[Document]) -> None:
        self.vector_store = vector_store
        self.all_docs = all_docs
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        top_k = top_k or settings.retrieval_top_k
        dense = self._dense_search(query, top_k)
        sparse = self._sparse_search(query, top_k)
        return self._reciprocal_rank_fusion(dense, sparse)[:top_k]

    def update_documents(self, all_docs: list[Document]) -> None:
        """Rebuild BM25 index after new documents are ingested."""
        self.all_docs = all_docs
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        tokenized = [doc.page_content.lower().split() for doc in self.all_docs]
        self.bm25: BM25Okapi | None = BM25Okapi(tokenized) if tokenized else None

    def _dense_search(self, query: str, k: int) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def _sparse_search(self, query: str, k: int) -> list[Document]:
        if not self.bm25 or not self.all_docs:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.all_docs[i] for i in top_indices]

    def _reciprocal_rank_fusion(
        self, *result_lists: list[Document], k: int = 60
    ) -> list[Document]:
        """
        RRF score = Σ 1 / (rank + k) across all result lists.
        Documents appearing in multiple lists are ranked higher.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                key = doc.page_content[:120]  # stable dedup key
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
                doc_map[key] = doc

        ranked_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in ranked_keys]
