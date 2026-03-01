"""Cross-encoder re-ranker: narrows the candidate pool to the top-K most relevant docs."""
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from config import settings


class Reranker:
    """
    Uses a lightweight cross-encoder model to score (query, passage) pairs
    and return only the top-K most relevant chunks.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    (~90 MB, fast inference, strong MS-MARCO relevance scores)
    """

    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        top_k = top_k or settings.rerank_top_k
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores: list[float] = self.model.predict(pairs).tolist()

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
