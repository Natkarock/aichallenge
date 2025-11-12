from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


# Minimal interface the retriever expects
@dataclass
class RerankItem:
    index: int
    text: str


class BaseReranker:
    name: str = "base"

    def rerank(self, query: str, items: List[RerankItem]) -> List[Tuple[int, float]]:
        """Return [(index, score)] sorted by score desc."""
        raise NotImplementedError


class NoopReranker(BaseReranker):
    name = "none"

    def rerank(self, query: str, items: List[RerankItem]) -> List[Tuple[int, float]]:
        # Keep original order, neutral score = 0.0
        return [(it.index, 0.0) for it in items]


class CohereReranker(BaseReranker):
    """
    Uses Cohere Rerank v3 (multilingual) if 'cohere' package is installed.
    Requires COHERE_API_KEY env.
    """

    name = "cohere"

    def __init__(
        self, model: str = "rerank-multilingual-v3.0", api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key

        try:
            import cohere  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Cohere package is not installed. Run: pip install cohere"
            ) from e
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self._client = cohere.Client(api_key=key)

    def rerank(self, query: str, items: List[RerankItem]) -> List[Tuple[int, float]]:
        docs = [it.text for it in items]
        resp = self._client.rerank(
            model=self.model, query=query, documents=docs, top_n=len(docs)
        )
        # resp.results[i].index points to the original docs index inside 'docs'
        # Map back to original 'items' indices
        out: List[Tuple[int, float]] = []
        for r in resp.results:
            orig_idx = items[r.index].index
            out.append((orig_idx, float(r.relevance_score)))
        # Cohere already in desc order
        return out


def make_reranker(kind: str, **kwargs) -> BaseReranker:
    kind = (kind or "none").lower()
    if kind in ("none", "noop", "off"):
        return NoopReranker()
    if kind in ("cohere", "cohere-v3", "cohere-multi"):
        return CohereReranker(**kwargs)
    raise ValueError(f"Unknown reranker kind: {kind}")
