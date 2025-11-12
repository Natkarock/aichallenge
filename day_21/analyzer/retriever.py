from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from openai import OpenAI

from .utils import DocChunk, cosine_sim
from .cache import EmbeddingCache
from .reranker import BaseReranker, RerankItem

console = Console()


class Retriever:
    """
    Two-stage retrieval:
      1) Vector similarity preselect (fast)
      2) Optional re-ranking (e.g., Cohere) + relevance filtering
    Backward compatible: if reranker is None -> old single-stage behavior.
    """

    def __init__(
        self,
        client: OpenAI,
        embed_model: str,
        cache: EmbeddingCache | None = None,
        reranker: BaseReranker | None = None,
        preselect_factor: int = 5,
        rerank_top_k: int | None = None,
        rerank_threshold: float | None = None,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.cache = cache
        self.reranker = reranker
        self.preselect_factor = max(1, preselect_factor)
        self.rerank_top_k = rerank_top_k
        self.rerank_threshold = rerank_threshold

    # ====== Embeddings ======
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        res = self.client.embeddings.create(model=self.embed_model, input=texts)
        vectors = np.array([d.embedding for d in res.data], dtype=np.float32)
        return vectors

    def embed_chunks(self, chunks: List[DocChunk]) -> None:
        """Embed chunks in-place, using cache if provided."""
        pending: List[DocChunk] = []
        hits = 0
        ready = 0

        # Отображаем прогресс сканирования кэша — чтобы не было «тишины»
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Проверяю кэш эмбеддингов[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("cache", total=len(chunks))
            for c in chunks:
                if c.vector is not None:
                    ready += 1
                elif self.cache:
                    vec = self.cache.get(c.path, c.idx, c.text)
                    if vec is not None:
                        c.vector = np.array(vec, dtype=np.float32)
                        hits += 1
                    else:
                        pending.append(c)
                else:
                    pending.append(c)
                progress.update(task, advance=1)

        console.print(
            f"[dim]Кэш эмбеддингов: готово={ready}, cache_hit={hits}, на расчёт={len(pending)}[/]"
        )

        if not pending:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Создаю эмбеддинги[/] (кэшируем)"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("emb", total=len(pending))
            batch_size = 16
            to_store: list[tuple[str, int, str, list]] = []
            for i in range(0, len(pending), batch_size):
                batch = pending[i : i + batch_size]
                inputs = [c.text for c in batch]
                emb = self.client.embeddings.create(
                    model=self.embed_model, input=inputs
                )
                for j, c in enumerate(batch):
                    vec = np.array(emb.data[j].embedding, dtype=np.float32)
                    c.vector = vec
                    if self.cache:
                        to_store.append((str(c.path), c.idx, c.text, vec.tolist()))
                progress.update(task, advance=len(batch))
            if self.cache and to_store:
                self.cache.put_many(to_store)

    # ====== Retrieval ======
    def _preselect_by_cosine(
        self, chunks: List[DocChunk], query: str, k: int
    ) -> List[int]:
        q_vec = self._embed_texts([query])[0]

        sims: List[Tuple[int, float]] = []
        total = len(chunks)
        step = max(1, total // 50)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]cosine[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("sim", total=total)
            for idx, c in enumerate(chunks):
                if c.vector is None:
                    continue
                sims.append((idx, cosine_sim(q_vec, c.vector)))
                if (idx + 1) % step == 0 or idx + 1 == total:
                    progress.update(task, advance=step)

        sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in sims[:k]]

    def query(self, chunks: List[DocChunk], query: str, top_k: int) -> List[DocChunk]:
        """Main entry used by CLI. Keeps backward compatibility if reranker is None."""
        if self.reranker is None:
            # Old behavior: single-stage cosine top_k
            pre_idx = self._preselect_by_cosine(chunks, query, min(top_k, len(chunks)))
            return [chunks[i] for i in pre_idx]

        # Stage 1: preselect wider pool
        pool_k = max(top_k * self.preselect_factor, top_k)
        pre_idx = self._preselect_by_cosine(chunks, query, min(pool_k, len(chunks)))
        items = [RerankItem(index=i, text=chunks[i].text) for i in pre_idx]

        # Stage 2: rerank with visible spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Reranking candidates[/]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task("rr", total=None)
            pairs = self.reranker.rerank(query, items)  # [(chunk_index, score)]

        # Optional filtering by threshold
        if self.rerank_threshold is not None:
            pairs = [(i, s) for (i, s) in pairs if s >= self.rerank_threshold]

        # Final top-k
        use_k = self.rerank_top_k or top_k
        chosen = pairs[:use_k]

        # Pretty-print results table
        try:
            table = Table(title="Top после реранка", show_lines=False)
            table.add_column("#", justify="right", style="bold")
            table.add_column("score", justify="right")
            table.add_column("file:index", overflow="fold")
            table.add_column("preview", overflow="fold")
            for rank, (ci, score) in enumerate(chosen, start=1):
                ch = chunks[ci]
                preview = (
                    (ch.text.replace("\n", " ")[:80] + "…")
                    if len(ch.text) > 80
                    else ch.text
                )
                table.add_row(str(rank), f"{score:.3f}", f"{ch.path}:{ch.idx}", preview)
            console.print(table)
        except Exception:
            pass

        return [chunks[i] for i, _ in chosen]
