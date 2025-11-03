from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from openai import OpenAI
from .utils import DocChunk, cosine_sim
from .cache import EmbeddingCache

console = Console()

class Retriever:
    def __init__(self, client: OpenAI, embed_model: str, cache: EmbeddingCache | None = None):
        self.client = client
        self.embed_model = embed_model
        self.cache = cache

    def embed_chunks(self, chunks: List[DocChunk]) -> None:
        pending = []
        for c in chunks:
            if self.cache:
                vec = self.cache.get(c.path, c.idx, c.text)
                if vec is not None:
                    c.vector = np.array(vec, dtype=np.float32)
                    continue
            pending.append(c)

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
            to_store = []
            for i in range(0, len(pending), batch_size):
                batch = pending[i : i + batch_size]
                inputs = [c.text for c in batch]
                emb = self.client.embeddings.create(model=self.embed_model, input=inputs)
                for j, c in enumerate(batch):
                    vec = np.array(emb.data[j].embedding, dtype=np.float32)
                    c.vector = vec
                    if self.cache:
                        to_store.append((c.path, c.idx, c.text, vec.tolist()))
                progress.update(task, advance=len(batch))
            if self.cache and to_store:
                self.cache.put_many(to_store)

    def query(self, chunks: List[DocChunk], question: str, top_k: int) -> List[DocChunk]:
        q = self.client.embeddings.create(model=self.embed_model, input=[question]).data[0].embedding
        q_vec = np.array(q, dtype=np.float32)

        sims: List[Tuple[int, float]] = []
        total = len(chunks)
        step = max(1, total // 100)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Поиск релевантных фрагментов[/]"),
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
        return [chunks[i] for i, _ in sims[:top_k]]
