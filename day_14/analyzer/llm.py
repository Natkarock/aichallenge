from __future__ import annotations

from pathlib import Path
from typing import List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from openai import OpenAI

from .utils import DocChunk

console = Console()

SYSTEM_PROMPT_RU = """Вы — ведущий архитектор программного обеспечения.
Проанализируйте репозиторий кода и подготовьте отчёт:
1) Кратко опишите высокоуровневую архитектуру (модули, слои, зависимости, поток данных).
2) Укажите очевидные баги, анти‑паттерны и риски (по конкретным файлам, если возможно).
3) Дайте рекомендации по рефакторингу, структуре каталогов/файлов, тестам, CI/CD, производительности и безопасности.
Формат ответа — сжато, по разделам и списками, на русском языке.
"""

ANALYZE_PROMPT_RU = """Корень проекта: {root}

Ниже — N топ‑релевантных фрагментов (чанков) из репозитория. Используйте их как контекст.
Если ссылаетесь на код/конфиг, указывайте путь к файлу.

### Контекст (top chunks):
{context}

### Задача
1) Обзор архитектуры (модули, слои, поток данных, ключевые зависимости).
2) Находим баги/риски (желательно привязывая к файлам).
3) Предлагаем структуру и следующие шаги (рефакторинг, тесты, CI, производительность, безопасность).

Ответ дайте в Markdown, на русском, с разделами и маркерами.
"""

class _LineStreamer:
    """Буферизованный стример: печатает построчно с корректными переносами."""
    def __init__(self, console: Console):
        self.console = console
        self._buf = ""

    def feed(self, delta: str):
        if not delta:
            return
        delta = delta.replace("\r\n", "\n").replace("\r", "\n")
        self._buf += delta
        if "\n" in self._buf:
            parts = self._buf.split("\n")
            for line in parts[:-1]:
                self.console.print(line)
            self._buf = parts[-1]

    def flush(self):
        if self._buf:
            self.console.print(self._buf)
            self._buf = ""

class LLMAnalyzer:
    def __init__(self, client: OpenAI, model: str, max_ctx_chunks: int):
        self.client = client
        self.model = model
        self.max_ctx_chunks = max_ctx_chunks

    def _build_context(self, chunks: List[DocChunk]) -> str:
        parts = []
        for c in chunks[: self.max_ctx_chunks]:
            header = f"\n---\n# {c.path} [chunk {c.idx}]\n"
            snippet = c.text[:4000]
            parts.append(header + snippet)
        return "".join(parts)

    def analyze_streaming(self, root: Path, top_chunks: List[DocChunk]) -> str:
        context = self._build_context(top_chunks)
        user_prompt = ANALYZE_PROMPT_RU.format(root=str(root), context=context)

        console.print(Panel.fit("⚙️  Запускаю LLM анализ (stream)..."))
        stream = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_RU},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        out = []
        streamer = _LineStreamer(console)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]LLM размышляет[/] (stream)"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("llm", total=None)
            try:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                streamer.feed(delta)
                                out.append(delta)
                        elif event.type == "response.completed":
                            break
                        elif event.type == "response.error":
                            console.print(f"[red]LLM error:[/] {getattr(event, 'error', '')}")
                            break
                progress.update(task, completed=1)
            finally:
                streamer.flush()

        return "".join(out)
