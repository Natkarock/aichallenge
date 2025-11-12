from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from openai import OpenAI

from .indexer import ProjectIndexer
from .retriever import Retriever
from .llm import LLMAnalyzer
from .cache import EmbeddingCache
from .reranker import make_reranker


console = Console()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("project_root", type=str, help="Путь к корню проекта")
    parser.add_argument("--exclude-dirs", nargs="*", default=[])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--max-file-bytes", type=int, default=1_500_000)
    parser.add_argument("--max-file-chars", type=int, default=120_000)
    parser.add_argument("--scan-timeout-ms", type=int, default=800)
    parser.add_argument("--max-chunks-per-file", type=int, default=100)
    parser.add_argument("--no-gitignore", action="store_true")
    parser.add_argument(
        "--skip-lockfiles", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-minified", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-sourcemaps", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verbose-files", action="store_true")
    parser.add_argument("--report-md", type=str, default="analysis_report.md")
    parser.add_argument("--multi-stage", action="store_true")
    parser.add_argument("--module-top-k", type=int, default=36)
    parser.add_argument("--global-top-k", type=int, default=48)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")

    # Новый блок для реранкинга
    parser.add_argument(
        "--reranker", type=str, default="none", choices=["none", "cohere"]
    )
    parser.add_argument(
        "--preselect-factor",
        type=int,
        default=5,
        help="Во сколько раз шире брать пул кандидатов перед реранком (>=1)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Сколько оставить после реранка (если не задано, используется top-k)",
    )
    parser.add_argument(
        "--rerank-threshold",
        type=float,
        default=None,
        help="Фильтровать кандидаты с релевантностью ниже порога",
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="Сравнить baseline vs rerank (сохраняет оба отчёта и файл сравнения)",
    )


def main() -> None:
    parser = argparse.ArgumentParser("analyzer")
    _add_common_args(parser)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        console.print(f"[red]Путь не найден:[/] {root}")
        raise SystemExit(2)

    if "OPENAI_API_KEY" not in os.environ:
        console.print("[red]Требуется переменная окружения OPENAI_API_KEY[/]")
        raise SystemExit(1)

    client = OpenAI()

    console.print(Panel.fit(f"📁 Анализ проекта: [bold]{root}[/]"))

    indexer = ProjectIndexer(
        root=root,
        exclude_dirs=args.exclude_dirs,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_file_chars=args.max_file_chars,
        scan_timeout_ms=args.scan_timeout_ms,
        max_chunks_per_file=args.max_chunks_per_file,
        skip_lockfiles=args.skip_lockfiles,
        skip_minified=args.skip_minified,
        skip_sourcemaps=args.skip_sourcemaps,
        verbose=args.verbose_files,
        respect_gitignore=not args.no_gitignore,
    )

    chunks = indexer.build_chunks(max_chars=args.max_chars, overlap=args.overlap)
    if not chunks:
        console.print("[red]Не найдено файлов для анализа.[/]")
        raise SystemExit(1)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Загрузка кеша[/]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("llm", total=None)
        cache = (
            None
            if args.no_cache
            else EmbeddingCache(
                Path(args.cache_dir)
                if args.cache_dir
                else root / ".proj_analyzer_cache"
            )
        )
        progress.update(task, completed=1)

    # Реранкер
    rr = make_reranker(args.reranker) if args.reranker != "none" else None

    retriever = Retriever(
        client,
        args.embed_model,
        cache=cache,
        reranker=rr,
        preselect_factor=args.preselect_factor,
        rerank_top_k=args.rerank_top_k,
        rerank_threshold=args.rerank_threshold,
    )
    retriever.embed_chunks(chunks)

    analyzer = LLMAnalyzer(client, args.llm_model, max_ctx_chunks=args.top_k)

    # === Одноэтапный режим ===
    if not args.multi_stage:
        question = "Проанализируй структуру проекта и найди очевидные баги и риски."

        retriever_baseline = Retriever(client, args.embed_model, cache=cache)
        retriever_baseline.embed_chunks(chunks)

        top_baseline = retriever_baseline.query(chunks, question, top_k=args.top_k)
        top_rerank = retriever.query(chunks, question, top_k=args.top_k)

        md_baseline = analyzer.analyze_single(root, top_baseline)
        md_rerank = analyzer.analyze_single(root, top_rerank)

        if args.compare_rerank and rr is not None:
            (root / "analysis_report_baseline.md").write_text(
                md_baseline, encoding="utf-8"
            )
            (root / "analysis_report_rerank.md").write_text(md_rerank, encoding="utf-8")
            cmp_md = (
                "# Сравнение отчётов\n\n"
                "## Baseline и Rerank\n"
                f"- Реранкер: {getattr(rr, 'name', 'none')}\n"
                f"- Порог отсечения: {args.rerank_threshold}\n"
                f"- Предвыборка (factor): {args.preselect_factor}\n"
                "\n"
                "Файлы:\n"
                "- analysis_report_baseline.md\n"
                "- analysis_report_rerank.md\n"
            )
            (root / "analysis_report_comparison.md").write_text(
                cmp_md, encoding="utf-8"
            )
            console.print(Panel.fit("✅ Сохранены baseline, rerank и comparison."))
        else:
            (root / args.report_md).write_text(
                md_rerank if rr is not None else md_baseline, encoding="utf-8"
            )
            console.print(
                Panel.fit(f"✅ Готово. Отчёт сохранён: [bold]{args.report_md}[/]")
            )
        return

    # === Многоэтапный режим ===
    modules = indexer.detect_modules(chunks)
    module_summaries = []

    for mod_name, mod_chunks in modules.items():
        question_mod = f"Проанализируй модуль '{mod_name}' и найди архитектурные аспекты, баги и риски."
        top_mod = retriever.query(mod_chunks, question_mod, top_k=args.module_top_k)
        md_mod = analyzer.analyze_module(root, mod_name, top_mod)
        (reports_dir / f"module_{mod_name}.md").write_text(md_mod, encoding="utf-8")
        module_summaries.append(f"## {mod_name}\n" + md_mod[:1200])

    question_global = "Сформируй общую картину архитектуры проекта и приоритетные риски на основе всех частей."
    top_global = retriever.query(chunks, question_global, top_k=args.global_top_k)
    global_md = analyzer.analyze_global(root, "".join(module_summaries), top_global)
    (root / args.report_md).write_text(global_md, encoding="utf-8")
    console.print(
        Panel.fit(f"✅ Готово. Итоговый отчёт сохранён: [bold]{args.report_md}[/]")
    )
    console.print(Markdown("Модульные отчёты лежат в папке `reports/`."))


if __name__ == "__main__":
    main()
