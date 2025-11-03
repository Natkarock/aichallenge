from __future__ import annotations

import argparse, os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from openai import OpenAI

from .indexer import ProjectIndexer
from .retriever import Retriever
from .llm import LLMAnalyzer
from .cache import EmbeddingCache

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Анализ любого проекта: устойчивое сканирование, кэш эмбеддингов, семантическая выборка, многоэтапный LLM (русские промпты)."
    )
    parser.add_argument("project_root", type=str, help="Путь к корню проекта")
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=[],
        help="Доп. имена директорий для исключения (добавляются к встроенным)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Ограничить общее число файлов"
    )
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--top-k",
        type=int,
        default=48,
        help="Сколько чанков передавать LLM (в одноэтапном режиме)",
    )
    parser.add_argument(
        "--module-top-k",
        type=int,
        default=36,
        help="Сколько чанков отдавать на модуль (многоэтапный)",
    )
    parser.add_argument(
        "--global-top-k",
        type=int,
        default=48,
        help="Сколько чанков для финального отчёта (многоэтапный)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000, help="Размер чанка (символов)"
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Перекрытие между чанками"
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="Отсечь файлы больше этого размера (байт) на входе",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=200_000,
        help="Сколько символов читать из начала текстовых файлов",
    )
    parser.add_argument(
        "--scan-timeout-ms",
        type=int,
        default=1500,
        help="Таймаут (мс) на обработку одного файла (чтение+чанкинг)",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=200,
        help="Максимум чанков для одного файла",
    )
    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Игнорировать .gitignore (по умолчанию учитывается)",
    )
    parser.add_argument(
        "--skip-lockfiles",
        action="store_true",
        default=True,
        help="Пропускать lock-файлы (включено по умолчанию)",
    )
    parser.add_argument(
        "--no-skip-lockfiles", action="store_true", help="Отключить пропуск lock-файлов"
    )
    parser.add_argument(
        "--skip-minified",
        action="store_true",
        default=True,
        help="Пропускать .min.js/.min.css (включено по умолчанию)",
    )
    parser.add_argument(
        "--no-skip-minified",
        action="store_true",
        help="Отключить пропуск minified-ассетов",
    )
    parser.add_argument(
        "--skip-sourcemaps",
        action="store_true",
        default=True,
        help="Пропускать *.map (включено по умолчанию)",
    )
    parser.add_argument(
        "--no-skip-sourcemaps", action="store_true", help="Отключить пропуск sourcemaps"
    )
    parser.add_argument(
        "--verbose-files", action="store_true", help="Печатать обрабатываемые файлы"
    )
    parser.add_argument("--report-md", type=str, default="reports/analysis_report.md")
    parser.add_argument(
        "--multi-stage",
        action="store_true",
        help="Включить многоэтапный анализ (по модулям + финальный отчёт)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Каталог для кэша эмбеддингов (по умолчанию: PROJECT_ROOT/.proj_analyzer_cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Не использовать кэш эмбеддингов"
    )

    args = parser.parse_args()

    skip_lockfiles = args.skip_lockfiles and not args.no_skip_lockfiles
    skip_minified = args.skip_minified and not args.no_skip_minified
    skip_sourcemaps = args.skip_sourcemaps and not args.no_skip_sourcemaps

    root = Path(args.project_root).resolve()
    if not root.exists():
        console.print(f"[red]Не найден путь:[/] {root}")
        raise SystemExit(1)

    if "OPENAI_API_KEY" not in os.environ:
        console.print(
            "[red]Требуется переменная окружения OPENAI_API_KEY[/]. Пример:\n  export OPENAI_API_KEY=sk-..."
        )
        raise SystemExit(1)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else (root / ".proj_analyzer_cache")
    )
    cache = None if args.no_cache else EmbeddingCache(cache_dir)

    client = OpenAI()

    console.print(Panel.fit(f"📁 Анализ проекта: [bold]{root}[/]"))
    if not args.no_gitignore:
        if (root / ".gitignore").exists():
            console.print("[green].gitignore обнаружен — игнорируем указанные пути.[/]")
        else:
            console.print(
                "[yellow].gitignore не найден — будут проанализированы все файлы (кроме служебных директорий).[/]"
            )

    indexer = ProjectIndexer(
        root=root,
        exclude_dirs=args.exclude_dirs,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_file_chars=args.max_file_chars,
        scan_timeout_ms=args.scan_timeout_ms,
        max_chunks_per_file=args.max_chunks_per_file,
        skip_lockfiles=skip_lockfiles,
        skip_minified=skip_minified,
        skip_sourcemaps=skip_sourcemaps,
        verbose=args.verbose_files,
        respect_gitignore=(not args.no_gitignore),
    )
    chunks = indexer.build_chunks(max_chars=args.max_chars, overlap=args.overlap)
    if not chunks:
        console.print("[red]Не найдено файлов для анализа.[/]")
        raise SystemExit(1)

    retriever = Retriever(client, args.embed_model, cache=cache)
    retriever.embed_chunks(chunks)

    analyzer = LLMAnalyzer(client, args.llm_model, max_ctx_chunks=args.top_k)

    if not args.multi_stage:
        question = "Проанализируй структуру проекта и найди очевидные баги и риски."
        top_chunks = retriever.query(chunks, question, top_k=args.top_k)
        md = analyzer.analyze_single(root, top_chunks)
        (root / args.report_md).write_text(md, encoding="utf-8")
        console.print(
            Panel.fit(f"✅ Готово. Отчёт сохранён: [bold]{args.report_md}[/]")
        )
        console.print(
            Markdown("Совет: открой отчёт в Markdown-вьювере или залей в репозиторий.")
        )
        return

    modules = indexer.detect_modules(chunks)
    module_summaries = []
    for mod_name, mod_chunks in modules.items():
        question_mod = f"Проанализируй модуль '{mod_name}' и найди архитектурные аспекты, баги и риски."
        top_mod = retriever.query(mod_chunks, question_mod, top_k=args.module_top_k)
        md_mod = analyzer.analyze_module(root, mod_name, top_mod)
        (reports_dir / f"module_{mod_name}.md").write_text(md_mod, encoding="utf-8")
        module_summaries.append(f"## {mod_name}" + md_mod[:1200])

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
