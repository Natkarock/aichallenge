from __future__ import annotations

import argparse, os
from pathlib import Path
from slugify import slugify
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from openai import OpenAI
from .utils import clean_html, remove_stopwords
from .git_utils import clone_or_update, new_branch, commit_all, push_branch
from .gdoc import fetch_gdoc_text
from .project_scan import snapshot_repo
from .llm import summarize_tor, propose_changes_for_project

console = Console()


def parse_globs(csv: str) -> list[str]:
    return [g.strip() for g in (csv or "").split(",") if g.strip()]


def run_pipeline(args):
    client = OpenAI()

    # 1) clone/pull
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Git[/] • {task.fields[step]}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("git", total=None, step="подготовка")
        repo_root = clone_or_update(args.repo, Path(args.workdir))
        p.update(t, step="ok")

    # 2) ToR preprocess (fetch -> clean -> stopwords -> summarize) — с прогрессом
    console.print(Panel.fit("📄 ТЗ: загружаю и готовлю"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]ToR[/] • {task.fields[step]}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("tor", total=4, step="fetch")
        # fetch
        tor_raw = fetch_gdoc_text(args.gdoc)
        p.update(t, advance=1, step="очистка")
        # clean
        tor_clean = clean_html(tor_raw)
        p.update(t, advance=1, step="стоп-слова")
        # stopwords
        tor_tokens = remove_stopwords(tor_clean)
        p.update(t, advance=1, step="summary")
        # summarize
        tor_summary = summarize_tor(client, args.llm_model, tor_tokens)
        p.update(t, advance=1, step="готово")

    # 3) project snapshot — с прогрессом по файлам
    console.print(Panel.fit("🗂️ Формирую snapshot проекта"))
    include_globs = parse_globs(args.include)
    exclude_globs = parse_globs(args.exclude)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Snapshot[/] • {task.fields[curfile]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("snap", total=0, curfile="-")
        project_snapshot = snapshot_repo(
            Path(repo_root),
            include_globs,
            exclude_globs,
            args.max_file_chars,
            args.max_files,
            progress=p,
            task_id=t,
        )

    # 4) new branch
    first_line = (tor_summary.splitlines() or ["autopub"])[0]
    slug = slugify(first_line[:40] or "autopub")
    branch = new_branch(repo_root, args.branch_prefix, slug)

    # 5) propose changes (whole project) — показываем таймер LLM-вызова
    console.print(Panel.fit("🧠 Генерирую изменения по проекту"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]LLM[/] • генерация изменений"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("llm", total=None)
        changeset = propose_changes_for_project(
            client, args.llm_model, tor_summary, project_snapshot
        )
        p.update(t, total=1)
        p.advance(t)

    # 6) apply changes — короткий вывод без бара
    applied = 0
    for ch in changeset.get("changes", []):
        path = repo_root / ch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ch["content"], encoding="utf-8")
        console.print(f"[green]→ перезаписан[/] {path}")
        applied += 1
    (repo_root / "change_notes.md").write_text(
        changeset.get("change_notes", ""), encoding="utf-8"
    )

    # 7) commit + push
    commit_all(repo_root, f"autopub: {applied} file(s) updated; branch {branch}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Git[/] • push"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("push", total=None)
        push_branch(repo_root, branch)
        p.update(t, total=1)
        p.advance(t)

    console.print(
        Panel.fit(f"✅ Готово! Изменено файлов: {applied}. Ветка: [bold]{branch}[/]")
    )
    console.print("Файл release_notes.md создан в корне репозитория.")


def main():
    ap = argparse.ArgumentParser(
        description="Авто-пайплайн v2: git → ToR → snapshot → изменения по всему проекту → release_notes → push"
    )
    ap.add_argument("--repo", required=True, help="URL репозитория (https/ssh)")
    ap.add_argument(
        "--gdoc",
        required=True,
        help="Ссылка на Google Docs (доступ по ссылке) или локальный .txt/.md",
    )
    ap.add_argument("--workdir", required=True, help="Директория для клона")
    ap.add_argument("--branch-prefix", default="auto", help="Префикс ветки")
    ap.add_argument(
        "--include", default="**/*", help="Что индексировать (glob, через запятую)"
    )
    ap.add_argument(
        "--exclude",
        default="**/*.png,**/*.jpg,**/*.jpeg,**/*.gif,**/*.webp,**/*.pdf,**/*.apk,**/*.aab,**/*.ipa,**/*.so,**/*.dylib,**/*.dll,**/*.jar,**/*.keystore,**/*.lock,**/*.zip,**/*.tar,**/*.gz",
        help="Что исключать (glob, через запятую)",
    )
    ap.add_argument(
        "--max-file-chars",
        type=int,
        default=6000,
        help="Сколько символов читать из начала каждого файла",
    )
    ap.add_argument(
        "--max-files", type=int, default=500, help="Максимум файлов в snapshot"
    )
    ap.add_argument(
        "--llm-model",
        default="gpt-4.1",
        help="Модель OpenAI для суммаризации и генерации изменений",
    )
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
