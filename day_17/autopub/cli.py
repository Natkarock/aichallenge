from __future__ import annotations

import argparse, os
from pathlib import Path
from slugify import slugify
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from openai import OpenAI
from .utils import clean_html, remove_stopwords
from .git_utils import clone_or_update, new_branch, commit_all, push_branch
from .gdoc import fetch_gdoc_text
from .changer import collect_docs_context, apply_changes, discover_targets
from .llm import summarize_tor, propose_changes

console = Console()

def run_pipeline(args):
    client = OpenAI()

    # 1) clone/pull
    with Progress(SpinnerColumn(), TextColumn("[bold]Git[/] • {task.fields[step]}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
        t = p.add_task("git", total=None, step="подготовка")
        repo_root = clone_or_update(args.repo, Path(args.workdir))
        p.update(t, step="ok")

    # 2) fetch ToR + preprocess
    console.print(Panel.fit("📄 ТЗ: скачиваю и прогоняю через мини‑пайплайн"))
    tor_raw = fetch_gdoc_text(args.gdoc)
    tor_clean = clean_html(tor_raw)
    tor_tokens = remove_stopwords(tor_clean)
    tor_summary = summarize_tor(client, args.llm_model, tor_tokens)

    # 2.5) gather docs context
    docs_ctx = collect_docs_context(repo_root)

    # 3) new branch
    first_line = (tor_summary.splitlines() or ["autopub"])[0]
    slug = slugify(first_line[:40] or "autopub")
    branch = new_branch(repo_root, args.branch_prefix, slug)

    # 4) propose changes within targets
    targets = discover_targets(repo_root, args.targets)
    changeset = propose_changes(client, args.llm_model, tor_summary, docs_ctx, targets)

    # apply
    apply_changes(repo_root, changeset["changes"])
    (repo_root / "release_notes.md").write_text(changeset["release_notes"], encoding="utf-8")

    # 5) commit + push
    commit_all(repo_root, f"autopub: apply changes from ToR; branch {branch}")
    push_branch(repo_root, branch)

    console.print(Panel.fit(f"✅ Готово! Ветка: [bold]{branch}[/]"))
    console.print("Файл release_notes.md создан в корне репозитория.")

def main():
    ap = argparse.ArgumentParser(description="Авто‑пайплайн: git → ToR → summary → изменения → push")
    ap.add_argument("--repo", required=True, help="URL репозитория (https/ssh)")
    ap.add_argument("--gdoc", required=True, help="Ссылка на Google Docs (доступ по ссылке) или локальный .txt/.md")
    ap.add_argument("--workdir", required=True, help="Директория для клона")
    ap.add_argument("--branch-prefix", default="auto", help="Префикс ветки")
    ap.add_argument("--targets", default="README.md,CHANGELOG.md,docs/*.md", help="Файлы, которые разрешено редактировать")
    ap.add_argument("--llm-model", default="gpt-4.1", help="Модель OpenAI для суммаризации и генерации изменений")
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
