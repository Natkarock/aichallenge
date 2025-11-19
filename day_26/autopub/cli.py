from __future__ import annotations

import argparse
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

from .git_utils import clone_or_update, new_branch, commit_all, push_branch
from .project_scan import snapshot_repo
from .llm import (
    select_files_for_tests,
    select_related_files_for_tests,
    generate_tests_for_file,
)

console = Console()


def parse_globs(csv: str) -> list[str]:
    return [g.strip() for g in (csv or "").split(",") if g.strip()]


def _read_text_safe(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _collect_related_files(
    client: OpenAI,
    model: str,
    repo_root: Path,
    target_rel_path: str,
    target_code_for_related: str,
    project_snapshot_short: str,
    max_file_chars: int,
) -> dict[str, str]:
    """
    Определяет с помощью LLM, какие файлы из snapshot важны как контекст
    для написания тестов к одному конкретному файлу, и возвращает
    {relative_path: truncated_content}.

    ВАЖНО:
    - сюда мы передаём уже УКОРOЧЕННЫЙ snapshot (project_snapshot_short),
      а не весь snapshot целиком;
    - target_code_for_related тоже ограничен по длине.
    """
    related_paths = select_related_files_for_tests(
        client=client,
        model=model,
        project_snapshot=project_snapshot_short,
        target_path=target_rel_path,
        target_code=target_code_for_related,
    )

    related: dict[str, str] = {}
    for rel in related_paths:
        path = repo_root / rel
        if not path.exists() or not path.is_file():
            continue
        related[rel] = _read_text_safe(path, max_file_chars)
    return related


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

    # 2) project snapshot — с прогрессом по файлам
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

    # 🔻 укороченный snapshot специально для поиска связанных файлов
    project_snapshot_for_related = project_snapshot[: args.related_snapshot_chars]

    # 3) new branch под unit-тесты
    slug_source = Path(repo_root).name or "unit-tests"
    slug = slugify(slug_source[:40] or "unit-tests")
    branch = new_branch(Path(repo_root), args.branch_prefix, slug)

    # 4) выбор файлов для написания unit-тестов (LLM)
    console.print(Panel.fit("🧪 Определяю файлы для unit-тестов"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Tests[/] • {task.fields[step]}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("select", total=3, step="подготовка")
        p.update(t, advance=1, step="LLM-анализ")
        target_files = select_files_for_tests(
            client=client,
            model=args.llm_model,
            project_snapshot=project_snapshot,
        )
        p.update(t, advance=1, step="готово")

    if not target_files:
        console.print(
            "[yellow]LLM не выбрал ни одного файла для тестов — изменений нет.[/]"
        )
        return

    # 5) генерация unit-тестов по каждому файлу
    console.print(
        Panel.fit(
            f"🧪 Генерирую unit-тесты для {len(target_files)} файлов "
            "(каждый запрос — отдельный ChangeSet)"
        )
    )

    combined_changes: dict[str, object] = {"change_notes": "", "changes": []}
    notes_parts: list[str] = []

    from typing import cast

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]LLM[/] • unit-тесты"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        task = p.add_task("tests", total=len(target_files))

        for rel_path in target_files:
            rel_path = rel_path.strip()
            if not rel_path:
                p.update(task, advance=1)
                continue

            src_path = Path(repo_root) / rel_path
            if not src_path.exists():
                console.print(f"[yellow]Пропускаю: не найден файл {rel_path}[/]")
                p.update(task, advance=1)
                continue

            console.print(f"[cyan]▶ Генерирую тесты для {rel_path}[/]")

            # Полный код файла — для генерации тестов
            target_code_full = _read_text_safe(src_path, args.max_file_chars)
            # Укороченный код — только для поиска связанных файлов
            target_code_for_related = target_code_full[: args.related_file_chars]

            related_files = _collect_related_files(
                client=client,
                model=args.llm_model,
                repo_root=Path(repo_root),
                target_rel_path=rel_path,
                target_code_for_related=target_code_for_related,
                project_snapshot_short=project_snapshot_for_related,
                max_file_chars=args.max_file_chars,
            )

            changeset = generate_tests_for_file(
                client=client,
                model=args.llm_model,
                target_path=rel_path,
                target_code=target_code_full,
                related_files=related_files,
            )

            if not changeset:
                p.update(task, advance=1)
                continue

            # накапливаем изменения и заметки
            combined_changes["changes"] = cast(list, combined_changes["changes"])
            combined_changes["changes"].extend(changeset.get("changes", []))  # type: ignore[index]
            note = changeset.get("change_notes", "") or ""
            if note:
                notes_parts.append(f"### Tests for {rel_path}\n{note}")

            p.update(task, advance=1)

    combined_changes["change_notes"] = "\n\n".join(notes_parts)

    # 6) apply changes — короткий вывод без бара
    applied = 0
    for ch in combined_changes.get("changes", []):  # type: ignore[union-attr]
        path = Path(repo_root) / ch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ch["content"], encoding="utf-8")
        console.print(f"[green]→ перезаписан[/] {path}")
        applied += 1

    (Path(repo_root) / "change_notes.md").write_text(
        combined_changes.get("change_notes", "") or "", encoding="utf-8"  # type: ignore[arg-type]
    )

    # 7) commit + push
    commit_all(
        Path(repo_root), f"autotests: {applied} file(s) updated; branch {branch}"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Git[/] • push"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as p:
        t = p.add_task("push", total=None)
        push_branch(Path(repo_root), branch)
        p.update(t, total=1)
        p.advance(t)

    console.print(
        Panel.fit(f"✅ Готово! Изменено файлов: {applied}. Ветка: [bold]{branch}[/]")
    )
    console.print("Файл change_notes.md создан в корне репозитория.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Авто-пайплайн: git → snapshot → выбор файлов → генерация unit-тестов "
            "→ change_notes → push (язык проекта определяется по файлам)"
        )
    )
    ap.add_argument("--repo", required=True, help="URL репозитория (https/ssh)")
    ap.add_argument("--workdir", required=True, help="Директория для клона")
    ap.add_argument("--branch-prefix", default="auto-tests", help="Префикс ветки")
    ap.add_argument(
        "--include", default="**/*", help="Что индексировать (glob, через запятую)"
    )
    ap.add_argument(
        "--exclude",
        default=(
            "**/*.png,**/*.jpg,**/*.jpeg,**/*.gif,**/*.webp,"
            "**/*.dll,**/*.jar,**/*.keystore,**/*.lock,"
            "**/*.zip,**/*.tar,**/*.gz"
        ),
        help="Что исключать (glob, через запятую)",
    )
    ap.add_argument(
        "--max-file-chars",
        type=int,
        default=6000,
        help="Сколько символов читать из начала каждого файла (для генерации тестов)",
    )
    ap.add_argument(
        "--max-files", type=int, default=500, help="Максимум файлов в snapshot"
    )
    ap.add_argument(
        "--related-snapshot-chars",
        type=int,
        default=8000,
        help="Сколько символов snapshot'а передавать в LLM при поиске связанных файлов",
    )
    ap.add_argument(
        "--related-file-chars",
        type=int,
        default=2000,
        help="Сколько символов кода файла передавать в LLM при поиске связанных файлов",
    )
    ap.add_argument(
        "--llm-model",
        default="gpt-4.1",
        help="Модель OpenAI для выбора файлов и генерации unit-тестов",
    )
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
