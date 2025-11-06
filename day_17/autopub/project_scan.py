from __future__ import annotations

from pathlib import Path
from typing import List
from .utils import is_probably_text

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "out",
    "target",
    "__pycache__",
    "venv",
    ".venv",
    ".idea",
    ".vscode",
}
DEFAULT_EXCLUDE_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".apk",
    ".aab",
    ".ipa",
    ".so",
    ".dylib",
    ".dll",
    ".jar",
    ".keystore",
    ".lock",
    ".zip",
    ".tar",
    ".gz",
}


def should_skip(path: Path, include_globs: List[str], exclude_globs: List[str]) -> bool:
    from fnmatch import fnmatch

    if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in DEFAULT_EXCLUDE_EXT:
        return True
    if include_globs:
        ok = any(fnmatch(str(path), pat) for pat in include_globs)
        if not ok:
            return True
    if exclude_globs and any(fnmatch(str(path), pat) for pat in exclude_globs):
        return True
    return False


def snapshot_repo(
    root: Path,
    include_globs: List[str],
    exclude_globs: List[str],
    max_file_chars: int,
    max_files: int,
    progress=None,
    task_id=None,
) -> str:
    """
    Собирает компактный snapshot: список текстовых файлов (+ head каждого).
    Если переданы progress и task_id (Rich Progress), показывает прогресс по файлам.
    """
    # 1) Соберём список кандидатов
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p, include_globs, exclude_globs):
            continue
        if not is_probably_text(p):
            continue
        files.append(p)
        if max_files and len(files) >= max_files:
            break

    # Обновим total — теперь известно кол-во файлов
    if progress is not None and task_id is not None:
        progress.update(task_id, total=len(files))

    # 2) Формируем текстовый snapshot с прогрессом
    out_lines: List[str] = []
    out_lines.append("### PROJECT SNAPSHOT\n")
    for idx, p in enumerate(files, start=1):
        rel = p.relative_to(root).as_posix()
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:max_file_chars]
        except Exception:
            txt = ""
        out_lines.append(f"\n--- {rel} ---\n{txt}")

        if progress is not None and task_id is not None:
            # Обновим подпись и подвинем прогресс
            progress.update(task_id, advance=1, **{"curfile": rel})

    return "".join(out_lines)
