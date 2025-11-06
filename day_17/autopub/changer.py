from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from rich.console import Console
from .utils import read_text_safe, glob_many

console = Console()

def collect_docs_context(repo_root: Path, limit_chars: int = 12000) -> str:
    candidates = [
        "README.md","CHANGELOG.md","RELEASE_NOTES.md","app/build.gradle","app/build.gradle.kts",
        "build.gradle","build.gradle.kts","pubspec.yaml","package.json"
    ]
    cuts = []
    for c in candidates:
        p = repo_root / c
        if p.exists():
            cuts.append(f"\n--- {c} ---\n{read_text_safe(p, limit_chars=3000)}")
    docs_dir = repo_root / "docs"
    if docs_dir.exists():
        for p in docs_dir.rglob("*.md"):
            cuts.append(f"\n--- docs/{p.name} ---\n{read_text_safe(p, limit_chars=1500)}")
    txt = "".join(cuts)
    return txt[:limit_chars]

def apply_changes(repo_root: Path, changes: List[Dict]) -> None:
    for ch in changes:
        path = repo_root / ch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ch["content"], encoding="utf-8")
        console.print(f"[green]→ перезаписан[/] {path}")

def discover_targets(repo_root: Path, pattern_csv: str) -> List[str]:
    pats = [p.strip() for p in pattern_csv.split(",") if p.strip()]
    paths = glob_many(repo_root, pats)
    return [str(p.relative_to(repo_root)) for p in paths]
