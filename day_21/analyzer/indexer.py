from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from pathspec import PathSpec
from .utils import (
    is_probably_text,
    fast_head_text,
    safe_chunk_text,
    DocChunk,
    binary_meta_chunk,
    DEFAULT_EXCLUDE_DIRS,
    LOCKFILE_BASENAMES,
)

console = Console()


def load_gitignore_spec(root: Path) -> PathSpec | None:
    gi = root / ".gitignore"
    if gi.exists():
        try:
            lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
            return PathSpec.from_lines("gitwildmatch", lines)
        except Exception:
            return None
    return None


def is_ignored_by_gitignore(root: Path, p: Path, spec: PathSpec | None) -> bool:
    if spec is None:
        return False
    rel = p.relative_to(root).as_posix()
    return spec.match_file(rel)


class ProjectIndexer:
    def __init__(
        self,
        root: Path,
        exclude_dirs: List[str],
        max_files: int | None,
        max_file_bytes: int,
        max_file_chars: int,
        scan_timeout_ms: int,
        max_chunks_per_file: int,
        skip_lockfiles: bool,
        skip_minified: bool,
        skip_sourcemaps: bool,
        verbose: bool,
        respect_gitignore: bool = True,
    ):
        self.root = root
        self.exclude_dirs = set(exclude_dirs) | DEFAULT_EXCLUDE_DIRS
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_file_chars = max_file_chars
        self.scan_timeout_ms = scan_timeout_ms
        self.max_chunks_per_file = max_chunks_per_file
        self.skip_lockfiles = skip_lockfiles
        self.skip_minified = skip_minified
        self.skip_sourcemaps = skip_sourcemaps
        self.verbose = verbose
        self.skipped: list[tuple[str, str]] = []
        self.gitignore_spec = load_gitignore_spec(root) if respect_gitignore else None
        self.respect_gitignore = respect_gitignore

    def _skip_by_name(self, p: Path) -> str | None:
        name = p.name
        ext = p.suffix.lower()
        if self.skip_lockfiles and name in LOCKFILE_BASENAMES:
            return "lockfile"
        if self.skip_sourcemaps and ext == ".map":
            return "sourcemap"
        if self.skip_minified and (
            name.endswith(".min.js") or name.endswith(".min.css")
        ):
            return "minified"
        return None

    def build_chunks(self, max_chars=4000, overlap=200) -> list[DocChunk]:
        # Gitignore notice
        if self.respect_gitignore and self.gitignore_spec is not None:
            console.print("[dim]Учитываю .gitignore[/] — правила будут применены")
        elif self.respect_gitignore:
            console.print("[dim].gitignore не найден — идём без него[/]")
        else:
            console.print("[dim]Игнорирование .gitignore отключено параметром[/]")

        # Quick tree walk with spinner (enumerate files)
        all_files: list[Path] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Обхожу дерево файлов[/]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("walk", total=None)
            for p in self.root.rglob("*"):
                if not p.is_file():
                    continue
                all_files.append(p)
            # end spinner

        all_files: list[Path] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in self.exclude_dirs for part in p.parts):
                continue
            if is_ignored_by_gitignore(self.root, p, self.gitignore_spec):
                self.skipped.append((str(p), ".gitignore"))
                continue
            reason = self._skip_by_name(p)
            if reason:
                self.skipped.append((str(p), reason))
                continue
            try:
                sz = p.stat().st_size
            except Exception:
                self.skipped.append((str(p), "stat failed"))
                continue
            if self.max_file_bytes and sz > self.max_file_bytes:
                self.skipped.append((str(p), f"too big: {sz} bytes"))
                continue
            all_files.append(p)

        if self.max_files:
            all_files = all_files[: self.max_files]
        if overlap >= max_chars:
            overlap = max(0, max_chars // 4)

        chunks: list[DocChunk] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Сканирую файлы[/] • {task.fields[curfile]}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("scan", total=len(all_files), curfile="-")
            for fp in all_files:
                progress.update(task, curfile=fp.name)
                if self.verbose:
                    console.print(f"[dim]→ {fp}[/]")
                per_file_budget = max(0.05, self.scan_timeout_ms / 1000.0)
                try:
                    if is_probably_text(fp):
                        raw = fast_head_text(
                            fp,
                            limit_chars=self.max_file_chars,
                            time_budget_s=per_file_budget,
                        )
                        file_chunks = safe_chunk_text(
                            raw,
                            max_chars=max_chars,
                            overlap=overlap,
                            max_chunks=self.max_chunks_per_file,
                        )
                        if not file_chunks:
                            file_chunks = [""]
                        for i, ch in enumerate(file_chunks):
                            chunks.append(DocChunk(path=str(fp), idx=i, text=ch))
                    else:
                        chunks.append(
                            DocChunk(path=str(fp), idx=0, text=binary_meta_chunk(fp))
                        )
                except TimeoutError:
                    self.skipped.append((str(fp), "scan timeout"))
                except KeyboardInterrupt:
                    self.skipped.append((str(fp), "keyboard interrupt while scan"))
                except Exception as e:
                    self.skipped.append((str(fp), f"scan error: {type(e).__name__}"))
                progress.update(task, advance=1)

        if self.skipped:
            console.print("\n[yellow]Пропущенные/ограниченные файлы:[/]")
            for pth, reason in self.skipped[:80]:
                console.print(f"  • {pth} — {reason}")
            if len(self.skipped) > 80:
                console.print(f"  ... и ещё {len(self.skipped)-80}")
        return chunks

    def detect_modules(self, chunks: list[DocChunk]) -> dict[str, list[DocChunk]]:
        modules: dict[str, list[DocChunk]] = {}
        for ch in chunks:
            path = Path(ch.path)
            rel = path.relative_to(self.root)
            parts = rel.parts
            if len(parts) == 1:
                key = "root_files"
            else:
                key = parts[0]
            modules.setdefault(key, []).append(ch)
        return modules
