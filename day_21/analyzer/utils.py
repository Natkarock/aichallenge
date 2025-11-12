from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

TEXT_EXT_HINTS = {
    ".kt", ".kts", ".java", ".groovy", ".gradle", ".gradle.kts",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash", ".zsh",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".rs", ".go", ".rb", ".php",
    ".swift", ".cs", ".m", ".mm",
    ".xml", ".html", ".xhtml", ".svg", ".css", ".scss", ".md", ".rst",
    ".json", ".yml", ".yaml", ".toml", ".ini", ".properties", ".cfg",
    ".pro", ".txt", ".csv", ".tsv", ".ipynb",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".svn", ".hg",
    ".idea", ".vscode",
    "build", ".gradle", ".gradle-kotlin-dsl",
    "node_modules", "dist", "out", "target",
    "__pycache__", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".env",
}

LOCKFILE_BASENAMES = {
    "yarn.lock","package-lock.json","pnpm-lock.yaml","bun.lockb",
    "poetry.lock","Cargo.lock","Pipfile.lock","Gemfile.lock",
    "composer.lock","go.sum","gradle-lockfile","Podfile.lock",
    "project.pbxproj","gradlew",
}

def is_probably_text(path: Path, sample_bytes: int = 8192) -> bool:
    ext = path.suffix.lower()
    if ext in TEXT_EXT_HINTS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_bytes)
    except Exception:
        return False
    if b"\x00" in chunk:
        return False
    printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in chunk)
    ratio = printable / (len(chunk) or 1)
    return ratio > 0.8

def fast_head_text(p: Path, limit_chars: int, time_budget_s: float | None = None) -> str:
    buf = []
    total = 0
    chunk_size = 8192
    start = time.perf_counter()
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            while total < limit_chars:
                if time_budget_s is not None and (time.perf_counter() - start) > time_budget_s:
                    raise TimeoutError("read timeout")
                need = min(chunk_size, limit_chars - total)
                s = f.read(need)
                if not s:
                    break
                buf.append(s)
                total += len(s)
    except TimeoutError:
        raise
    except Exception:
        return ""
    return "".join(buf)

def safe_chunk_text(text: str, max_chars: int = 4000, overlap: int = 200, max_chunks: int = 200) -> List[str]:
    if not text:
        return []
    if max_chars <= 0:
        max_chars = 4000
    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)
    L = len(text)
    if L <= max_chars:
        return [text]
    chunks = []
    i = 0
    limit = 0
    while i < L and limit < max_chunks:
        end = min(L, i + max_chars)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
        limit += 1
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

@dataclass
class DocChunk:
    path: str
    idx: int
    text: str
    vector: np.ndarray | None = None

def binary_meta_chunk(path: Path) -> str:
    sz = path.stat().st_size
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return (
        f"[BINARY FILE]\n"
        f"path: {path}\n"
        f"size_bytes: {sz}\n"
        f"mime: {mime}\n"
        f"note: binary content omitted from index"
    )
