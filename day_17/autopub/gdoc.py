from __future__ import annotations

import re, requests
from pathlib import Path
from rich.console import Console

console = Console()

def fetch_gdoc_text(link_or_path: str) -> str:
    p = Path(link_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", link_or_path)
    if not m:
        raise ValueError("Не распознан формат ссылки на Google Docs")
    doc_id = m.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    console.log(f"[dim]GET {export_url}[/]")
    r = requests.get(export_url, timeout=30)
    r.raise_for_status()
    return r.text
