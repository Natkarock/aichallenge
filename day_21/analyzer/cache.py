from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Iterable

def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / "embeddings.jsonl"
        self._index: Dict[str, Dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
                try:
                    rec = json.loads(line)
                    self._index[rec["key"]] = rec
                except Exception:
                    continue

    def _key(self, path: str, idx: int) -> str:
        return f"{path}:::{idx}"

    def get(self, path: str, idx: int, text: str) -> Optional[list]:
        k = self._key(path, idx)
        rec = self._index.get(k)
        if not rec:
            return None
        if rec.get("text_hash") != _sha(text):
            return None
        return rec.get("vector")

    def put_many(self, items: Iterable[tuple[str,int,str,list]]):
        with self.path.open("a", encoding="utf-8") as f:
            for path, idx, text, vec in items:
                k = self._key(path, idx)
                rec = {"key": k, "text_hash": _sha(text), "vector": vec}
                self._index[k] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
