from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore

# ====== DIRECTORIES ======
BASE_DIR = os.path.dirname(__file__)
MEM_DIR = os.path.join(BASE_DIR, "memory")
RAG_DIR = os.path.join(MEM_DIR, "rag")

os.makedirs(RAG_DIR, exist_ok=True)

INDEX_PATH = os.path.join(RAG_DIR, "faiss_index")
CACHE_PATH = os.path.join(RAG_DIR, "emb_cache")
FILES_META = os.path.join(RAG_DIR, "files.json")


def _ensure_registry() -> None:
    """Ensure files.json exists even if empty."""
    if not os.path.exists(FILES_META):
        os.makedirs(RAG_DIR, exist_ok=True)
        with open(FILES_META, "w", encoding="utf-8") as f:
            json.dump({"files": []}, f, ensure_ascii=False, indent=2)


# Create empty registry at import time
_ensure_registry()


# ====== META I/O ======
def _load_meta() -> Dict[str, Any]:
    try:
        if not os.path.exists(FILES_META):
            _ensure_registry()
        with open(FILES_META, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "files" not in data:
            return {"files": []}
        return data
    except Exception:
        return {"files": []}


def _save_meta(data: Dict[str, Any]) -> None:
    os.makedirs(RAG_DIR, exist_ok=True)
    tmp = FILES_META + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FILES_META)


# ====== EMBEDDINGS ======
def _embeddings():
    os.makedirs(CACHE_PATH, exist_ok=True)
    _ = LocalFileStore(CACHE_PATH)
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


# ====== VECTORSTORE I/O ======
def _load_vectorstore() -> FAISS | None:
    if os.path.isdir(INDEX_PATH):
        try:
            return FAISS.load_local(
                INDEX_PATH, _embeddings(), allow_dangerous_deserialization=True
            )
        except Exception:
            return None
    return None


def _save_vectorstore(vs: FAISS) -> None:
    vs.save_local(INDEX_PATH)


# ====== CHUNKING ======
def _split_text(text: str, source: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(text or "")


# ====== PUBLIC API ======
def add_texts(pairs: List[Tuple[str, str]]) -> int:
    docs: List[Document] = []
    for text, src in pairs:
        for c in _split_text(text, src or "uploaded"):
            docs.append(
                Document(page_content=c, metadata={"source": src or "uploaded"})
            )

    vs = _load_vectorstore()
    if vs is None:
        vs = FAISS.from_documents(docs, _embeddings())
    else:
        vs.add_documents(docs)
    _save_vectorstore(vs)
    return len(docs)


def add_files(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    meta = _load_meta()
    records: List[Dict[str, Any]] = []

    vs = _load_vectorstore()
    if vs is None:
        first_docs: List[Document] = []
        first_ids: List[str] = []
        for text, name in pairs:
            file_id = str(uuid.uuid4())
            chunks = _split_text(text, name or "uploaded")
            uuids = [f"{file_id}:{i}" for i in range(len(chunks))]
            docs = [
                Document(page_content=c, metadata={"source": name, "file_id": file_id})
                for c in chunks
            ]
            first_docs.extend(docs)
            first_ids.extend(uuids)
            rec = {
                "file_id": file_id,
                "name": name,
                "chunk_ids": uuids,
                "num_chunks": len(uuids),
                "added_at": datetime.utcnow().isoformat() + "Z",
            }
            meta["files"].append(rec)
            records.append(
                {k: rec[k] for k in ("file_id", "name", "num_chunks", "added_at")}
            )
        if first_docs:
            vs = FAISS.from_documents(first_docs, _embeddings(), ids=first_ids)
            _save_vectorstore(vs)
        _save_meta(meta)
        return records

    for text, name in pairs:
        file_id = str(uuid.uuid4())
        chunks = _split_text(text, name or "uploaded")
        uuids = [f"{file_id}:{i}" for i in range(len(chunks))]
        docs = [
            Document(page_content=c, metadata={"source": name, "file_id": file_id})
            for c in chunks
        ]
        if docs:
            try:
                vs.add_documents(docs, ids=uuids)
            except TypeError:
                vs.add_documents(docs, uuids=uuids)
        rec = {
            "file_id": file_id,
            "name": name,
            "chunk_ids": uuids,
            "num_chunks": len(uuids),
            "added_at": datetime.utcnow().isoformat() + "Z",
        }
        meta["files"].append(rec)
        records.append(
            {k: rec[k] for k in ("file_id", "name", "num_chunks", "added_at")}
        )

    _save_vectorstore(vs)
    _save_meta(meta)
    return records


def list_files() -> List[Dict[str, Any]]:
    meta = _load_meta()
    return [
        {
            "file_id": f.get("file_id"),
            "name": f.get("name"),
            "num_chunks": int(f.get("num_chunks") or 0),
            "added_at": f.get("added_at"),
        }
        for f in meta.get("files", [])
    ]


def remove_file(file_id: str) -> bool:
    if not file_id:
        return False
    meta = _load_meta()
    entry = next(
        (f for f in meta.get("files", []) if f.get("file_id") == file_id), None
    )
    if not entry:
        return False

    vs = _load_vectorstore()
    if vs is not None:
        chunk_ids = entry.get("chunk_ids") or []
        if chunk_ids:
            try:
                vs.delete(ids=chunk_ids)
            except Exception:
                pass
            _save_vectorstore(vs)

    meta["files"] = [f for f in meta.get("files", []) if f.get("file_id") != file_id]
    _save_meta(meta)
    return True


def registry_info() -> dict:
    try:
        if not os.path.exists(FILES_META):
            _ensure_registry()
        with open(FILES_META, "r", encoding="utf-8") as f:
            data = json.load(f)
        files = data.get("files", []) if isinstance(data, dict) else []
        return {"path": FILES_META, "count": len(files)}
    except Exception:
        return {"path": FILES_META, "count": 0}


def similarity_search(query: str, k: int = 3):
    """
    Retrieve top-k documents relevant to the query.
    Returns List[Document]
    """
    vs = _load_vectorstore()
    if vs is None:
        return []
    return vs.similarity_search(query, k=k)
