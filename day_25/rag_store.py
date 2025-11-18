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
FILES_DIR = os.path.join(RAG_DIR, "files")  # здесь будут лежать копии исходных текстов

os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

FILES_META = os.path.join(RAG_DIR, "meta.json")
VECTOR_INDEX_DIR = os.path.join(RAG_DIR, "faiss_index")
VECTOR_STORE = LocalFileStore(VECTOR_INDEX_DIR)


def _load_meta() -> Dict[str, Any]:
    if not os.path.exists(FILES_META):
        return {"files": []}
    try:
        with open(FILES_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": []}


def _save_meta(meta: Dict[str, Any]):
    with open(FILES_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _split_text(text: str, name: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _load_vectorstore():
    if not os.path.exists(VECTOR_INDEX_DIR):
        return None
    try:
        return FAISS.load_local(
            VECTOR_INDEX_DIR,
            _embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def _save_vectorstore(vs: FAISS):
    os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
    vs.save_local(VECTOR_INDEX_DIR)


def _make_file_copy(text: str, file_id: str, original_name: str | None) -> str:
    """
    Создаёт служебную копию текста на диске и возвращает ПОЛНЫЙ путь до файла.
    """
    base_name = (original_name or "uploaded").strip() or "uploaded"
    base_name = os.path.basename(base_name)
    # На всякий случай убираем разделители путей
    base_name = base_name.replace(os.sep, "_")

    filename = f"{file_id}_{base_name}"
    file_path = os.path.join(FILES_DIR, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        # fallback — хотя бы что-то сохранить
        fallback = f"{file_id}_uploaded.txt"
        file_path = os.path.join(FILES_DIR, fallback)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    return file_path


def reindex_all(files: List[Tuple[str, str]]) -> int:
    """
    Полная переиндексация (редкий кейс).
    files — список (text, source_name)
    """
    docs: List[Document] = []
    meta: Dict[str, Any] = {"files": []}

    for text, src in files:
        file_id = str(uuid.uuid4())
        file_path = _make_file_copy(text, file_id, src)

        chunks = _split_text(text, src or "uploaded")
        uuids = [f"{file_id}:{i}" for i in range(len(chunks))]
        for i, c in enumerate(chunks):
            docs.append(
                Document(
                    page_content=c,
                    metadata={
                        "source": src,
                        "file_id": file_id,
                        "file_name": src or "uploaded",
                        "file_path": file_path,  # <- УЖЕ реальный путь до файла-копии
                    },
                )
            )

        rec = {
            "file_id": file_id,
            "name": src,
            "file_path": file_path,
            "chunk_ids": uuids,
            "num_chunks": len(chunks),
            "added_at": datetime.utcnow().isoformat() + "Z",
        }
        meta["files"].append(rec)

    if docs:
        vs = FAISS.from_documents(docs, _embeddings())
        _save_vectorstore(vs)
    _save_meta(meta)
    return len(docs)


def add_files(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Индексация новых файлов.
    pairs: список кортежей (text, original_name)
    """
    meta = _load_meta()
    records: List[Dict[str, Any]] = []

    vs = _load_vectorstore()

    # Если индекс ещё не создан, инициализируем его пачкой
    if vs is None:
        first_docs: List[Document] = []
        first_ids: List[str] = []

        for text, name in pairs:
            file_id = str(uuid.uuid4())
            file_path = _make_file_copy(text, file_id, name)

            chunks = _split_text(text, name or "uploaded")
            uuids = [f"{file_id}:{i}" for i in range(len(chunks))]
            docs = [
                Document(
                    page_content=c,
                    metadata={
                        "source": name,
                        "file_id": file_id,
                        "file_name": name or "uploaded",
                        "file_path": file_path,  # <- реальный путь до служебного файла
                    },
                )
                for c in chunks
            ]
            first_docs.extend(docs)
            first_ids.extend(uuids)

            rec = {
                "file_id": file_id,
                "name": name,
                "file_path": file_path,
                "chunk_ids": uuids,
                "num_chunks": len(chunks),
                "added_at": datetime.utcnow().isoformat() + "Z",
            }
            meta.setdefault("files", []).append(rec)
            records.append(
                {k: rec[k] for k in ("file_id", "name", "num_chunks", "added_at")}
            )

        if first_docs:
            vs = FAISS.from_documents(first_docs, _embeddings(), ids=first_ids)
            _save_vectorstore(vs)
        _save_meta(meta)
        return records

    # Индекс уже есть — просто добавляем документы
    for text, name in pairs:
        file_id = str(uuid.uuid4())
        file_path = _make_file_copy(text, file_id, name)

        chunks = _split_text(text, name or "uploaded")
        uuids = [f"{file_id}:{i}" for i in range(len(chunks))]
        docs = [
            Document(
                page_content=c,
                metadata={
                    "source": name,
                    "file_id": file_id,
                    "file_name": name or "uploaded",
                    "file_path": file_path,  # <- реальный путь
                },
            )
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
            "file_path": file_path,
            "chunk_ids": uuids,
            "num_chunks": len(chunks),
            "added_at": datetime.utcnow().isoformat() + "Z",
        }
        meta.setdefault("files", []).append(rec)
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
            "file_path": f.get("file_path"),
            "num_chunks": int(f.get("num_chunks") or 0),
            "added_at": f.get("added_at"),
        }
        for f in meta.get("files", [])
    ]


def remove_file(file_id: str) -> bool:
    meta = _load_meta()
    files = meta.get("files", [])
    to_keep = [f for f in files if f.get("file_id") != file_id]
    removed_any = len(to_keep) != len(files)
    if not removed_any:
        return False

    # Можно удалить и физический файл (по желанию)
    for f in files:
        if f.get("file_id") == file_id:
            fp = f.get("file_path")
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
            break

    meta["files"] = to_keep
    _save_meta(meta)

    # На этом этапе мы не переиндексируем FAISS (это дорого),
    # просто оставляем «лишние» векторные точки — для учебного проекта ок.
    return True


def debug_meta_info() -> Dict[str, Any]:
    try:
        meta = _load_meta()
        files = meta.get("files", [])
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
