
from __future__ import annotations

import os
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(__file__)
MEM_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEM_DIR, exist_ok=True)

JSON_PATH = os.path.join(MEM_DIR, "chats.json")
DB_PATH = os.path.join(MEM_DIR, "chats.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            summary TEXT,
            rag_enabled INTEGER DEFAULT 0
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
        """)
        cur.execute("PRAGMA foreign_keys = ON;")
        conn.commit()


def _db_has_data() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
            if cur.fetchone() is None:
                return False
            cur.execute("SELECT COUNT(*) FROM chats")
            return cur.fetchone()[0] > 0
    except Exception:
        return False


def _migrate_json_to_db() -> None:
    if not os.path.exists(JSON_PATH):
        return
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict) or "chats" not in data:
        return

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM chats")
        for chat in data.get("chats", []):
            cid = chat.get("id") or str(uuid.uuid4())
            title = chat.get("title") or "Чат"
            created_at = chat.get("created_at") or (datetime.utcnow().isoformat()+"Z")
            summary = chat.get("summary")
            rag_enabled = int(1 if chat.get("rag_enabled") else 0)
            cur.execute("INSERT OR REPLACE INTO chats(id, title, created_at, summary, rag_enabled) VALUES (?, ?, ?, ?, ?)",
                        (cid, title, created_at, summary, rag_enabled))
            for m in chat.get("messages", []):
                role = m.get("role") or "user"
                content = m.get("content") or ""
                ts = m.get("ts") or (datetime.utcnow().isoformat()+"Z")
                cur.execute("INSERT INTO messages(chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
                            (cid, role, content, ts))
        conn.commit()


def _ensure_ready() -> None:
    _init_db()
    # Ensure new columns exist (idempotent)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(chats)")
        cols = [r[1] for r in cur.fetchall()]
        if "summary" not in cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN summary TEXT")
                conn.commit()
            except Exception:
                pass
        if "rag_enabled" not in cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN rag_enabled INTEGER DEFAULT 0")
                conn.commit()
            except Exception:
                pass
    if not _db_has_data() and os.path.exists(JSON_PATH):
        _migrate_json_to_db()


def load_store() -> Dict[str, Any]:
    """Return store as dict: {'chats': [ {id,title,created_at,summary,rag_enabled,messages:[{role,content,ts},...]}, ... ]}"""
    _ensure_ready()
    out = {"chats": []}
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at, summary, rag_enabled FROM chats ORDER BY created_at DESC")
        for c in cur.fetchall():
            cid = c["id"]
            cur.execute("SELECT role, content, ts FROM messages WHERE chat_id=? ORDER BY id ASC", (cid,))
            msgs = [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in cur.fetchall()]
            out["chats"].append({
                "id": cid,
                "title": c["title"],
                "created_at": c["created_at"],
                "summary": c["summary"],
                "rag_enabled": bool(c["rag_enabled"]),
                "messages": msgs,
            })
    return out


def save_store(store: Dict[str, Any]) -> None:
    """Persist PROVIDED store fully into SQLite (full-sync)."""
    _ensure_ready()
    chats = store.get("chats", []) if isinstance(store, dict) else []
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM chats")
        for chat in chats:
            cid = chat.get("id") or str(uuid.uuid4())
            title = chat.get("title") or "Чат"
            created_at = chat.get("created_at") or (datetime.utcnow().isoformat()+"Z")
            summary = chat.get("summary")
            rag_enabled = int(1 if chat.get("rag_enabled") else 0)
            cur.execute("INSERT OR REPLACE INTO chats(id, title, created_at, summary, rag_enabled) VALUES (?, ?, ?, ?, ?)",
                        (cid, title, created_at, summary, rag_enabled))
            for m in chat.get("messages", []):
                role = (m.get("role") or "user").strip()
                content = m.get("content") or ""
                ts = m.get("ts") or (datetime.utcnow().isoformat()+"Z")
                cur.execute("INSERT INTO messages(chat_id, role, content, ts) VALUES (?, ?, ?, ?)", (cid, role, content, ts))
        conn.commit()


def new_chat(title: str = "Новый чат") -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "summary": None,
        "rag_enabled": False,
        "messages": [],
    }


def delete_chat(store: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
    if not store or "chats" not in store:
        return store
    store["chats"] = [c for c in store["chats"] if c.get("id") != chat_id]
    _ensure_ready()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    return store


def get_chat_summary(chat_id: str) -> Optional[str]:
    _ensure_ready()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT summary FROM chats WHERE id=?", (chat_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def update_chat_summary(chat_id: str, summary: str) -> None:
    _ensure_ready()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE chats SET summary=? WHERE id=?", (summary, chat_id))
        conn.commit()


def get_chat_rag_enabled(chat_id: str) -> bool:
    _ensure_ready()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT rag_enabled FROM chats WHERE id=?", (chat_id,))
        row = cur.fetchone()
        return bool(row[0]) if row else False


def set_chat_rag_enabled(chat_id: str, enabled: bool) -> None:
    _ensure_ready()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE chats SET rag_enabled=? WHERE id=?", (1 if enabled else 0, chat_id))
        conn.commit()
