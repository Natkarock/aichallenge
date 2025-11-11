# RAG Chat Agent

This project implements a Streamlit-based chat agent with Retrieval-Augmented Generation (RAG), chat history summarization, persistent document storage, and multi-chat management.

## Features

### ✅ Chat with RAG or without RAG
- Each chat can independently toggle RAG mode.
- Baseline GPT model responses (no context).
- RAG responses using similarity search from a FAISS vector index.
- Automatic chat summarization every 10 messages.

### ✅ Document Upload & RAG Indexing
Supports:
- Any text-based file (.txt, .md, .py, .json, .html, etc.)
- Excel files (.xls, .xlsx)
- Automatic parsing and extraction of textual content
- Embedding generation and vector indexing using FAISS
- Documents persist between sessions
- Ability to delete individual documents

### ✅ Sidebar Controls
- Create/delete chats
- Switch between chats
- Toggle RAG per chat
- Upload files
- View and delete stored documents

### ✅ Persistent Storage
- Chat data stored in local SQLite/JSON store
- RAG files tracked in memory/rag/files.json
- FAISS index stored in memory/rag/faiss_index/

### ✅ Status Indicators
- “Ищу контекст…”
- “Генерирую ответ…”
- “Создаю эмбеддинги…”
- Ensures visibility before long operations

---

## File Structure

```
project/
│
├── main.py               # UI and agent logic
├── rag_store.py          # RAG storage, FAISS interactions, registry
├── llm.py                # RAG and baseline LLM calls
├── cache.py              # persistent chat storage and settings
│
└── memory/
    └── rag/
        ├── faiss_index/  # FAISS vector index
        ├── emb_cache/    # optional embedding cache
        └── files.json    # document registry
```

---

## Installation

### Install dependencies

```
pip install -r requirements.txt
```
---

## Running

```
streamlit run main.py
```

Will open in browser at:

```
http://localhost:8501
```

---

## How RAG Works

### Uploading Documents
1. User uploads documents
2. System extracts text/TSV
3. Text is split into chunks
4. Embeddings generated
5. Vector index updated
6. Registry updated

### Chat Flow with RAG
1. User asks a question
2. If RAG on:
   - Retrieve top-k semantic matches
   - Build augmented context
   - Generate final answer
3. Otherwise: model responds without context

---

## Supported File Types

### Text
- .txt  
- .md  
- .py  
- .json  
- .html  
- .csv  
- .js  
- .css  
- etc.

### Excel
- .xls  
- .xlsx  

### Any other binary file
- Attempt best-effort UTF‑8 decoding
- Non-text becomes empty or ignored

---

## Summarization Logic

Every 10 messages:  
```
summary = summarize_messages(previous_summary, all_messages)
```

Stored in DB for future use.

---

## Future Improvements

- Document preview
- Batch delete documents
- Export chat as PDF/Markdown
- Multi-vector-store support
- Streaming RAG output
