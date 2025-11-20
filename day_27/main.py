import os
import io
import pathlib
from datetime import datetime
from typing import Dict, Any, List
import hashlib  # <<< добавили для хэша аудио

import streamlit as st

# === ИМПОРТЫ ДЛЯ ГОЛОСА ===
from streamlit_mic_recorder import mic_recorder
from speech_recognition_service import transcribe_audio_bytes

# ==========================

from cache import (
    load_store as _load_store,
    save_store as _save_store,
    new_chat as _new_chat,
    delete_chat,
    update_chat_summary,
    get_chat_summary,
    get_chat_rag_enabled,
    set_chat_rag_enabled,
)

from rag_store import (
    add_files,
    list_files,
    remove_file,
    similarity_search,
)

from llm import (
    generate_reply,
    summarize_messages,
    generate_rag_reply,
    set_is_Local,
    get_is_local,
)

st.set_page_config(page_title="RAG Chat Agent", page_icon="🧠", layout="wide")

# Инициализируем мапу источников в сессии (если ты ей пользуешься)
if "message_sources" not in st.session_state:
    st.session_state["message_sources"] = {}

# Хэш последней обработанной голосовой записи, чтобы не зацикливаться
if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = None


# ============================================================
# Файлы
# ============================================================
def _extract_text_from_upload(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.read()
    text = ""
    suffix = pathlib.Path(name).suffix.lower()

    if suffix in [".txt", ".md", ".py", ".log", ".json", ".csv"]:
        text = data.decode("utf-8", errors="ignore")

    elif suffix in [".xls", ".xlsx"]:
        try:
            import pandas as pd

            df = pd.read_excel(io.BytesIO(data))
            text = df.to_csv(index=False)
        except Exception:
            text = ""

    elif suffix == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            text = ""

    elif suffix == ".docx":
        try:
            import docx

            doc = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            text = ""

    else:
        text = data.decode("utf-8", errors="ignore")

    return text, name


def _render_file_upload():
    st.sidebar.subheader("Индексируемые файлы")

    uploaded_files = st.sidebar.file_uploader(
        "Загрузите файлы",
        type=["txt", "md", "pdf", "docx", "xls", "xlsx", "py", "log", "json", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.sidebar.button("Добавить в индекс"):
        docs_to_add = []
        for uf in uploaded_files:
            text, name = _extract_text_from_upload(uf)
            if text.strip():
                docs_to_add.append((text, name))

        if docs_to_add:
            add_files(docs_to_add)
            st.sidebar.success(f"Добавлено: {len(docs_to_add)}")

    existing = list_files()
    if existing:
        st.sidebar.markdown("**Уже в индексе:**")
        for f in existing:
            file_id = f["file_id"]
            file_name = f["file_name"]
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(file_name)
            with col2:
                if st.button("❌", key=f"rm_{file_id}"):
                    remove_file(file_id)
                    st.rerun()


# ============================================================
# Store
# ============================================================
def _load_store_safe():
    try:
        return _load_store()
    except Exception:
        return {"chats": []}


def _save_store_safe(store):
    try:
        _save_store(store)
    except Exception:
        pass


def _ensure_chat(store, chat_id):
    for c in store.get("chats", []):
        if c["id"] == chat_id:
            return c
    return None


# ============================================================
# Sidebar – список чатов
# ============================================================
def _render_chat_list(store):
    st.sidebar.header("Чаты")

    if "selected_chat_id" not in st.session_state:
        st.session_state["selected_chat_id"] = None

    if st.sidebar.button("➕ Новый чат", use_container_width=True):
        nc = _new_chat("Новый чат")
        store["chats"].insert(0, nc)
        _save_store_safe(store)
        st.session_state["selected_chat_id"] = nc["id"]
        st.rerun()

    chats = store.get("chats", [])
    if not chats:
        return None

    ids = [c["id"] for c in chats]
    titles = [(c["title"] or "Без названия") for c in chats]

    current_id = st.session_state["selected_chat_id"]
    index = ids.index(current_id) if current_id in ids else 0

    selected_index = st.sidebar.selectbox(
        "Выберите чат",
        list(range(len(chats))),
        index=index,
        format_func=lambda i: titles[i],
        key="select_chat_id",
    )

    selected_id = ids[selected_index]
    st.session_state["selected_chat_id"] = selected_id

    if st.sidebar.button("🗑 Удалить чат", use_container_width=True):
        delete_chat(store, selected_id)
        st.session_state["selected_chat_id"] = None
        st.rerun()

    return selected_id


# ============================================================
# Настройки чата
# ============================================================
def _render_chat_settings(chat_id):
    st.sidebar.subheader("Настройки чата")

    rag_enabled = get_chat_rag_enabled(chat_id)
    new_val = st.sidebar.checkbox("Использовать RAG", value=rag_enabled)

    if new_val != rag_enabled:
        set_chat_rag_enabled(chat_id, new_val)

    show_sources = st.sidebar.checkbox("Показывать источники", value=False)

    is_local = get_is_local()
    use_local = st.sidebar.checkbox("Использовать локальную модель", value=get_is_local)
    if use_local != is_local:
        set_is_Local(use_local)

    with st.sidebar:
        audio_data = mic_recorder(
            start_prompt="🎤 Записать",
            stop_prompt="■ Остановить",
            key=f"mic_{chat_id}",
            just_once=True,
            use_container_width=True,
            format="wav",
        )

    return new_val, show_sources, audio_data


# ============================================================
# История сообщений
# ============================================================
def render_messages(chat):
    for m in chat["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("🧠 Твой дружелюбный помощник")

    store = _load_store_safe()

    _render_file_upload()

    chat_id = _render_chat_list(store)
    if not chat_id:
        st.info("Создайте чат")
        return

    chat = _ensure_chat(store, chat_id)
    if not chat:
        st.error("Чат не найден")
        return

    rag_enabled, show_sources, audio_data = _render_chat_settings(chat_id)

    summary = get_chat_summary(chat_id)

    st.markdown(f"### Чат: {chat['title']}")

    # история
    render_messages(chat)

    # ============================================================
    # 📌 ВВОД СООБЩЕНИЯ (текст + голос)
    # ============================================================
    text_prompt = st.chat_input("Введите сообщение")

    # audio_data = mic_recorder(
    #     start_prompt="🎤 Записать",
    #     stop_prompt="■ Остановить",
    #     key=f"mic_{chat_id}",
    #     just_once=True,
    #     use_container_width=True,
    #     format="wav",
    # )

    prompt = None

    # если записан голос → распознаём ОДИН РАЗ на этот аудиоклип
    if audio_data and audio_data.get("bytes"):
        audio_bytes = audio_data["bytes"]
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        # обрабатываем только если это НОВАЯ запись
        if audio_hash != st.session_state.get("last_audio_hash"):
            with st.spinner("🎧 Распознаю речь..."):
                voice_text = transcribe_audio_bytes(audio_bytes, language="ru")
            if voice_text:
                prompt = voice_text
                st.session_state["last_audio_hash"] = audio_hash

    # если голосового ввода не было в этом проходе – используем текст
    if prompt is None:
        prompt = text_prompt

    # ============================================================
    # ОТПРАВКА СООБЩЕНИЯ
    # ============================================================
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        chat["messages"].append({"role": "user", "content": prompt})

        # генерируем ответ
        with st.chat_message("assistant"):
            with st.status("Генерирую ответ…", expanded=True) as status:
                if rag_enabled:
                    retrieved = similarity_search(prompt, k=3)
                    reply = generate_rag_reply(
                        prompt,
                        summary=summary,
                        retrieved_docs=retrieved,
                        with_sources=show_sources,
                    )
                else:
                    reply = generate_reply(chat["messages"], summary)

            st.write(reply)

        chat["messages"].append({"role": "assistant", "content": reply})

        # пересчёт summary
        new_summary = summarize_messages(summary, chat["messages"])
        update_chat_summary(chat_id, new_summary)

        _save_store_safe(store)

        st.rerun()


if __name__ == "__main__":
    main()
