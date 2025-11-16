import os
import io
import pathlib
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st

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
    generate_agent_rag_reply,
)

st.set_page_config(page_title="RAG Chat Agent", page_icon="🧠", layout="wide")

# Инициализируем мапу источников в сессии
if "message_sources" not in st.session_state:
    st.session_state["message_sources"] = {}


# --------------
# Helpers для файлов
# --------------
def _extract_text_from_upload(uploaded_file):
    """
    Возвращает (text, display_name)
    """
    name = uploaded_file.name
    data = uploaded_file.read()
    text = ""
    suffix = pathlib.Path(name).suffix.lower()

    if suffix in [".txt", ".md", ".py", ".log", ".json", ".csv"]:
        text = data.decode("utf-8", errors="ignore")
    elif suffix in [".xls", ".xlsx"]:
        try:
            import pandas as pd

            xls = pd.ExcelFile(io.BytesIO(data))
            parts = []
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                parts.append(df.to_string())
            text = "\n\n".join(parts)
        except Exception as e:
            text = f"Не удалось прочитать Excel: {e}"
    else:
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    return text, name


def _build_sources_from_docs(retrieved_docs) -> List[Dict[str, Any]]:
    """
    Собирает список источников из документов RAG: dedup по file_path.
    """
    sources_by_path: Dict[str, Dict[str, Any]] = {}

    for d in retrieved_docs or []:
        try:
            meta = (
                getattr(d, "metadata", {})
                if hasattr(d, "metadata")
                else d.get("metadata", {})
            )
            content = (
                getattr(d, "page_content", "")
                if hasattr(d, "page_content")
                else d.get("page_content", "")
            )
        except Exception:
            meta, content = {}, ""

        file_path = meta.get("file_path") or meta.get("source") or "unknown"
        file_name = meta.get("file_name") or meta.get("source") or file_path
        snippet = (content or "").strip()[:200]

        if file_path not in sources_by_path:
            sources_by_path[file_path] = {
                "file_path": file_path,
                "file_name": file_name,
                "snippets": [],
            }
        if snippet and snippet not in sources_by_path[file_path]["snippets"]:
            sources_by_path[file_path]["snippets"].append(snippet)

    sources: List[Dict[str, Any]] = []
    for fp, data in sources_by_path.items():
        snippets = data.get("snippets") or []
        full_snippet = " ".join(snippets)
        if len(full_snippet) > 200:
            full_snippet = full_snippet[:197] + "..."
        sources.append(
            {
                "file_path": data["file_path"],
                "file_name": data["file_name"],
                "snippet": full_snippet,
            }
        )
    return sources


# --------------
# Sidebar: чаты и файлы
# --------------
def ensure_sidebar_chat_ui(store):
    # Кнопка: новый чат
    if st.sidebar.button("➕ Новый чат", use_container_width=True, key="btn_new_chat"):
        nc = _new_chat("Новый чат")
        store["chats"].insert(0, nc)
        try:
            _save_store(store)
        except Exception:
            pass
        st.session_state["selected_chat_id"] = nc["id"]
        st.session_state.pop("select_chat_id", None)
        st.rerun()

    chats = store.get("chats", [])
    ids = [c.get("id") for c in chats]
    titles = {c.get("id"): (c.get("title") or c.get("id")) for c in chats}

    # Выбор чата
    if ids:
        try:
            idx = (
                ids.index(st.session_state["selected_chat_id"])
                if st.session_state["selected_chat_id"] in ids
                else 0
            )
        except Exception:
            idx = 0
        selected_id = st.sidebar.selectbox(
            "Выберите чат",
            options=ids,
            index=idx,
            format_func=lambda cid: titles.get(cid, cid),
            key="select_chat_id",
        )
        st.session_state["selected_chat_id"] = selected_id
        has_chats = True
    else:
        st.sidebar.info("Нет чатов")
        selected_id = None
        has_chats = False

    # Кнопка: удалить чат
    if st.sidebar.button(
        "🗑️ Удалить чат",
        use_container_width=True,
        disabled=not bool(st.session_state.get("selected_chat_id")),
        key="btn_delete_chat",
    ):
        sid = st.session_state.get("selected_chat_id")
        if sid:
            try:
                store = delete_chat(store, sid)
                _save_store(store)
            except Exception as e:
                st.sidebar.error(f"Ошибка при удалении: {e}")
            if store.get("chats"):
                st.session_state["selected_chat_id"] = store["chats"][0]["id"]
            else:
                st.session_state.pop("selected_chat_id", None)
                st.session_state.pop("select_chat_id", None)
            st.rerun()

    # Флаг RAG для выбранного чата
    current_chat = next(
        (
            c
            for c in store.get("chats", [])
            if c.get("id") == st.session_state.get("selected_chat_id")
        ),
        None,
    )
    current_enabled_db = (
        get_chat_rag_enabled(current_chat.get("id")) if current_chat else False
    )
    rag_on = st.sidebar.checkbox(
        "🔍 Использовать RAG для этого чата",
        value=current_enabled_db,
        key="chk_rag_enabled",
    )
    if current_chat and rag_on != current_enabled_db:
        try:
            set_chat_rag_enabled(current_chat.get("id"), rag_on)
            current_chat["rag_enabled"] = rag_on  # зеркалим в in-memory
            _save_store(store)
        except Exception:
            pass

    # Режим: добавлять ли ссылки на файлы в ответы RAG
    show_sources = st.sidebar.checkbox(
        "📎 Включать ссылки на файлы в ответ",
        value=True,
        key="chk_show_sources",
        help="Если включено, LLM будет добавлять в конец ответа раздел 'Источники' с путями к файлам.",
    )

    # ── Загрузка документов ───────────────────────
    uploaded = st.sidebar.file_uploader(
        "Загрузить документы (текстовые, .xls/.xlsx и др.)",
        accept_multiple_files=True,
        key="uploader_docs",
    )

    # Сначала — индексация (чтобы новые файлы сразу появились в списке)
    if uploaded:
        sig_parts = []
        for f in uploaded:
            size = getattr(f, "size", None)
            if size is None:
                try:
                    pos = f.tell()
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(pos)
                except Exception:
                    size = 0
            sig_parts.append(f"{f.name}:{size}")
        current_sig = ";".join(sig_parts)
        last_sig = st.session_state.get("upload_sig")

        should_process = (
            uploaded
            and current_sig
            and current_sig != last_sig
            and not st.session_state.get("upload_processed", False)
        )

        if should_process:
            with st.spinner("Создаю эмбеддинги и обновляю индекс..."):
                pairs = []
                for f in uploaded:
                    f.seek(0)
                    text, display_name = _extract_text_from_upload(f)
                    if text and text.strip():
                        pairs.append((text, display_name))

                if pairs:
                    try:
                        add_files(pairs)
                        st.success(f"Добавлено файлов: {len(pairs)}")
                    except Exception as e:
                        st.error(f"Ошибка индексации: {e}")
                else:
                    st.info("Не удалось извлечь текст ни из одного файла.")

                st.session_state["upload_sig"] = current_sig
                st.session_state["upload_processed"] = True

    # Теперь показываем список проиндексированных файлов — уже после возможной индексации
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Проиндексированные файлы:**")

    try:
        files_info = list_files()
    except Exception as e:
        st.sidebar.error(f"Ошибка чтения метаданных файлов: {e}")
        files_info = []

    if not files_info:
        st.sidebar.write("_Пока нет файлов_")
    else:
        for f_info in files_info:
            fid = f_info.get("file_id")
            name = f_info.get("name")
            num_chunks = f_info.get("num_chunks")
            added_at = f_info.get("added_at")
            with st.sidebar.expander(f"{name} ({num_chunks} фрагментов)"):
                st.write(f"ID: `{fid}`")
                st.write(f"Добавлен: {added_at}")
                if st.sidebar.button(
                    "Удалить",
                    key=f"btn_del_file_{fid}",
                    use_container_width=True,
                ):
                    try:
                        ok = remove_file(fid)
                        if ok:
                            st.success("Файл удалён из индекса.")
                        else:
                            st.warning("Файл не найден в индексе.")
                    except Exception as _e:
                        st.sidebar.error(f"Ошибка удаления: {_e}")

    return st.session_state.get("selected_chat_id"), store, has_chats


def render_messages(chat):
    sources_map: Dict[str, List[Dict[str, Any]]] = st.session_state.get(
        "message_sources", {}
    )
    chat_id = chat.get("id")

    for m in chat.get("messages", []):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.write(m["content"])
        else:
            with st.chat_message("assistant"):
                st.write(m["content"])

                ts = m.get("ts")
                key = f"{chat_id}:{ts}" if ts else None
                sources = sources_map.get(key, [])
                if sources:
                    st.markdown("**Открыть источники:**")
                    for src in sources:
                        file_path = src["file_path"]
                        file_name = src["file_name"]
                        desc = src.get("snippet") or ""

                        with st.expander(f"{file_name}"):
                            st.code(file_path, language="text")
                            st.write(desc or "Фрагмент текста из файла.")
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                st.text_area(
                                    "Содержимое файла",
                                    value=content,
                                    height=200,
                                    disabled=True,
                                )
                            except Exception as e:
                                st.error(f"Не удалось открыть файл: {e}")


# ---------------- Main ----------------
def main():
    st.title("🧠 Твой дружелюбный помощник")
    store = _load_store()

    _selected_chat_id, store, _has_chats = ensure_sidebar_chat_ui(store)
    chat = next(
        (c for c in store.get("chats", []) if c.get("id") == _selected_chat_id), None
    )

    if not chat:
        st.info("Создайте новый чат в сайдбаре.")
        return

    chat_id = chat.get("id")
    summary = get_chat_summary(chat_id)
    rag_enabled = get_chat_rag_enabled(chat_id)

    # Header
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.write(f"**Чат:** {chat.get('title')}")
    with col2:
        st.write(f"**RAG:** {'включен' if rag_enabled else 'выключен'}")
    with col3:
        st.write(f"**Сообщений:** {len(chat.get('messages', []))}")

    # 1. Рисуем историю
    render_messages(chat)

    # 2. Ждём новое сообщение
    prompt = st.chat_input("Введите сообщение")

    if prompt:
        # 2.1 Сразу показываем сообщение пользователя
        with st.chat_message("user"):
            st.write(prompt)

        # 2.2 Добавляем в историю (ts поставит save_store при необходимости)
        chat["messages"].append({"role": "user", "content": prompt})

        # 2.3 Ответ ассистента
        with st.chat_message("assistant"):
            placeholder = st.empty()
            sources_for_message: List[Dict[str, Any]] = []
            reply = ""

            if rag_enabled:
                with st.status(
                    "Ищу контекст и генерирую ответ…", expanded=True
                ) as status:
                    placeholder.markdown("**Ищу контекст…**")
                    try:
                        q = prompt
                        retrieved = similarity_search(q, k=3)
                        status.update(
                            label="Генерирую ответ с учётом контекста…",
                            state="running",
                        )
                        placeholder.markdown("**Генерирую ответ…**")

                        with_sources_flag = st.session_state.get(
                            "chk_show_sources", False
                        )
                        # reply = generate_rag_reply(
                        #     q,
                        #     summary=summary,
                        #     retrieved_docs=retrieved,
                        #     with_sources=with_sources_flag,
                        # )
                        reply = generate_agent_rag_reply(
                            question=q,
                            summary=summary,
                        )

                        if with_sources_flag:
                            sources_for_message = _build_sources_from_docs(retrieved)

                        status.update(label="Готово", state="complete")
                    except Exception as e:
                        reply = f"RAG недоступен: {e}"
                        status.update(label="Ошибка RAG", state="error")

                placeholder.markdown(reply)

                if sources_for_message:
                    st.markdown("**Открыть источники:**")
                    for src in sources_for_message:
                        file_path = src["file_path"]
                        file_name = src["file_name"]
                        desc = src.get("snippet") or ""

                        with st.expander(f"{file_name}"):
                            st.code(file_path, language="text")
                            st.write(desc or "Фрагмент текста из файла.")
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                st.text_area(
                                    "Содержимое файла",
                                    value=content,
                                    height=200,
                                    disabled=True,
                                )
                            except Exception as e:
                                st.error(f"Не удалось открыть файл: {e}")

            else:
                with st.status("Генерирую ответ…", expanded=True) as status:
                    placeholder.markdown("**Генерирую ответ…**")
                    try:
                        reply = generate_reply(chat["messages"], summary=summary)
                        status.update(label="Готово", state="complete")
                    except Exception as e:
                        reply = f"Ошибка при обращении к LLM: {e}"
                        status.update(label="Ошибка", state="error")
                placeholder.markdown(reply)

        # 2.4 Сохраняем ответ ассистента и привязываем источники
        ts_now = datetime.utcnow().isoformat() + "Z"
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": reply,
            "ts": ts_now,
        }
        chat["messages"].append(assistant_msg)

        if sources_for_message:
            key = f"{chat_id}:{ts_now}"
            st.session_state["message_sources"][key] = sources_for_message

        # 2.5 Обновляем summary
        try:
            new_summary = summarize_messages(summary, chat["messages"])
        except Exception:
            new_summary = summary
        if new_summary and new_summary != summary:
            try:
                update_chat_summary(chat_id, new_summary)
            except Exception:
                pass

        _save_store(store)
        st.rerun()

    with st.expander("Сводка (summary) текущего чата"):
        st.write(summary or chat.get("summary") or "_пока пусто_")


if __name__ == "__main__":
    main()
