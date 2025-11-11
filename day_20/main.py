import os
import io
import pathlib
from datetime import datetime
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
)

st.set_page_config(page_title="RAG Chat Agent", page_icon="🧠", layout="wide")


# ---------------- Helpers ----------------
def _upload_sig(files):
    """Сигнатура выборки из file_uploader (имя + размер) для защиты от рекурсивной обработки."""
    if not files:
        return []
    sig = []
    for f in files:
        size = getattr(f, "size", None)
        if size is None:
            try:
                size = len(f.getbuffer())
            except Exception:
                size = None
        sig.append((f.name, size))
    return sig


def _decode_best_effort(raw: bytes) -> str:
    """Попробовать декодировать текст тремя популярными кодировками."""
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    # как крайний случай: заменить невалидные символы
    return raw.decode("utf-8", errors="ignore")


def _excel_to_text(raw: bytes, filename: str) -> str:
    """Извлечь текст из Excel (.xls/.xlsx) — все листы, TSV-представление."""
    import pandas as pd

    bio = io.BytesIO(raw)
    try:
        xls = pd.ExcelFile(bio)  # использует openpyxl для xlsx и xlrd для xls
    except Exception as e:
        return f"[excel-parse-error:{filename}] {e}"

    parts = [f"# file: {filename}"]
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet, dtype=str, na_filter=False)
        except Exception as e:
            parts.append(f"## sheet: {sheet}\n[excel-parse-error:{sheet}] {e}")
            continue
        # TSV
        tsv = df.to_csv(sep="\t", index=False)
        parts.append(f"## sheet: {sheet}\n{tsv}")
    return "\n\n".join(parts)


def _extract_text_from_upload(f) -> tuple[str, str]:
    """
    Привести загруженный файл к тексту для эмбеддингов.
    Возвращает (text, display_name).
    """
    name = f.name or "uploaded"
    suffix = pathlib.Path(name).suffix.lower()

    raw = f.read()
    if not raw:
        return ("", name)

    # Excel
    if suffix in (".xls", ".xlsx"):
        return (_excel_to_text(raw, name), name)

    # Попробовать как текст
    text = _decode_best_effort(raw)
    return (text, name)


# ---------------- Sidebar: chat UI + RAG + uploader ----------------
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

    if "selected_chat_id" not in st.session_state:
        st.session_state["selected_chat_id"] = ids[0] if ids else None

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
            store = delete_chat(store, sid)
            try:
                _save_store(store)
            except Exception:
                pass
            new_ids = [c.get("id") for c in store.get("chats", [])]
            st.session_state["selected_chat_id"] = new_ids[0] if new_ids else None
            st.session_state.pop("select_chat_id", None)
            st.rerun()

    # Флаг RAG (читаем из БД каждый рендер — источник истины)
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

    # ── Загрузка документов — анти-рекурсия по сигнатуре ───────────────────────
    uploaded = st.sidebar.file_uploader(
        "Загрузить документы (текстовые, .xls/.xlsx и др.)",
        # type=None => любые файлы
        accept_multiple_files=True,
        key="uploader_docs",
    )

    current_sig = _upload_sig(uploaded)
    last_sig = st.session_state.get("last_upload_sig")
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
                # Важно: заново прочитать буфер — file_uploader может «залипать» курсор
                f.seek(0)
                text, display_name = _extract_text_from_upload(f)
                if text and text.strip():
                    pairs.append((text, display_name))
            try:
                if pairs:
                    records = add_files(pairs)
                    total_chunks = sum(r.get("num_chunks", 0) for r in records)
                    st.sidebar.success(
                        f"Загружено файлов: {len(records)}, всего чанков: {total_chunks}"
                    )
                else:
                    st.sidebar.warning(
                        "Файлы не распознаны как текст/таблицы или пустые."
                    )
                st.session_state["last_upload_sig"] = current_sig
                st.session_state["upload_processed"] = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Не удалось добавить документы: {e}")

    # После rerun: сброс флага, чтобы следующая новая выборка обработалась
    if st.session_state.get("upload_processed", False):
        st.session_state["upload_processed"] = False

    # ── СПИСОК ЗАГРУЖЕННЫХ ФАЙЛОВ ПОД file_uploader (без дублирования) ────────
    with st.sidebar.container():
        try:
            _files = list_files()
            if not _files:
                st.caption("Загруженные файлы: пока пусто")
            else:
                st.caption("Загруженные файлы:")
                for _f in _files:
                    cols = st.columns([5, 1])
                    with cols[0]:
                        st.write(
                            f"• {_f.get('name')}  \n_chunks_: {_f.get('num_chunks')}  \n_added_: {_f.get('added_at')}"
                        )
                    with cols[1]:
                        if st.button("✖", key=f"del_{_f.get('file_id')}_upl"):
                            try:
                                if remove_file(_f.get("file_id")):
                                    st.success("Удалено")
                                    st.rerun()
                                else:
                                    st.warning("Не найдено")
                            except Exception as _e:
                                st.error(f"Ошибка удаления: {_e}")
        except Exception as _e:
            st.error(f"Ошибка чтения списка файлов: {_e}")

    return st.session_state.get("selected_chat_id"), store, has_chats


def render_messages(chat):
    for m in chat.get("messages", []):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.write(m["content"])
        else:
            with st.chat_message("assistant"):
                st.write(m["content"])


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

    # Читаем summary и флаг RAG из БД (источник истины)
    chat_id = chat.get("id")
    summary = get_chat_summary(chat_id) or chat.get("summary")
    rag_enabled = get_chat_rag_enabled(chat_id)

    # Header
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.write(f"**Чат:** {chat.get('title')}")
    with col2:
        st.write(f"**RAG:** {'включен' if rag_enabled else 'выключен'}")
    with col3:
        st.write(f"**Сообщений:** {len(chat.get('messages', []))}")

    render_messages(chat)

    prompt = st.chat_input("Введите сообщение")
    if prompt:
        # Добавляем сообщение пользователя
        chat["messages"].append(
            {
                "role": "user",
                "content": prompt,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        _save_store(store)  # чтобы в случае долгих ответов не потерять ввод

        # Плейсхолдер ассистента, чтобы статус был виден сразу
        with st.chat_message("assistant"):
            placeholder = st.empty()

            if rag_enabled:
                with st.status(
                    "Ищу контекст и генерирую ответ…", expanded=True
                ) as status:
                    placeholder.markdown("**Ищу контекст…**")
                    try:
                        q = prompt
                        retrieved = similarity_search(q, k=3)
                        status.update(
                            label="Генерирую ответ с учётом контекста…", state="running"
                        )
                        placeholder.markdown("**Генерирую ответ…**")

                        reply = generate_rag_reply(
                            q, summary=summary, retrieved_docs=retrieved
                        )

                        status.update(label="Готово", state="complete")
                    except Exception as e:
                        reply = f"RAG недоступен: {e}"
                        status.update(label="Ошибка RAG", state="error")
                placeholder.markdown(reply)
            else:
                with st.status("Генерирую ответ…", expanded=True) as status:
                    placeholder.markdown("**Генерирую ответ…**")
                    try:
                        reply = generate_reply(chat["messages"], summary=summary)
                        status.update(label="Готово", state="complete")
                    except Exception as e:
                        reply = f"Ошибка LLM: {e}"
                        status.update(label="Ошибка", state="error")
                placeholder.markdown(reply)

        # Добавляем ответ ассистента в историю
        chat["messages"].append(
            {
                "role": "assistant",
                "content": reply,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )

        # Саммаризация каждые 10 сообщений (вся история продолжает сохраняться)
        total = len(chat["messages"])
        if total % 10 == 0:
            prev_summary = summary or chat.get("summary")
            new_summary = summarize_messages(prev_summary, chat["messages"])
            chat["summary"] = new_summary
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
