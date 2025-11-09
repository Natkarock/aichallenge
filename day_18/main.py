import os
import base64
import asyncio
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import requests
import streamlit.components.v1 as components

import agent
from llm import (
    generate_reply,
)  # должен содержать async def run_with_message(message: str) -> dict
from cache import load_store as _load_store
from cache import save_store as _save_store
from cache import new_chat as _new_chat
from cache import delete_chat as delete_chat
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# --- Stable sidebar with chat select & delete ---
def ensure_sidebar_chat_ui(store):
    try:
        import streamlit as st  # type: ignore
    except Exception:
        # Streamlit not available; just pick first chat id
        selected_id = store["chats"][0]["id"] if store.get("chats") else None
        return selected_id, store

    chats = store.get("chats", [])
    title_by_id = {c.get("id"): (c.get("title") or c.get("id")) for c in chats}
    ids = [c.get("id") for c in chats]

    if "selected_chat_id" not in st.session_state:
        st.session_state["selected_chat_id"] = ids[0] if ids else None

    # Compute index for selectbox
    idx = 0
    if st.session_state.get("selected_chat_id") in ids:
        idx = ids.index(st.session_state["selected_chat_id"])

    selected_id = st.sidebar.selectbox(
        "Выберите чат",
        options=ids if ids else ["__no_chats__"],
        index=idx if ids else 0,
        format_func=(
            lambda cid: (
                title_by_id.get(cid, "Нет чатов")
                if cid != "__no_chats__"
                else "Нет чатов"
            )
        ),
        key="selected_chat_id_select",
    )
    if selected_id != "__no_chats__":
        st.session_state["selected_chat_id"] = selected_id
    else:
        st.session_state["selected_chat_id"] = None

    # Delete button — always visible, bound to the selected chat id
    if st.sidebar.button(
        "🗑️ Удалить чат",
        use_container_width=True,
        disabled=not bool(st.session_state.get("selected_chat_id")),
    ):
        sid = st.session_state.get("selected_chat_id")
        if sid:
            store = delete_chat(store, sid)
            # Persist
            try:
                _save_store(store)
            except Exception:
                pass
            # Reset selection
            new_ids = [c.get("id") for c in store.get("chats", [])]
            st.session_state["selected_chat_id"] = new_ids[0] if new_ids else None
            # Clear selectbox cache so index recalculates
            st.session_state.pop("selected_chat_id_select", None)
            st.rerun()

    return st.session_state.get("selected_chat_id"), store


# ---------------- App config ----------------
st.set_page_config(
    page_title="⛅ Твой универсальный помощник", page_icon="⛅", layout="wide"
)
st.title("⛅ Твой универсальный помощник")

# === Sidebar multi-chat (vanilla) ===

BASE_DIR = os.path.dirname(__file__)
MEM_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEM_DIR, exist_ok=True)
CHATS_JSON = os.path.join(MEM_DIR, "chats.json")


store = _load_store()

# ---- Sidebar: select & delete chat (stable) ----
_selected_chat_id, store = ensure_sidebar_chat_ui(store)
chat = next(
    (c for c in store.get("chats", []) if c.get("id") == _selected_chat_id), None
)

if "selected_view" not in st.session_state:
    st.session_state.selected_view = "mcp"

st.sidebar.header("Навигация")
if st.sidebar.button("➕ Новый чат", use_container_width=True):
    chat = _new_chat()

    # --- Delete chat UI (safe-guarded) ---
    try:
        import streamlit as st  # type: ignore

        if chat and isinstance(chat, dict):
            if st.sidebar.button("🗑️ Удалить чат", use_container_width=True):
                store = delete_chat(store, chat.get("id"))
                _save_store(store)
                st.success("Чат удалён")
                st.rerun()
    except Exception:
        # If Streamlit is not available or sidebar absent, skip UI
        pass

    store["chats"].insert(0, chat)
    _save_store(store)
    st.session_state.selected_view = chat["id"]

st.sidebar.markdown("### 💬 Мои чаты")
if not store["chats"]:
    st.sidebar.caption("Пока нет чатов. Нажмите “Новый чат”.")

for chat in store["chats"]:
    label = chat["title"] or "Без названия"
    if st.sidebar.button(
        f"🗨️ {label}", key=f"sel_{chat['id']}", use_container_width=True
    ):
        st.session_state.selected_view = chat["id"]

st.sidebar.markdown("---")
if st.sidebar.button("⛅ Открыть Weather MCP", use_container_width=True):
    st.session_state.selected_view = "mcp"

# Route: if not MCP, render vanilla chat and stop the script to keep original MCP code intact
if st.session_state.selected_view != "mcp":
    chat_id = st.session_state.selected_view
    chat = next((c for c in store["chats"] if c["id"] == chat_id), None)
    if not chat:
        st.error("Чат не найден.")
        st.stop()

    st.subheader(f"💬 Чат: {chat['title'] or 'Без названия'}")
    for m in chat["messages"]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    user_msg = st.chat_input("Напишите сообщение…")
    if user_msg:
        chat["messages"].append({"role": "user", "content": user_msg})
        if len([m for m in chat["messages"] if m["role"] == "user"]) == 1:
            chat["title"] = user_msg[:40] + ("…" if len(user_msg) > 40 else "")
        _save_store(store)

        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Думаю…"):
                try:
                    llm = ChatOpenAI(
                        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0.2
                    )
                    msgs = [SystemMessage(content="Ты дружелюбный помощник.")]
                    for m in chat["messages"]:
                        msgs.append(
                            HumanMessage(content=m["content"])
                            if m["role"] == "user"
                            else AIMessage(content=m["content"])
                        )
                    reply = generate_reply(chat["messages"])
                except Exception as e:
                    reply = f"Ошибка LLM: {e}"
            st.markdown(reply)
        chat["messages"].append({"role": "assistant", "content": reply})
        _save_store(store)

    st.stop()
st.caption(
    "Этот интерфейс вызывает MCP с weather API, затем генерирует изображения через MCP Replicate и создаёт PDF."
)

# ---------------- Sidebar -------------------
with st.sidebar:
    st.markdown("### Настройки")
    st.text_input(
        "OPENAI_MODEL",
        value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        key="OPENAI_MODEL",
    )
    st.text_input(
        "REPLICATE_MODEL",
        value=os.getenv("REPLICATE_MODEL", "google/nano-banana"),
        key="REPLICATE_MODEL",
    )
    st.markdown("---")

# ---------------- Chat state ----------------
if "history" not in st.session_state:
    # Для ассистента сохраняем: desc, images, pdf, raw
    st.session_state.history: List[Dict[str, Any]] = []

# Рендерим прошлую историю
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("desc"):
                st.markdown(msg["desc"])
            for url in msg.get("images", []):
                st.image(url, use_container_width=True)
            if msg.get("pdf"):
                # Встроим PDF/кнопку, не вызывая rerun
                def _render_pdf_inline(pdf_path: str):
                    if not pdf_path:
                        return

                    filename = _os.path.basename(pdf_path)
                    components.html(
                        f"""
                        <a download="{filename}"
                           href="data:application/pdf;base64,{b64}"
                           style="display:inline-block;margin-top:8px;padding:10px 14px;border-radius:8px;border:1px solid #ccc;text-decoration:none">
                           Скачать PDF
                        </a>
                        """,
                        height=60,
                    )

                _render_pdf_inline(msg["pdf"])
        else:
            st.markdown(msg.get("content", ""))


# ---------- Шиммер ----------
def _show_shimmer(placeholder, aspect_ratio: str = "16/9"):
    """Отображает серый shimmer-плейсхолдер, пока картинка грузится."""
    placeholder.markdown(
        f"""
        <div class="shimmer" style="
            width:100%;
            aspect-ratio:{aspect_ratio};
            border-radius:12px;
            background:#f6f7f8;
            position:relative;
            overflow:hidden;">
        </div>
        <style>
        .shimmer::before {{
            content:'';
            position:absolute;
            top:0; left:-150%;
            height:100%; width:50%;
            background:linear-gradient(90deg, rgba(246,247,248,0) 0%,
                                              rgba(220,220,220,0.7) 50%,
                                              rgba(246,247,248,0) 100%);
            animation: shimmer 1.2s infinite;
        }}
        @keyframes shimmer {{
            0% {{ left:-150%; }}
            100% {{ left:150%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _image_with_shimmer(url: str, aspect_ratio: str = "16/9", timeout: int = 25):
    """Показывает shimmer, пока изображение загружается по URL."""
    ph = st.empty()
    _show_shimmer(ph, aspect_ratio=aspect_ratio)
    try:
        import requests as _rq

        r = _rq.get(url, timeout=timeout)
        r.raise_for_status()
        ph.image(r.content, use_column_width=True)
    except Exception as e:
        ph.error(f"Не удалось загрузить изображение: {url}\n{e}")


def _render_pdf_inline(pdf_path: str):
    """Показывает PDF и даёт скачать без rerun (через data: URL). Поддерживает локальный путь и URL."""
    if not pdf_path:
        return
    st.markdown("#### PDF")
    if pdf_path.lower().startswith(("http://", "https://")):
        components.html(
            f'<iframe src="{pdf_path}" width="100%" height="600" style="border:none;"></iframe>',
            height=620,
            scrolling=True,
        )
        st.link_button(
            "Открыть PDF в новой вкладке", pdf_path, use_container_width=True
        )
        return
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        st.error(f"Не удалось открыть PDF: {e}")
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    import os as _os

    filename = _os.path.basename(pdf_path)
    components.html(
        f"""
        <a download="{filename}"
           href="data:application/pdf;base64,{b64}"
           style="display:inline-block;margin-top:8px;padding:10px 14px;border-radius:8px;border:1px solid #ccc;text-decoration:none">
           Скачать PDF
        </a>
        """,
        height=60,
    )


# ---- Core logic ----
async def _call_agent(prompt: str, progress_placeholder) -> Dict[str, Any]:
    os.environ["OPENAI_MODEL"] = st.session_state.OPENAI_MODEL
    os.environ["REPLICATE_MODEL"] = st.session_state.REPLICATE_MODEL

    # Очередь для live-сообщений
    q: asyncio.Queue[str] = asyncio.Queue()

    def emit_to_ui(s: str):
        try:
            q.put_nowait(str(s))
        except Exception:
            pass

    agent.set_live_emitter(emit_to_ui)

    # Стартуем агент в фоне
    task = asyncio.create_task(agent.run_with_message(prompt))

    # Рендер прогресса в одном плейсхолдере (внутри ассистент-бабла)
    live_lines: List[str] = []

    def render_progress():
        if live_lines:
            progress_placeholder.markdown(
                "###### Прогресс выполнения\n"
                + "\n".join(f"- {line}" for line in live_lines)
            )
        else:
            progress_placeholder.markdown("###### Прогресс выполнения\n- запуск…")

    # Помповый цикл
    while not task.done():
        drained = False
        while True:
            try:
                msg = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                live_lines.append(msg)
                drained = True
        if drained:
            render_progress()
        await asyncio.sleep(0.05)

    # Дочищаем очередь
    while True:
        try:
            msg = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            live_lines.append(msg)
    render_progress()

    # Получаем результат
    try:
        result = await task
    except Exception as e:
        progress_placeholder.error(f"Ошибка агента: {e}")
        return {"description": f"Ошибка агента: {e}", "images": [], "raw": None}

    if not result or "messages" not in result:
        progress_placeholder.warning("Пустой ответ от агента")
        return {"description": "Пустой ответ от агента", "images": [], "raw": result}

    # Достаём финальный контент
    final_msg = result["messages"][-1]
    content = (
        final_msg.get("content", "")
        if isinstance(final_msg, dict)
        else getattr(final_msg, "content", "")
    )

    # Пытаемся распарсить JSON, иначе текст
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except Exception:
        data = {"description": str(content), "images": [], "warning": "Ответ не JSON"}

    data["raw"] = result
    return data


def call_agent(prompt: str) -> Dict[str, Any]:
    try:
        return asyncio.run(
            _call_agent(prompt, st.empty())
        )  # fallback, если зовут напрямую
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_call_agent(prompt, st.empty()))


# ---- Streamlit chat ----
prompt = st.chat_input("Например: Погода в Краснодаре")
if prompt:
    # История: пользователь
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ассистент
    with st.chat_message("assistant"):
        # Плейсхолдер для прогресса именно внутри этого бабла
        progress_placeholder = st.empty()

        # Спиннер + live-обновления в progress_placeholder
        with st.spinner("Загружаю данные..."):
            result = asyncio.run(_call_agent(prompt, progress_placeholder))

        # Финальный вывод ассистента
        desc = result.get("description", "")
        images = result.get("images", [])
        pdf_path = result.get("path") or result.get("pdf")

        if desc:
            st.markdown(desc)
        for url in images:
            _image_with_shimmer(url)

        if pdf_path:
            _render_pdf_inline(pdf_path)

        if "warning" in result:
            st.warning(result["warning"])

        with st.expander("Показать полный result"):
            st.json(result["raw"])

        # Сохраним в историю полный ответ ассистента (для устойчивости при rerun)
        st.session_state.history.append(
            {
                "role": "assistant",
                "desc": desc,
                "images": images,
                "pdf": pdf_path,
                "raw": result.get("raw"),
            }
        )

st.markdown("---")
