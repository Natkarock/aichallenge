import os, base64
import asyncio
import json
import streamlit as st
import requests
from typing import Dict, Any
import streamlit.components.v1 as components

import agent  # должен содержать async def run_with_message(message: str) -> dict

st.set_page_config(page_title="⛅ Погодный MCP-агент", page_icon="⛅", layout="wide")
st.title("⛅ Погодный MCP-агент\nWeather API -> Image generation -> Create PDF")
st.caption(
    "Этот интерфейс вызывает MCP с weather API, а потом генерирует изображение через MCP replicate. Далее генерирует и сохраняет PDF"
)


# ---- Sidebar ----
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

# ---- Chat state ----
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


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
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        ph.image(r.content, use_column_width=True)
    except Exception as e:
        ph.error(f"Не удалось загрузить изображение: {url}\n{e}")


def _render_pdf_inline(pdf_path: str):
    if not pdf_path:
        return
    st.markdown("#### PDF")
    # URL → просто в iframe + ссылка
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
    # Локальный путь → читаем и встраиваем
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        st.error(f"Не удалось открыть PDF: {e}")
        return

    st.download_button(
        "Скачать PDF",
        data=pdf_bytes,
        file_name=os.path.basename(pdf_path),
        mime="application/pdf",
        use_container_width=True,
    )


# ---- Core logic ----
async def _call_agent(prompt: str) -> Dict[str, Any]:
    os.environ["OPENAI_MODEL"] = st.session_state.OPENAI_MODEL
    os.environ["REPLICATE_MODEL"] = st.session_state.REPLICATE_MODEL

    # НЕ chat_message — обычные контейнеры
    outer_container = st.container()
    progress_placeholder = outer_container.empty()
    result_placeholder = outer_container.empty()

    # Очередь для live-сообщений
    q: asyncio.Queue[str] = asyncio.Queue()

    def emit_to_ui(s: str):
        try:
            q.put_nowait(str(s))
        except Exception:
            pass

    # Подключаем LiveData-подобный эмиттер
    agent.set_live_emitter(emit_to_ui)

    # Стартуем агент в фоне
    task = asyncio.create_task(agent.run_with_message(prompt))

    # Рендер прогресса в одном плейсхолдере
    live_lines: list[str] = []

    async def render_progress():
        if live_lines:
            progress_placeholder.markdown(
                "###### Прогресс выполнения\n"
                + "\n".join(f"- {line}" for line in live_lines)
            )
        else:
            progress_placeholder.markdown("#### Прогресс\n- запуск…")

    # Помповый цикл
    while not task.done():
        drained = False
        while True:
            try:
                msg = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                drained = True
                live_lines.append(msg)
        if drained:
            await render_progress()
        await asyncio.sleep(0.05)

    # Дочищаем очередь
    while True:
        try:
            msg = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            live_lines.append(msg)
    await render_progress()

    # Получаем результат
    try:
        result = await task
    except Exception as e:
        result_placeholder.error(f"Ошибка агента: {e}")
        return {"description": f"Ошибка агента: {e}", "images": [], "raw": None}

    if not result or "messages" not in result:
        result_placeholder.warning("Пустой ответ от агента")
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
        return asyncio.run(_call_agent(prompt))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_call_agent(prompt))


# ---- Streamlit chat ----
prompt = st.chat_input("Например: Погода в Краснодаре")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    # Рисуем пузырь пользователя
    with st.chat_message("user"):
        st.markdown(prompt)

    # Один раз рисуем пузырь ассистента и внутри НЕ вызываем chat_message снова
    with st.chat_message("assistant"):
        with st.spinner("Загружаю данные..."):
            # тут можно написать что-то статичное (например заголовок),
            # а прогресс и итог нарисует _call_agent внутри своих контейнеров
            result = asyncio.run(_call_agent(prompt))

            desc = result.get("description", "")
            if desc:
                st.markdown(desc)
                st.session_state.history.append({"role": "assistant", "content": desc})
            # --- Загрузка изображений с shimmer ---
            for url in result.get("images", []):
                _image_with_shimmer(url)

            # --- PDF (и URL, и локальный путь поддерживаются) ---
            pdf_path = result.get("path") or result.get("pdf")
            if pdf_path:
                _render_pdf_inline(pdf_path)

            if "warning" in result:
                st.warning(result["warning"])

            # отладочная секция
            with st.expander("Показать полный result"):
                st.json(result["raw"])

st.markdown("---")
