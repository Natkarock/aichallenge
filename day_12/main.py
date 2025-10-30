import os
import asyncio
import json
import streamlit as st
import requests
from typing import Dict, Any

import agent  # должен содержать async def run_with_message(message: str) -> dict

st.set_page_config(page_title="⛅ Погодный MCP-агент", page_icon="⛅", layout="wide")
st.title("⛅ Погодный MCP-агент\nWeather API -> Image generation -> Create PDF")
st.caption("Этот интерфейс вызывает MCP с weather API, а потом генерирует изображение через MCP replicate. Далее генерирует и сохраняет PDF")


# ---- Sidebar ----
with st.sidebar:
    st.markdown("### Настройки")
    st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), key="OPENAI_MODEL")
    st.text_input("REPLICATE_MODEL", value=os.getenv("REPLICATE_MODEL", "google/nano-banana"), key="REPLICATE_MODEL")
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
        unsafe_allow_html=True
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


# ---- Core logic ----
async def _call_agent(prompt: str) -> Dict[str, Any]:
    os.environ["OPENAI_MODEL"] = st.session_state.OPENAI_MODEL
    os.environ["REPLICATE_MODEL"] = st.session_state.REPLICATE_MODEL

    result = await agent.run_with_message(prompt)
    if not result or "messages" not in result:
        return {"description": "Пустой ответ от агента", "images": [], "raw": result}

    final_msg = result["messages"][-1]
    content = getattr(final_msg, "content", "")

    try:
        data = json.loads(content)
    except Exception:
        data = {"description": content, "images": [], "warning": "Ответ не JSON"}

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
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Загружаю данные..."):
            result = call_agent(prompt)

        desc = result.get("description", "")
        if desc:
            st.markdown(desc)
            st.session_state.history.append({"role": "assistant", "content": desc})

        # --- Загрузка изображений с shimmer ---
        for url in result.get("images", []):
            _image_with_shimmer(url)

        if "warning" in result:
            st.warning(result["warning"])

        # отладочная секция
        with st.expander("Показать полный result"):
            st.json(result["raw"])

st.markdown("---")
