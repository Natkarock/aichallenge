# main.py — Streamlit GUI for MCP-enabled Agent
# Run: streamlit run main.py

import asyncio
import json
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="MCP Агент — Chat UI", page_icon="💬", layout="centered")


# ---------------------------- Utilities --------------------------------------
def _obj_to_dict_safe(x):
    """Best-effort conversion of message-like objects to a simple dict for display."""
    try:
        base = {
            "type": getattr(x, "__class__", type("X", (), {})).__name__,
            "content": getattr(x, "content", repr(x)),
        }
        tc = getattr(x, "tool_calls", None)
        if tc:
            base["tool_calls"] = tc
        return base
    except Exception:
        return {"repr": repr(x)}


def _render_logs(messages_like):
    """Render the 'Полная история сообщений' under an expander."""
    with st.expander("🔎 Полная история сообщений (раскрыть/скрыть)"):
        if not messages_like:
            st.write("Нет логов.")
            return
        for i, msg in enumerate(messages_like, start=1):
            data = _obj_to_dict_safe(msg) if not isinstance(msg, dict) else msg
            st.markdown(f"**{i}. {data.get('type', 'Message')}**")
            st.write(data.get("content", ""))
            tool_calls = data.get("tool_calls")
            if tool_calls is not None:
                # Pretty JSON for tool calls / traces
                try:
                    st.code(json.dumps(tool_calls, ensure_ascii=False, indent=2))
                except Exception:
                    st.code(str(tool_calls))


# ---------------------------- Agent bridge -----------------------------------
def _import_run_agent():
    """Try to import run_agent from local agent.py (MCP-enabled).
    Required signature:
        run_agent(user_input: str, chat_history: list[dict]) -> dict with keys: text, messages
    """
    try:
        import importlib.util
        import sys
        agent_path = Path(__file__).with_name("agent.py")
        if not agent_path.exists():
            return None

        spec = importlib.util.spec_from_file_location("agent", str(agent_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = mod
        spec.loader.exec_module(mod)  # type: ignore

        if hasattr(mod, "run_agent") and callable(getattr(mod, "run_agent")):
            return getattr(mod, "run_agent")
        return None
    except Exception:
        # Show in sidebar for debugging
        st.sidebar.error("Ошибка импорта run_agent из agent.py")
        st.sidebar.code(traceback.format_exc())
        return None


RUN_AGENT = _import_run_agent()


async def call_agent_async(user_input: str, history: list[dict]):
    """Call the real agent if available; else a fallback echo agent."""
    if RUN_AGENT is not None:
        try:
            result = RUN_AGENT(user_input, history)
            # Support both sync and async implementations
            if asyncio.iscoroutine(result):
                result = await result
            text = result.get("text", "")
            messages = result.get("messages", [])
            return text, messages
        except Exception as e:
            return f"⚠️ Ошибка в агенте: {e}", [{"type": "Traceback", "content": traceback.format_exc()}]

    # ---- Fallback echo agent (so the UI runs even without a proper agent) ----
    await asyncio.sleep(0.05)
    return f"(Fallback) Вы сказали: {user_input}", [{"type": "Echo", "content": f"Echo of: {user_input}", "tool_calls": []}]


def call_agent(user_input: str, history: list[dict]):
    """Helper to call async function from Streamlit's sync context."""
    try:
        return asyncio.run(call_agent_async(user_input, history))
    except RuntimeError:
        # If already in an event loop (e.g., future Streamlit), use a new loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(call_agent_async(user_input, history))
        finally:
            loop.close()


# ---------------------------- State & UI -------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {role: "user"/"assistant", content: str, logs: list}

st.title("💬 MCP Агент — Подбор маршрута")

# Render chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            _render_logs(m.get("logs"))

# Input
user_text = st.chat_input("Напишите сообщение агенту…")
if user_text:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": user_text})

    # Query agent
    with st.spinner("Агент думает…"):
        reply, logs = call_agent(user_text, st.session_state.chat)

    # Add assistant message with logs
    st.session_state.chat.append({"role": "assistant", "content": reply, "logs": logs})

    # Rerun to display new content
    st.rerun()
