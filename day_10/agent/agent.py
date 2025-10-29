# agent.py — MCP-enabled agent wrapper around OpenAI Responses API
# Provides a function run_agent(user_input: str, chat_history: list[dict]) -> dict
#
# It sends the conversation (briefly summarized) plus the latest user input to OpenAI Responses,
# and registers an MCP tool server (e.g., your weather tool) so the model can call it.
#
# ENV VARS required:
#   OPENAI_API_KEY        — your OpenAI API key
# Optional ENV VARS:
#   OPENAI_MODEL          — default "gpt-5"
#   MCP_SERVER_URL        — your MCP server URL (e.g., "https://your-mcp.example.com/sse")
#   MCP_SERVER_LABEL      — label for the server in tools list, default "mcp"
#   MCP_SERVER_DESC       — description for the server, default "MCP server"
#   MCP_REQUIRE_APPROVAL  — "never" (default) | "auto" | "always"
#
# NOTE:
# - This file keeps things minimal and robust. It uses Responses API with the MCP tool config.
# - We aggregate textual output and also return a compact log with the raw response payload so
#   the Streamlit UI can show it in a collapsible logs expander.

import os
import textwrap
from typing import List, Dict, Any

try:
    from openai import OpenAI
except Exception as e:  # provide a helpful error if SDK missing
    OpenAI = None

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")


def _build_history_summary(chat_history: List[Dict[str, str]], max_chars: int = 4000) -> str:
    """Create a compact, readable history block to give the model some context."""
    lines = []
    for m in chat_history[-10:]:  # last 10 turns should be enough for short context
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def _collect_text_output(resp: Any) -> str:
    """Try to collect text from Responses API result in a forward-compatible way."""
    # Newer SDKs expose .output_text directly.
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt

    # Fallback: attempt to traverse .output list
    try:
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            parts = []
            for item in out:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    parts.append(item.get("text", ""))
                else:
                    # Try object-like access (Pydantic models in SDKs)
                    t = getattr(item, "type", None)
                    if t == "output_text":
                        parts.append(getattr(item, "text", ""))
            return "\n".join([p for p in parts if p])
    except Exception:
        pass

    # Last resort: string repr
    return str(resp)


def run_agent(user_input: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Main entry for the Streamlit app.
    Returns:
        {
          "text": "<assistant final text>",
          "messages": [ { "type": "...", "content": "...", "tool_calls": {...}? }, ... ]
        }
    """
    if OpenAI is None:
        return {
            "text": "Установите пакет openai: pip install --upgrade openai",
            "messages": [{"type": "Error", "content": "openai SDK не установлен"}],
        }

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        return {
            "text": "Отсутствует OPENAI_API_KEY в окружении.",
            "messages": [{"type": "Error", "content": "Не найден OPENAI_API_KEY"}],
        }

    # Prepare MCP tool spec
    mcp_url = os.environ.get("MCP_SERVER_URL")
    mcp_label = os.environ.get("MCP_SERVER_LABEL", "mcp")
    mcp_desc = os.environ.get("MCP_SERVER_DESC", "MCP server")
    mcp_approval = os.environ.get("MCP_REQUIRE_APPROVAL", "never")  # never|auto|always

    tools = []
    if True:
        tools.append({
            "type": "mcp",
            "server_label": mcp_label,
            "server_description": mcp_desc,
            "server_url": "https://weathermcp-natkarock.amvera.io/mcp",
            "require_approval": mcp_approval
        })

    # Build prompt with compact history
    history_block = _build_history_summary(chat_history)
    system = textwrap.dedent("""\
        Ты агент-оператор составления туристических маршрутов. Тебе на вход подаются данные о городах следования, ты должен подобрать лучший маршрут.
    """)
    prompt = f"{system}\n\nRecent conversation:\n{history_block}\n\nUser: {user_input}"

    # Call Responses API
    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=prompt,
        tools=tools if tools else None,
    )

    final_text = _collect_text_output(response)

    # Minimal logs we attach for the Streamlit expander.
    # We include the raw response (safe) so tooling/tool_calls are visible.
    logs = [
        {"type": "UserMessage", "content": user_input},
        {
            "type": "OpenAIResponseSummary",
            "content": f"model={DEFAULT_MODEL}; tools={bool(tools)}",
        },
        {
            "type": "OpenAIResponseRaw",
            "content": "Сырые данные ответа (усечены для читаемости)",
            "tool_calls": getattr(response, "model_dump", lambda: str(response))(),
        },
    ]

    return {"text": final_text, "messages": logs}
