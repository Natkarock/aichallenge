from __future__ import annotations

import os
import json
from typing import List, Dict, Any

SYSTEM_PROMPT = "Ты дружелюбный помощник."
WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "12"))
SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", "250"))


def summarize_messages(
    previous_summary: str | None, messages: List[Dict[str, str]]
) -> str:
    instruction = (
        "Суммируй историю диалога в краткий конспект для последующего контекста. "
        "Сохраняй факты, вводы пользователя, принятые решения и договорённости. "
        f"Обнови прежнюю сводку (если дана). Объем ~{SUMMARY_TARGET_TOKENS} токенов."
    )
    if not messages:
        return previous_summary or ""
    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        last_n = WINDOW_MESSAGES
        ctx = messages[-last_n:]
        transcript_lines = []
        for m in ctx:
            role = m.get("role")
            content = m.get("content", "")
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)
        prev = previous_summary or "(пусто)"

        prompt = (
            f"system: Ты лаконичный ассистент суммаризации.\n"
            f"user: {instruction}\n\n"
            f"Предыдущая сводка:\n{prev}\n\n"
            f"Новая стенограмма:\n{transcript}\n\n"
            f"Верни только обновлённую сводку."
        )
        resp = client.responses.create(model=model, input=prompt)
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        return c.text.strip()
        return previous_summary or ""
    except Exception:
        return previous_summary or ""


def generate_reply(messages: List[Dict[str, str]], summary: str | None) -> str:
    if not messages:
        return "Привет! О чём поговорим?"

    ctx = messages[-WINDOW_MESSAGES:]
    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        lines = [f"system: {SYSTEM_PROMPT}"]
        if summary:
            lines.append(f"system: Краткая сводка контекста: {summary}")
        for m in ctx:
            lines.append(f"{m.get('role')}: {m.get('content','')}")
        prompt = "\n".join(lines)
        resp = client.responses.create(model=model, input=prompt)
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        return c.text
        last_user = next(
            (m["content"] for m in reversed(ctx) if m["role"] == "user"), ""
        )
        return f"Я не смог сгенерировать нормальный ответ. Повторю твой вопрос: {last_user}"
    except Exception as e:
        return f"Ошибка при обращении к LLM: {e}"


def _format_context(docs):
    lines = []
    for d in docs or []:
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
        lines.append(f"Source: {meta}\nContent: {content}")
    return "\n\n".join(lines)


def generate_rag_reply(
    question: str,
    summary: str | None,
    retrieved_docs,
    with_sources: bool = False,
) -> str:
    """
    Сгенерировать ответ с использованием RAG.

    Если with_sources=True, в промпт передаётся список источников (file_path, file_name, snippet),
    и LLM получает жёсткую инструкцию в конце ответа добавить раздел "Источники:"
    с путями к файлам.
    """
    context_block = _format_context(retrieved_docs)

    if not with_sources:
        combined = f"Вопрос: {question}\n\n" f"Контекст (из RAG):\n{context_block}"
        return generate_reply(
            [{"role": "user", "content": combined}],
            summary=summary,
        )

    # ---- 1. Собираем источники и дедуплим по file_path ----
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

    try:
        sources_json = json.dumps(sources, ensure_ascii=False, indent=2)
    except Exception:
        sources_json = "[]"

    prompt = f"""
Ты отвечаешь на вопрос пользователя на основе контекста и списка источников.

Тебе даны:
1) Вопрос.
2) Контекст с выдержками из документов.
3) Список источников в формате JSON.

ТВОЯ ЗАДАЧА:
- Дай связный, понятный ответ на вопрос на русском языке.
- Используй ТОЛЬКО приведённый контекст. Не выдумывай факты, которых нет.
- В КОНЦЕ ответа ОБЯЗАТЕЛЬНО добавь раздел "Источники:".
- В разделе "Источники:" выведи по одной строке на каждый источник из JSON.
- Формат каждой строки в разделе "Источники:":
 file_name -  краcочное пояснение
- НЕ выводи один и тот же file_path или file_name больше 1 раза.

Формат ответа:
1. Сначала нормальный текстовый ответ.
2. Потом пустая строка.
3. Затем строка "Источники:" и список источников по одному на строку.

Вопрос:
{question}

Контекст:
{context_block}

Список источников (JSON):
{sources_json}
""".strip()

    return generate_reply(
        [{"role": "user", "content": prompt}],
        summary=summary,
    )


def judge_rag_help(question: str, baseline: str, rag_answer: str) -> str:
    prompt = (
        "Определи, помог ли RAG улучшить ответ. "
        "Критерии: точность фактов, полнота, наличие ссылок на контекст. "
        "Верни короткий вывод в 2-3 предложениях.\n\n"
        f"Вопрос: {question}\n---\nБез RAG:\n{baseline}\n---\nС RAG:\n{rag_answer}"
    )
    return generate_reply([{"role": "user", "content": prompt}], summary=None)
