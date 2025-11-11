
from __future__ import annotations

import os
from typing import List, Dict

SYSTEM_PROMPT = "Ты дружелюбный помощник."
WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "12"))
SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", "250"))

def summarize_messages(previous_summary: str | None, messages: List[Dict[str, str]]) -> str:
    instruction = (
        "Суммируй историю диалога в краткий конспект для последующего контекста. "
        "Сохраняй факты, вводы пользователя, принятые решения и договорённости. "
        f"Обнови прежнюю сводку (если дана). Объем ~{SUMMARY_TARGET_TOKENS} токенов."
    )
    prev = previous_summary or ""
    transcript = "\n".join([f"{m.get('role')}: {m.get('content','')}" for m in messages])

    try:
        import importlib.util
        if importlib.util.find_spec("langchain_openai") is not None:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                             temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")))
            msgs = [
                SystemMessage(content="Ты лаконичный ассистент суммаризации."),
                HumanMessage(content=f"{instruction}\n\nПредыдущая сводка:\n{prev}\n\nНовая стенограмма:\n{transcript}\n\nВерни только обновлённую сводку.")
            ]
            return llm.invoke(msgs).content.strip()
        else:
            from openai import OpenAI
            client = OpenAI()
            model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            prompt = (
                f"system: Ты лаконичный ассистент суммаризации.\n"
                f"user: {instruction}\n\nПредыдущая сводка:\n{prev}\n\n"
                f"Новая стенограмма:\n{transcript}\n\nВерни только обновлённую сводку."
            )
            resp = client.responses.create(model=model, input=prompt)
            for item in resp.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            return c.text.strip()
            return prev or ""
    except Exception:
        return previous_summary or ""


def generate_reply(history: List[Dict[str, str]], summary: str | None = None) -> str:
    try:
        import importlib.util
        ctx = history[-WINDOW_MESSAGES:] if len(history) > WINDOW_MESSAGES else history

        if importlib.util.find_spec("langchain_openai") is not None:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

            model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
            llm = ChatOpenAI(model=model, temperature=temperature)

            msgs = [SystemMessage(content=SYSTEM_PROMPT)]
            if summary:
                msgs.append(SystemMessage(content=f"Краткая сводка контекста: {summary}"))
            for m in ctx:
                role = m.get("role"); content = m.get("content", "")
                msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
            return llm.invoke(msgs).content

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
        last_user = next((m["content"] for m in reversed(ctx) if m.get("role")=="user"), "")
        return f"(fallback) Вы сказали: {last_user}"
    except Exception as e:
        last_user = next((m["content"] for m in reversed(history) if m.get("role")=="user"), "")
        return f"Ошибка LLM: {e}\n(fallback) Вы сказали: {last_user}"


def _format_context(docs):
    lines = []
    for d in docs or []:
        try:
            meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else d.get("metadata", {})
            content = getattr(d, "page_content", "") if hasattr(d, "page_content") else d.get("page_content", "")
        except Exception:
            meta, content = {}, ""
        lines.append(f"Source: {meta}\nContent: {content}")
    return "\n\n".join(lines)

def generate_rag_reply(question: str, summary: str | None, retrieved_docs) -> str:
    combined = f"Вопрос: {question}\n\nКонтекст (из RAG):\n{_format_context(retrieved_docs)}"
    return generate_reply([{"role":"user","content": combined}], summary=summary)

def judge_rag_help(question: str, baseline: str, rag_answer: str) -> str:
    prompt = (
        "Определи, помог ли RAG улучшить ответ. "
        "Критерии: точность фактов, полнота, наличие ссылок на контекст. "
        "Верни короткий вывод в 2-3 предложениях.\n\n"
        f"Вопрос: {question}\n---\nБез RAG:\n{baseline}\n---\nС RAG:\n{rag_answer}"
    )
    return generate_reply([{"role":"user","content": prompt}], summary=None)
