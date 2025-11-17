from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# Локальный RAG-хранилище (FAISS + метаданные)
# similarity_search уже реализован в rag_store и используется как backend для инструмента.
try:
    from rag_store import similarity_search as _similarity_search
except Exception:
    _similarity_search = (
        None  # чтобы модуль можно было импортировать даже без rag_store
    )

SYSTEM_PROMPT = "Ты дружелюбный помощник."
WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "12"))
SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", "250"))


def _chat_model() -> ChatOpenAI:
    """
    Единая точка создания LLM-модели через LangChain.

    Модель берётся из переменной окружения OPENAI_MODEL (по умолчанию gpt-4.1-mini).
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    # ChatOpenAI сам возьмёт OPENAI_API_KEY из окружения
    return ChatOpenAI(model=model_name, temperature=0.3)


def summarize_messages(
    previous_summary: Optional[str], messages: List[Dict[str, str]]
) -> str:
    """
    Обновляет краткое текстовое summary диалога.

    Использует LangChain ChatOpenAI вместо прямого вызова openai.responses.
    """
    instruction = (
        "Суммируй историю диалога в краткий конспект для последующего контекста. "
        "Сохраняй факты, вводы пользователя, принятые решения и договорённости. "
        f"Обнови прежнюю сводку (если дана). Объем ~{SUMMARY_TARGET_TOKENS} токенов."
    )
    if not messages:
        return previous_summary or ""

    try:
        last_n = WINDOW_MESSAGES
        ctx = messages[-last_n:]
        transcript_lines: List[str] = []
        for m in ctx:
            role = m.get("role")
            content = m.get("content", "")
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)
        prev = previous_summary or "(пусто)"

        llm = _chat_model()
        lc_messages = [
            SystemMessage(content="Ты лаконичный ассистент суммаризации."),
            HumanMessage(
                content=(
                    f"{instruction}\n\n"
                    f"Предыдущая сводка:\n{prev}\n\n"
                    f"Новая стенограмма:\n{transcript}\n\n"
                    "Верни только обновлённую сводку."
                )
            ),
        ]
        resp = llm.invoke(lc_messages)
        return (resp.content or "").strip()
    except Exception:
        # В случае любых проблем — просто вернём старое summary,
        # чтобы не ломать основную логику диалога.
        return previous_summary or ""


def generate_reply(messages: List[Dict[str, str]], summary: Optional[str]) -> str:
    """
    Базовый ответ ассистента БЕЗ RAG.

    Использует LangChain ChatOpenAI.
    """
    if not messages:
        return "Привет! О чём поговорим?"

    ctx = messages[-WINDOW_MESSAGES:]

    try:
        llm = _chat_model()
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if summary:
            lc_messages.append(
                SystemMessage(content=f"Краткая сводка контекста: {summary}")
            )

        for m in ctx:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                # всё остальное считаем пользовательским вводом
                lc_messages.append(HumanMessage(content=content))

        resp = llm.invoke(lc_messages)
        return (resp.content or "").strip()
    except Exception as e:
        # Поведение максимально похоже на старую реализацию:
        # возвращаем диагностическое сообщение вместо падения.
        last_user = next(
            (m.get("content", "") for m in reversed(ctx) if m.get("role") == "user"),
            "",
        )
        if last_user:
            return (
                "Я не смог сгенерировать нормальный ответ. "
                f"Повторю твой вопрос: {last_user}"
            )
        return f"Ошибка при обращении к LLM: {e}"


def _format_context(docs):
    """
    Преобразует список Document (или dict-подобных объектов) в текстовый блок
    для включения в prompt.
    """
    lines: List[str] = []
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
    summary: Optional[str],
    retrieved_docs,
    with_sources: bool = False,
) -> str:
    """
    Сгенерировать ответ с использованием RAG.

    Если with_sources=True, в промпт передаётся список источников (file_path, file_name, snippet),
    и LLM получает жёсткую инструкцию в конце ответа добавить раздел "Источники:"
    с путями к файлам.

    ВАЖНО: retrieved_docs сюда по-прежнему пробрасываются из main.py,
    чтобы UI мог отрисовать кликабельные источники. Фактический поиск
    уже может быть реализован и как LangChain tool (см. retrieve_context ниже).
    """
    context_block = _format_context(retrieved_docs)

    # Простой режим: используем контекст напрямую, без жёсткого блока «Источники»
    if not with_sources:
        combined = f"Вопрос: {question}\n\nКонтекст (из RAG):\n{context_block}"
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
    for _fp, data in sources_by_path.items():
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
 file_name - краcочное пояснение
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
    """
    Небольшая утилита, которая сравнивает ответы «с RAG» и «без RAG».
    """
    prompt = (
        "Определи, помог ли RAG улучшить ответ. "
        "Критерии: точность фактов, полнота, наличие ссылок на контекст. "
        "Верни короткий вывод в 2-3 предложениях.\n\n"
        f"Вопрос: {question}\n---\nБез RAG:\n{baseline}\n---\nС RAG:\n{rag_answer}"
    )
    return generate_reply([{"role": "user", "content": prompt}], summary=None)


# =============================================================================
# 🔧 LangChain tool для поиска контекста через similarity_search
# =============================================================================


@tool(response_format="content_and_artifact")
def retrieve_context(query: str, k: int = 3):
    """
    Retrieve information to help answer a query из RAG-хранилища.

    Под капотом вызывает rag_store.similarity_search, чтобы получить top-k документов.
    Возвращает:
      - content: сериализованный текст с метаданными и содержимым,
      - artifact: исходный список документов (для дальнейшей обработки агентом).
    """
    if _similarity_search is None:
        return "RAG-хранилище недоступно (similarity_search не инициализировано).", []

    retrieved_docs = _similarity_search(query, k=k)
    serialized = "\n\n".join(
        "Source: {meta}\nContent: {content}".format(
            meta=getattr(doc, "metadata", {}),
            content=getattr(doc, "page_content", ""),
        )
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


TOOLS = [retrieve_context]


def create_rag_agent(
    model: Optional[str] = None,
):
    """
    Вспомогательная фабрика для LangGraph-агента с инструментом retrieve_context.

    Как в weather_mcp: возвращаем runnable-граф, который принимает {"messages": [...]}.
    """
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    lc_model_id = f"openai:{model_name}"

    # Без system_prompt — его будем передавать через SystemMessage при вызове
    agent = create_agent(
        model=lc_model_id,
        tools=TOOLS,
    )
    return agent


def generate_agent_rag_reply(
    question: str,
    summary: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Генерирует ответ через LangGraph-агента, который сам вызывает инструмент retrieve_context.

    Использование:
        result = generate_agent_rag_reply("Как готовить плов?")
    """
    try:
        agent = create_rag_agent(model=model)

        base_system_prompt = (
            "Ты помощник, который отвечает на вопросы, используя RAG-хранилище.\n"
            "У тебя есть инструмент retrieve_context, который может находить релевантные фрагменты документов.\n"
            "Когда вопрос зависит от содержимого документов, обязательно вызывай этот инструмент. "
            "Подавай на вход инструмента вопрос пользователя БЕЗ искажений. "
            "Затем используй полученный контекст для ответа."
        )

        if summary:
            base_system_prompt += (
                "\n\nДополнительная сводка диалога для контекста:\n" + summary
            )

        messages = [
            SystemMessage(content=base_system_prompt),
            HumanMessage(content=question),
        ]

        result = agent.invoke({"messages": messages})

        # Для create_agent/CompiledStateGraph результат обычно:
        # {"messages": [SystemMessage, ..., AIMessage]}
        if isinstance(result, dict):
            if "messages" in result:
                msgs = result["messages"]
                if isinstance(msgs, list) and msgs:
                    # Ищем последний AIMessage
                    for m in reversed(msgs):
                        if isinstance(m, AIMessage):
                            return (m.content or "").strip()
                    # Если AIMessage не нашли — берём содержимое последнего
                    last = msgs[-1]
                    return getattr(last, "content", str(last))
            # На всякий случай — поддержка варианта с "output"
            if "output" in result:
                return str(result["output"])

        # Фоллбек — просто строковое представление
        return str(result)

    except Exception as e:
        return f"Ошибка агента: {e}"
