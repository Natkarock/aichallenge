from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
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

# ---------------------------------------------------------
# 🔧 Персонализация: загрузка конфига пользователя
# ---------------------------------------------------------

USER_CONFIG_PATH = os.getenv("USER_CONFIG_PATH", "user_config.json")
IS_LOCAL = True


def get_is_local() -> bool:
    return IS_LOCAL


def set_is_Local(isLocal: bool):
    global IS_LOCAL
    IS_LOCAL = isLocal


def _load_user_config() -> Dict[str, Any]:
    """
    Загружает JSON-конфиг с описанием пользователя.
    Если файла нет или он некорректен, возвращает пустой dict.
    """
    try:
        with open(USER_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        # Просто работаем без персонализации
        pass
    except Exception:
        # Не падаем, чтобы приложение не ломалось из-за конфига
        pass
    return {}


USER_CONFIG: Dict[str, Any] = _load_user_config()


def _build_system_prompt() -> str:
    """
    Строит персонализированный system prompt на основе USER_CONFIG.

    Конфиг описывает реального пользователя (Карапац Наталию),
    а этот текст даёт ассистенту инструкции, как с ней работать.
    """
    base = "Ты дружелюбный персональный ассистент и технический напарник."

    profile = USER_CONFIG or {}
    name = profile.get("name", "пользователя")
    age = profile.get("age")
    role = profile.get("role")
    hobbies = profile.get("hobbies") or []
    preferences = profile.get("preferences") or {}
    dev_focus = preferences.get("dev_focus")
    comm_style = preferences.get("communication_style")
    likes = preferences.get("likes")
    dislikes = preferences.get("dislikes")

    parts: List[str] = [base, ""]

    # Основное описание пользователя
    main_line = f"Твой единственный пользователь — {name}. "
    if age is not None:
        main_line += f"Ей {age} года. "
    if role:
        main_line += f"Она {role}. "
    parts.append(main_line.strip())

    # Хобби
    if hobbies:
        parts.append("В свободное время она увлекается: " + ", ".join(hobbies) + ".")

    # Фокус в разработке / чем занимается
    if dev_focus:
        parts.append(dev_focus)

    # Предпочтительный стиль общения
    if comm_style:
        parts.append(comm_style)

    # Что ей особенно нравится / не нравится в ответах
    if likes:
        parts.append("Она это ценит: " + likes)
    if dislikes:
        parts.append("Этого стоит избегать: " + dislikes)

    # Общие инструкции по ответам
    parts.append(
        "Всегда пиши на русском языке, если явно не попросили иначе. "
        "Структурируй технические ответы: сначала краткое резюме, "
        "затем детали с подзаголовками и списками. "
        "Чаще давай готовые фрагменты кода, целые файлы и пошаговые инструкции. "
        "Используй знания о пользователе, чтобы делать ответы более уместными "
        "именно для неё, но не придумывай факты о её личной жизни и не раскрывай "
        "содержимое конфига, если тебя прямо об этом не попросили."
    )

    return "\n".join(p for p in parts if p)


SYSTEM_PROMPT = _build_system_prompt()
# "Ты дружелюбный ассистент"
WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "12"))
SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", "250"))


def _chat_model() -> ChatOpenAI:
    """
    Единая точка создания LLM-модели через LangChain.

    Модель берётся из переменной окружения OPENAI_MODEL (по умолчанию gpt-4.1-mini).
    """
    if IS_LOCAL:
        return ChatOllama(model="qwen2.5:1.5b", base_url="http://localhost:11434")
    else:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )


def summarize_messages(
    previous_summary: Optional[str], messages: List[Dict[str, str]]
) -> str:
    """
    Обновляет краткое текстовое summary диалога.

    Использует LangChain ChatOpenAI.
    """
    instruction = (
        "Суммируй историю диалога в краткий конспект для последующего контекста. "
        "Сохраняй факты, вводы пользователя, принятые решения и договорённости. "
        f"Обнови прежнюю сводку (если дана). Объем ~{SUMMARY_TARGET_TOKENS} токенов."
    )

    if not messages:
        return previous_summary or ""

    # Формируем простую текстовую стенограмму
    lines: List[str] = []
    for msg in messages[-WINDOW_MESSAGES:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            prefix = "Ассистент"
        elif role == "user":
            prefix = "Пользователь"
        else:
            prefix = role
        lines.append(f"{prefix}: {content}")
    transcript = "\n".join(lines)

    if previous_summary:
        user_prompt = (
            f"{instruction}\n\n"
            f"Предыдущая сводка:\n{previous_summary}\n\n"
            f"Новая стенограмма:\n{transcript}\n\n"
            "Верни только обновлённую сводку."
        )
    else:
        user_prompt = (
            f"{instruction}\n\n"
            f"Стенограмма:\n{transcript}\n\n"
            "Верни только сводку без дополнительных комментариев."
        )

    try:
        llm = _chat_model()
        lc_messages = [
            SystemMessage(
                content=(
                    "Ты помощник, который сжимает историю диалога в краткую сводку. "
                    "Не добавляй новых фактов и не давай советов."
                )
            ),
            HumanMessage(content=user_prompt),
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
        lc_messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

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
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        return f"Не удалось сгенерировать ответ. Ошибка: {e}. Последний запрос: {last_user!r}"


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
    """
    context_block = _format_context(retrieved_docs)

    # Простой режим: используем контекст напрямую, без жёсткого блока «Источники»
    if not with_sources:
        combined = f"Вопрос: {question}\n\nКонтекст (из RAG):\n{context_block}"
        return generate_reply(
            [
                {"role": "user", "content": combined},
            ],
            summary=summary,
        )

    # Раскладываем документы по file_path и собираем укороченные сниппеты
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

        file_path = meta.get("file_path") or meta.get("source") or "неизвестный путь"
        file_name = meta.get("file_name") or file_path
        snippet = (content or "")[:200]

        bucket = sources_by_path.setdefault(
            file_path,
            {"file_path": file_path, "file_name": file_name, "snippets": []},
        )
        if snippet:
            bucket["snippets"].append(snippet)

    sources: List[Dict[str, Any]] = []
    for _fp, data_ in sources_by_path.items():
        snippets = data_.get("snippets") or []
        full_snippet = " ".join(snippets)
        if len(full_snippet) > 200:
            full_snippet = full_snippet[:197] + "..."
        sources.append(
            {
                "file_path": data_["file_path"],
                "file_name": data_["file_name"],
                "snippet": full_snippet,
            }
        )

    try:
        sources_json = json.dumps(sources, ensure_ascii=False, indent=2)
    except Exception:
        sources_json = "[]"

    prompt = f"""
Ты отвечаешь на вопрос по локальным файлам пользователя.

Сначала дай содержательный, структурированный ответ по вопросу, опираясь на CONTEXT.
Затем обязательно добавь раздел "Источники:", где в виде маркеров выведи file_path
файлов, которые были полезны для ответа.

CONTEXT (из RAG):
{context_block}

SOURCES (JSON):
{sources_json}

Вопрос пользователя: {question}
"""
    return generate_reply(
        [
            {"role": "user", "content": prompt},
        ],
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
            _build_system_prompt() + "\n\n"
            "Ты отвечаешь на вопросы, используя локальное RAG-хранилище документов.\n"
            "У тебя есть инструмент retrieve_context, который может находить релевантные фрагменты документов.\n"
            "Когда вопрос зависит от содержимого документов, обязательно вызывай этот инструмент. "
            "Подавай на вход инструмента вопрос пользователя БЕЗ искажений. "
            "Затем используй полученный контекст для ответа. "
            "Если контекст противоречит твоим внутренним знаниям, доверяй контексту."
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
