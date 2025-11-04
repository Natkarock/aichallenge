#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os
from typing import TypedDict, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------

SUMMARY_EVERY = 10
MODEL_NAME = "gpt-4o-mini"

QUESTIONS = [
    "Хочу спроектировать умного агента на Python. С чего начать, если цель — сделать систему, которая может вести диалог и вызывать внешние инструменты?",
    "Какие основные архитектурные компоненты стоит предусмотреть для такого агента?",
    "Как лучше организовать поток данных между этими компонентами, чтобы код был расширяемым и модульным?",
    "Что посоветуешь использовать для хранения краткосрочной и долгосрочной памяти диалога?",
    "Как бы ты реализовал интерфейс между агентом и внешними API, например погодным сервисом?",
    "В чём преимущества LangGraph по сравнению с обычными цепочками LangChain?",
    "Какой минимальный набор узлов сделать в графе агента, чтобы показать базовый сценарий общения?",
    "Можешь привести пример условного перехода в LangGraph — например, если вопрос о погоде?",
    "Как добавить в граф память, чтобы агент опирался на предыдущие ответы?",
    "Когда стоит применять саммаризацию сообщений, а когда достаточно обычного контекстного окна?",
    "Как реализовать постепенное отображение ответа в терминале, чтобы пользователь видел ход генерации?",
    "Как посчитать и логировать расход токенов по каждому шагу?",
    "Какие есть способы оптимизации количества токенов без потери качества?",
    "Можно ли комбинировать саммаризацию и RAG для улучшения памяти агента?",
    "Как структурировать SystemMessage, чтобы агент придерживался нужного стиля?",
    "Ты упоминал, что саммаризация помогает экономить токены. Объясни, как именно.",
    "Как бы ты расширил механизм краткосрочной памяти, чтобы агент понимал цели пользователя?",
    "Покажи, как добавить новый узел в уже существующий граф из предыдущего примера.",
    "Если взять пример с погодным API, как интегрировать туда RAG или базу знаний?",
    "Исходя из всего, составь короткий план создания прототипа диалогового агента с памятью и саммаризацией.",
]

# ---------------------------------------------------------------
# Типы состояния
# ---------------------------------------------------------------


class TokenUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GraphState(TypedDict, total=False):
    messages: List[BaseMessage]
    summary: str
    tokens: TokenUsage


# ---------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------

console = Console()


def _add_usage(dst: TokenUsage, add: Dict[str, int]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        if k in add:
            dst[k] = dst.get(k, 0) + int(add[k])


def _normalize_usage(meta: Dict[str, Any]) -> TokenUsage:
    u = meta.get("token_usage", {}) or meta.get("usage", {}) or {}
    p = u.get("prompt_tokens", u.get("input_tokens", 0))
    c = u.get("completion_tokens", u.get("output_tokens", 0))
    return TokenUsage(input_tokens=p, output_tokens=c, total_tokens=p + c)


# ---------------------------------------------------------------
# Узлы графа (одиночные)
# ---------------------------------------------------------------


def build_llm(model_name: str):
    return ChatOpenAI(
        model=model_name, temperature=0.3, max_retries=2, request_timeout=60
    )


def make_respond_graph(llm):
    """Граф для одного шага ответа."""
    sg = StateGraph(GraphState)

    def node_respond(state: GraphState) -> GraphState:
        messages = []
        messages.append(SystemMessage(content="Коротко отвечай на вопросы"))
        if state.get("summary"):
            messages.append(
                SystemMessage(
                    content=f"Краткое резюме прошлой беседы:\n{state['summary']}"
                )
            )
        messages += state.get("messages", [])
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as p:
            t = p.add_task(description="Генерирую ответ…", total=None)
            ai: AIMessage = llm.invoke(messages)
            p.remove_task(t)
        usage = _normalize_usage(ai.response_metadata or {})
        new_tokens = state.get("tokens", TokenUsage())
        _add_usage(new_tokens, usage)
        state["messages"].append(ai)
        state["tokens"] = new_tokens
        return state

    sg.add_node("respond", node_respond)
    sg.set_entry_point("respond")
    sg.add_edge("respond", END)
    return sg.compile()


def make_summarize_graph(llm):
    """Граф для шага суммаризации."""
    sg = StateGraph(GraphState)

    def node_summarize(state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state
        prompt = [
            SystemMessage(content="Ты кратко суммируешь суть диалога в 5–8 пунктов."),
            HumanMessage(
                content="\n".join(f"{m.type.upper()}: {m.content}" for m in messages)
            ),
        ]
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as p:
            t = p.add_task(description="Сжимаю историю…", total=None)
            ai: AIMessage = llm.invoke(prompt)
            p.remove_task(t)
        usage = _normalize_usage(ai.response_metadata or {})
        new_tokens = state.get("tokens", TokenUsage())
        _add_usage(new_tokens, usage)
        state["summary"] = ai.content
        state["messages"] = messages[-1:]
        state["tokens"] = new_tokens
        return state

    sg.add_node("summarize", node_summarize)
    sg.set_entry_point("summarize")
    sg.add_edge("summarize", END)
    return sg.compile()


# ---------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------


def run_demo(summarize_enabled: bool, title: str):
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    llm = build_llm(MODEL_NAME)
    respond_app = make_respond_graph(llm)
    summarize_app = make_summarize_graph(llm)

    state: GraphState = {
        "messages": [SystemMessage(content="Ты краток, точен, отвечай на русском.")],
        "summary": "",
        "tokens": TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    }

    per_turn_usage = []

    for i, q in enumerate(QUESTIONS, start=1):
        console.print(
            Panel.fit(f"[bold]Q{i}:[/bold] {q}", title="Вопрос", border_style="yellow")
        )
        state["messages"].append(HumanMessage(content=q))
        before = dict(state["tokens"])

        try:
            state = respond_app.invoke(state)
        except KeyboardInterrupt:
            console.print("[red]Остановлено пользователем[/red]")
            break

        after = state["tokens"]
        turn_usage = TokenUsage(
            input_tokens=after.get("input_tokens", 0) - before.get("input_tokens", 0),
            output_tokens=after.get("output_tokens", 0)
            - before.get("output_tokens", 0),
            total_tokens=after.get("total_tokens", 0) - before.get("total_tokens", 0),
        )
        per_turn_usage.append(turn_usage)

        ai = state["messages"][-1]
        console.print(Panel(ai.content, title="Ответ", border_style="green"))

        # Саммаризация каждые N вопросов
        if summarize_enabled and i % SUMMARY_EVERY == 0:
            state = summarize_app.invoke(state)
            console.print(
                Panel(
                    state["summary"],
                    title=f"Сжатие истории (после Q{i})",
                    border_style="magenta",
                )
            )

    # статистика
    table = Table(title=f"Итоги: {title}", box=box.SIMPLE, show_lines=True)
    table.add_column("Поворот", justify="right")
    table.add_column("prompt", justify="right")
    table.add_column("completion", justify="right")
    table.add_column("total", justify="right")
    for idx, u in enumerate(per_turn_usage, start=1):
        table.add_row(
            str(idx),
            str(u.get("input_tokens", 0)),
            str(u.get("output_tokens", 0)),
            str(u.get("total_tokens", 0)),
        )
    console.print(table)

    totals = state["tokens"]
    console.print(
        Panel.fit(
            f"[bold]ИТОГО токенов[/bold]\n"
            f"prompt: {totals.get('input_tokens', 0)}\n"
            f"completion: {totals.get('output_tokens', 0)}\n"
            f"total: {totals.get('total_tokens', 0)}",
            border_style="cyan",
            title="Статистика",
        )
    )
    return {"totals": totals}


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def main():
    global SUMMARY_EVERY, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Демо краткосрочной памяти с саммаризацией."
    )
    parser.add_argument(
        "--mode", choices=["both", "summarize", "nosummary"], default="both"
    )
    parser.add_argument("--every", type=int, default=SUMMARY_EVERY)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    SUMMARY_EVERY = max(2, args.every)
    MODEL_NAME = args.model

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY не задан[/red]")
        raise SystemExit(1)

    if args.mode in ("both", "nosummary"):
        res_no = run_demo(False, "Режим БЕЗ сжатия")
    if args.mode in ("both", "summarize"):
        res_sm = run_demo(True, "Режим СО сжатием")

    if args.mode == "both":
        table = Table(title="Сравнение расхода токенов", box=box.SIMPLE_HEAVY)
        table.add_column("Режим")
        table.add_column("prompt")
        table.add_column("completion")
        table.add_column("total")
        table.add_row(
            "Без сжатия",
            str(res_no["totals"].get("input_tokens", 0)),
            str(res_no["totals"].get("output_tokens", 0)),
            str(res_no["totals"].get("total_tokens", 0)),
        )
        table.add_row(
            "Со сжатием",
            str(res_sm["totals"].get("input_tokens", 0)),
            str(res_sm["totals"].get("output_tokens", 0)),
            str(res_sm["totals"].get("total_tokens", 0)),
        )
        console.rule("[bold]Сравнение[/bold]")
        console.print(table)


if __name__ == "__main__":
    main()
