from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI


# ── Pydantic схемы для структурированного ответа ───────────────────────────────


class Change(BaseModel):
    path: str
    content: str


class ChangeSet(BaseModel):
    change_notes: str
    changes: list[Change]


# ── Системный промпт ───────────────────────────────────────────────────────────

SYSTEM = (
    "Вы — помощник разработчика. На основе ТЗ и snapshot'а проекта предложите точечные изменения по всему проекту: "
    "включая новые файлы, правки существующих модулей, конфиги и документацию. "
    "Также сформируй подробный change_notes в формате markdown"
    "Соблюдайте текущий стек и стилистику кода. Верните строго структуру ChangeSet. "
    "НЕ добавляйте ничего вне заданной структуры."
)


# ── Публичные функции ──────────────────────────────────────────────────────────


def summarize_tor(client: OpenAI, model: str, text: str) -> str:
    """
    Краткая выжимка ТЗ (обычный текст). Можно оставить через responses.create.
    """
    res = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Суммаризатор: выдели цели, требования, ограничения в 8–12 маркеров на русском.",
            },
            {"role": "user", "content": text},
        ],
    )
    return res.output_text


def propose_changes_for_project(
    client: OpenAI, model: str, tor_summary: str, project_snapshot: str
) -> Dict[str, Any]:
    """
    Генерация изменений по всему проекту с использованием Structured Outputs:
    client.responses.parse(..., text_format=ChangeSet)
    """
    user = (
        f"Краткое ТЗ:\n{tor_summary}\n\n"
        f"{project_snapshot}\n\n"
        "Сформируй изменения. Помни:\n"
        "- Изменять можно любые текстовые файлы из snapshot; можно создавать новые.\n"
        "- Бинарные/крупные файлы игнорируем.\n"
        "- Если нужен новый README/CONFIG/gradle task — создай файл с полным содержимым.\n"
        "- Пиши минимум, необходимый для запуска/сборки (если требуется)."
    )

    # ВАЖНО: responses.parse требует совместимую модель (например, gpt-4.1 или gpt-4o-2024-08-06).
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=ChangeSet,  # <-- Pydantic класс
    )

    parsed: ChangeSet = resp.output_parsed  # уже валидированная структура
    # Преобразуем в обычный dict, чтобы код дальше ничего не ломал:
    return {
        "change_notes": parsed.change_notes,
        "changes": [c.model_dump() for c in parsed.changes],
    }
