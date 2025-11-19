from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI


# ── Pydantic-схемы для структурированных ответов ───────────────────────────────


class Change(BaseModel):
    """
    Одно изменение файла.

    path    — относительный путь к файлу в репозитории
    content — полное итоговое содержимое файла (а не патч)
    """

    path: str
    content: str


class ChangeSet(BaseModel):
    """
    Набор изменений по проекту.

    change_notes — человекочитаемое описание изменений
    changes      — список файлов, которые нужно создать/обновить
    """

    change_notes: str
    changes: list[Change]


class TestTargets(BaseModel):
    """
    Список файлов, к которым LLM предлагает написать unit-тесты.
    """

    files: List[str]


class RelatedFilesForTests(BaseModel):
    """
    Список файлов, которые стоит использовать как дополнительный контекст
    при написании unit-тестов для одного целевого файла.
    """

    files: List[str]


# ── LLM-вызовы ──────────────────────────────────────────────────────────────────


def select_files_for_tests(
    client: OpenAI,
    model: str,
    project_snapshot: str,
) -> List[str]:
    """
    Выбор приоритетных файлов для написания unit-тестов.

    На вход подаём snapshot репозитория, на выход — список относительных путей.
    Не завязан ни на какой конкретный язык программирования.
    """
    SYSTEM = (
        "Ты помогаешь выбирать файлы исходного кода, для которых нужно написать "
        "unit-тесты. У тебя есть snapshot проекта с перечислением файлов и "
        "фрагментами их содержимого.\n\n"
        "Твоя задача — выбрать ограниченный список файлов (обычно 3–15), "
        "где тесты дадут наибольшую пользу: бизнес-логика, алгоритмы, "
        "критичные утилиты, сложные функции, некоторая инфраструктура.\n\n"
        "Проект может быть на любом языке (Python, Java, Kotlin, Swift, "
        "JavaScript/TypeScript, Go, C#, C++, и т.д.). Определи язык по расширению "
        "файла и стилю кода. Не ограничивайся только Python-файлами.\n\n"
        "Не выбирай уже существующие файлы с тестами (например, каталоги вроде "
        "tests/, src/test, или файлы с именами вроде *_test.ext, test_*.ext и т.п.).\n\n"
        "Отвечай строго в виде JSON, который соответствует схеме TestTargets."
    )

    user = (
        "Вот snapshot проекта. По нему определи, для каких файлов исходного кода "
        "нужно в первую очередь написать unit-тесты. "
        "В ответе верни только существующие относительные пути файлов из snapshot.\n\n"
        f"{project_snapshot}"
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=TestTargets,
    )

    parsed: TestTargets = resp.output_parsed
    # Нормализуем и удаляем дубликаты
    seen: set[str] = set()
    out: List[str] = []
    for p in parsed.files:
        p = p.strip()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def select_related_files_for_tests(
    client: OpenAI,
    model: str,
    project_snapshot: str,
    target_path: str,
    target_code: str,
    max_files: int = 8,
) -> List[str]:
    """
    Определяет, какие файлы (из snapshot) важны как контекст для написания
    unit-тестов к одному целевому файлу.

    Мы передаём:
    - полный snapshot проекта;
    - путь и содержимое целевого файла.

    На выходе — список относительных путей файлов, которые стоит подгрузить.
    Функция не привязана к конкретному языку.
    """
    SYSTEM = (
        "Ты опытный инженер, который пишет unit-тесты для проектов на разных языках. "
        "Тебе дают:\n"
        "1) snapshot всего репозитория с перечислением файлов и фрагментами кода;\n"
        "2) путь и содержимое одного целевого файла.\n\n"
        "Твоя задача — выбрать ограниченный список файлов из snapshot, которые стоит "
        "использовать как дополнительный контекст при генерации тестов для этого "
        "целевого файла. Это могут быть модули с вспомогательной логикой, "
        "интерфейсы, модели данных, конфигурация, инфраструктурный код и т.п.\n\n"
        "Выбирай только реально существующие файлы из snapshot и НЕ включай "
        "сам целевой файл в этот список. Отвечай строго структурой RelatedFilesForTests."
    )

    user = (
        "Вот целевой файл, для которого нужно написать unit-тесты:\n\n"
        f"Путь: {target_path}\n\n"
        "Содержимое:\n"
        "```text\n"
        f"{target_code}\n"
        "```\n\n"
        "Ниже — snapshot репозитория. Используя его, определи, какие файлы "
        "нужны как дополнительный контекст при написании тестов для целевого файла. "
        f"Верни не более {max_files} путей.\n\n"
        f"{project_snapshot}"
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=RelatedFilesForTests,
    )

    parsed: RelatedFilesForTests = resp.output_parsed
    seen: set[str] = set()
    out: List[str] = []
    for p in parsed.files:
        p = p.strip()
        if not p or p == target_path:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def generate_tests_for_file(
    client: OpenAI,
    model: str,
    target_path: str,
    target_code: str,
    related_files: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Генерация unit-тестов для одного файла.
    Возвращает ChangeSet-подобный dict: {change_notes, changes: [...]}.

    LLM может создавать/обновлять файлы с тестами (например, tests/test_*.ext,
    *_test.ext и т.п.), а также при необходимости аккуратно править код в
    самом целевом файле (но по возможности минимально).

    ВНИМАНИЕ: агент универсальный — не завязан на Python.
    Язык и стек тестирования нужно определять по расширению и содержимому
    файла (JUnit для Java, pytest/unittest для Python, Jest/Vitest для JS/TS,
    XCTest для Swift, NUnit/xUnit для .NET, и т.д.) или по уже существующим
    тестам в проекте.
    """
    related_files = related_files or {}

    context_parts: List[str] = []
    context_parts.append(
        "Целевой файл, для которого нужно написать unit-тесты:\n\n"
        f"Путь: {target_path}\n\n"
        "Содержимое:\n"
        "```text\n"
        f"{target_code}\n"
        "```"
    )

    if related_files:
        context_parts.append(
            "\n\nСвязанные файлы, которые можно использовать как контекст:\n"
        )
        for rel_path, code in related_files.items():
            context_parts.append(
                f"\n--- {rel_path} ---\n" "```text\n" f"{code}\n" "```"
            )

    PROJECT_CONTEXT = "\n".join(context_parts)

    SYSTEM = (
        "Ты опытный инженер по разработке и тестированию, который пишет "
        "качественные unit-тесты для проектов на разных языках программирования.\n\n"
        "Тебе дают один целевой файл и (опционально) несколько связанных файлов. "
        "Определи язык и тип проекта по расширению и стилю кода и выбери "
        "подходящий стек тестирования (например: JUnit/TestNG для Java; "
        "pytest/unittest для Python; Jest/Vitest/Mocha для JavaScript/TypeScript; "
        "XCTest для Swift; NUnit/xUnit/MSTest для .NET; и т.п.).\n\n"
        "Правила:\n"
        "1. Основная цель — покрыть ключевую бизнес-логику, ветвления, обработку "
        "ошибок и важные инварианты.\n"
        "2. Следуй уже существующему стилю тестов в проекте, если он заметен "
        "по связанным файлам или snapshot-у (каталоги tests/, src/test, и т.п.).\n"
        "3. Ты можешь:\n"
        "   • создавать новые файлы с тестами (например, tests/TestFoo.java, "
        "      tests/test_foo.py, src/test/..., и т.п.);\n"
        "   • дополнять существующие файлы с тестами;\n"
        "   • при необходимости вносить минимальные изменения в целевой файл "
        "      для улучшения тестируемости (dependency injection, выделение "
        "      вспомогательных методов и т.п.), но избегай агрессивных "
        "      рефакторингов.\n"
        "4. Все изменения проекта нужно вернуть в формате ChangeSet: "
        "каждый элемент changes должен содержать относительный путь файла и "
        "ИТОГОВОЕ содержимое этого файла целиком.\n"
        "5. Не добавляй в ответ ничего, кроме корректного JSON, который "
        "строго соответствует схеме ChangeSet."
    )

    user = (
        "Сгенерируй изменения в проекте, которые добавляют или улучшают unit-тесты "
        "для указанного целевого файла. Помни, что проект может быть на любом языке, "
        "поэтому сначала определи язык и подходящий стек тестирования по коду.\n\n"
        f"{PROJECT_CONTEXT}"
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=ChangeSet,
    )

    parsed: ChangeSet = resp.output_parsed
    return {
        "change_notes": parsed.change_notes,
        "changes": [c.model_dump() for c in parsed.changes],
    }
