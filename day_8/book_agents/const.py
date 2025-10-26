# -*- coding: utf-8 -*-
import os

API_URL = os.getenv("API_URL", "https://api.openai.com/v1/responses")

WEAK_MODEL = os.getenv("WEAK_MODEL", "gpt-4o")
STRONG_MODEL = os.getenv("STRONG_MODEL", "gpt-4o")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "120000"))
SOFT_LIMIT_RATIO = float(os.getenv("SOFT_LIMIT_RATIO", "0.85"))

AGENT1_SCHEMA = {
    "type": "json_schema",
    "name": "BookFinderTurn",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message": {"type": "string"},
            "keywords": {"type": "string"},
            "books": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["title", "author", "reason"]
                }
            }
        },
        "required": ["message", "keywords", "books"]
    },
    "strict": True
}

AGENT1_SYSTEM = (
    "Ты — помощник по рекомендациям художественной литературы. "
    "Твоя задача: получить от пользователя ключевые слова/описание (жанры, настроение, тематика) "
    "и выдать список из ровно трёх подходящих художественных книг (романы, рассказы и т.п.). "
    "Всегда возвращай строгий JSON по схеме BookFinderTurn. "
    "Верни message='краткая вводная', keywords='нормализованные ключевые слова', "
    "и books: ровно три книги, каждая с title, author, reason. Не добавляй ничего вне JSON. "
    "Всегда возвращай какие-то 3 книги."
)

AGENT2_SYSTEM = (
    "Ты — опытный литературный редактор и книжный обозреватель. "
    "По списку художественных книг (название, автор, краткая причина выбора) "
    "сделай по каждой книге короткую аннотацию (2–4 предложения): "
    "о чём книга, ключевые темы/настроение, кому зайдёт. Пиши на языке входных данных."
)
