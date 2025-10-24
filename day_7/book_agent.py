#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import threading
from urllib import request
from typing import Dict, Any, List

API_URL = "https://api.openai.com/v1/responses"

# ----------------------------------------------------------
# МОДЕЛИ
# ----------------------------------------------------------
WEAK_MODEL = os.getenv("WEAK_MODEL", "gpt-4o")  # слабая модель для Агента 1
STRONG_MODEL = os.getenv("STRONG_MODEL", "gpt-4o")              # сильная модель для Агента 2

# ----------------------------------------------------------
# СПИННЕР
# ----------------------------------------------------------
class Spinner:
    def __init__(self, message="Минутку, думаю…"):
        self.message = message
        self.running = False
        self.thread = None

    def _animate(self):
        frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while self.running:
            sys.stdout.write(f"\r{frames[i % len(frames)]} {self.message}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)
        # очистка строки
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

# ----------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ API
# ----------------------------------------------------------
def _post(payload: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _extract_text(resp_obj: Dict[str, Any]) -> str:
    text = resp_obj.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks: List[str] = []
    for item in resp_obj.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                t = c.get("text")
                if isinstance(t, str):
                    chunks.append(t)
    return "".join(chunks).strip()

# ----------------------------------------------------------
# АГЕНТ 1 — подбирает 3 книги по ключевым словам/описанию (Structured Outputs)
# ----------------------------------------------------------
AGENT1_SCHEMA = {
    "type": "json_schema",
    "name": "BookFinderTurn",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message": {
                "type": "string",
                "description": "Короткая реплика ассистента для пользователя."
            },
            "keywords": {
                "type": "string",
                "description": "Нормализованные ключевые слова/описание пользователя"
            },
            "books": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "reason": {"type": "string", "description": "почему эта книга подходит под ключевые слова"}
                    },
                    "required": ["title", "author", "reason"]
                },
                "description": "Ровно три художественные книги, подходящие под запрос."
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
    "и books: ровно три книги, каждая с title, author, reason. Не добавляй ничего вне JSON.Всегда возвращй какие-то 3 книги."
)

class Agent1BookFinder:
    def __init__(self, model: str = WEAK_MODEL):
        self.model = model

    def run(self, user_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": AGENT1_SYSTEM},
                {"role": "user", "content": user_text}
            ],
            "text": {"format": AGENT1_SCHEMA, "verbosity": "medium"},
            "temperature": 0.5
        }
        with Spinner("Agent1 подбирает книги…"):
            try:
                resp = _post(payload)
            except Exception:
                raise RuntimeError("Попробуй ещё раз! (Agent1)")
        txt = _extract_text(resp)
        if not txt:
            raise RuntimeError("Попробуй ещё раз! (пустой ответ Agent1)")
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            raise RuntimeError("Agent1 вернул некорректный JSON.")
        # нормализуем выход
        return {
            "mode": data.get("mode", ""),
            "message": data.get("message", ""),
            "keywords": data.get("keywords", ""),
            "books": data.get("books", []) or []
        }

# ----------------------------------------------------------
# АГЕНТ 2 — принимает список книг и делает краткие описания/аннотации
# ----------------------------------------------------------
AGENT2_SYSTEM = (
    "Ты — опытный литературный редактор и книжный обозреватель. "
    "По списку художественных книг (название, автор, краткая причина выбора) "
    "сделай по каждой книге внятную, короткую аннотацию (2–4 предложения): "
    "о чём книга, ключевые темы/настроение, кому зайдёт. Пиши на языке входных данных."
)

class Agent2BookSummarizer:
    def __init__(self, model: str = STRONG_MODEL):
        self.model = model

    def improve_list(self, keywords: str, books: List[Dict[str, str]]) -> str:
        lines = []
        lines.append("Ключевые слова пользователя:")
        lines.append(keywords.strip() or "(не указаны)")
        lines.append("\nСписок книг для аннотаций:")
        for i, b in enumerate(books, 1):
            title = b.get("title", "").strip()
            author = b.get("author", "").strip()
            reason = b.get("reason", "").strip()
            lines.append(f"{i}. {title} — {author}. Причина выбора: {reason}")
        user_content = "\n".join(lines)

        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.6
        }
        with Spinner("Agent2 пишет аннотации…"):
            try:
                resp = _post(payload)
            except Exception:
                raise RuntimeError("Попробуй ещё раз! (Agent2)")
        txt = _extract_text(resp)
        return txt or "Agent2 не вернул текст аннотаций."

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print("📚 Два агента: подбор книг (Agent1) → аннотации (Agent2)\n")
    print("Опишите, что хотите почитать (жанры, настроение, тематика) или введите 'выход' для завершения.\n")

    agent1 = Agent1BookFinder()
    agent2 = Agent2BookSummarizer()

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("выход", "exit", "quit"):
            print("👋 Пока!")
            break

        # Цикл: пока Agent1 не отдаст список книг
        while True:
            result = agent1.run(user_input)
            message = result["message"]
            keywords = result["keywords"]
            books = result["books"]
            print(f"\n[Agent1 • подбор] {message}")
            if keywords:
                print(f"Ключевые слова: {keywords}")
            print("\nТри подходящие книги:")
            for i, b in enumerate(books, 1):
                print(f"{i}. {b.get('title','')} — {b.get('author','')}")
            # Agent2 создаёт аннотации
            try:
                annotations = agent2.improve_list(keywords, books)
            except RuntimeError as e:
                print(str(e))
                break
            print("\n[Agent2 • аннотации]\n")
            print(annotations.strip())
            print("\n— — —\n")
            print("Хотите еще найти книги по ключевым словам?")
            break
            # на случай неожиданных состояний
            print("Попробуй ещё раз!")
            break

if __name__ == "__main__":
    main()

