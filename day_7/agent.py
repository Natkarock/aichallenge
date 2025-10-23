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
WEAK_MODEL = os.getenv("WEAK_MODEL", "gpt-4o-mini-2024-07-18")  # слабая
STRONG_MODEL = os.getenv("STRONG_MODEL", "gpt-4o")              # сильная

# ----------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------------------------------------
def _post(payload: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    """Отправка запроса в OpenAI Responses API."""
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
    """Достаёт текст из Responses API."""
    if resp_obj.get("output_text"):
        return resp_obj["output_text"].strip()
    chunks: List[str] = []
    for item in resp_obj.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                t = c.get("text")
                if isinstance(t, str):
                    chunks.append(t)
    return "".join(chunks).strip()

# ----------------------------------------------------------
# СПИННЕР для красивого ожидания
# ----------------------------------------------------------
class Spinner:
    def __init__(self, message="Минутку, думаю…"):
        self.message = message
        self.running = False
        self.thread = None

    def _animate(self):
        chars = "|/-\\"
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {chars[i % len(chars)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

# ----------------------------------------------------------
# AGENT 1 — слабая модель с structured outputs
# ----------------------------------------------------------
AGENT1_SCHEMA = {
    "type": "json_schema",
    "name": "QAOrAsk",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["ask", "answer"]},
            "message": {"type": "string"},
            "normalized_question": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["mode", "message", "normalized_question", "answer"]
    },
    "strict": True
}

AGENT1_SYSTEM = (
    "Ты — быстрый помощник. Определи, задал ли пользователь явный вопрос. "
    "Если вопроса нет — mode='ask', message='вежливо попроси уточнить вопрос', остальное пусто. "
    "Если вопрос есть — mode='answer', message='краткая вводная', normalized_question='вопрос', answer='короткий ответ'. "
    "Ответ всегда строго в JSON по схеме QAOrAsk."
)

class Agent1Weak:
    def __init__(self, model=WEAK_MODEL):
        self.model = model

    def run(self, user_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": AGENT1_SYSTEM},
                {"role": "user", "content": user_text}
            ],
            "text": {"format": AGENT1_SCHEMA},
            "temperature": 0.3
        }
        with Spinner("Agent1 думает…"):
            try:
                resp = _post(payload)
            except Exception:
                raise RuntimeError("Ошибка при обращении к Agent1.")
        txt = _extract_text(resp)
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            raise RuntimeError("Agent1 вернул некорректный JSON.")
        return {
            "mode": data.get("mode", ""),
            "message": data.get("message", ""),
            "question": data.get("normalized_question", ""),
            "answer": data.get("answer", "")
        }

# ----------------------------------------------------------
# AGENT 2 — сильный редактор
# ----------------------------------------------------------
AGENT2_SYSTEM = (
    "Ты — опытный редактор и технический писатель. "
    "Возьми ответ младшего ассистента и перепиши его в структурированном, "
    "логичном и подробном виде, сохранив смысл. Добавь ясное введение, пункты, примеры."
)

class Agent2StrongEditor:
    def __init__(self, model=STRONG_MODEL):
        self.model = model

    def improve(self, question: str, base_answer: str) -> str:
        user_content = (
            f"Вопрос пользователя:\n{question}\n\n"
            f"Ответ младшего ассистента:\n{base_answer}"
        )
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.5
        }
        with Spinner("Agent2 улучшает ответ…"):
            try:
                resp = _post(payload)
            except Exception:
                raise RuntimeError("Ошибка при обращении к Agent2.")
        txt = _extract_text(resp)
        return txt or "Agent2 не вернул ответ."

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------
def main():
    print("🤝 Двухагентная система (Agent1 → Agent2)\n")
    print("Введите вопрос. Если его нет, Agent1 попросит уточнить. Для выхода: 'выход'.\n")

    agent1 = Agent1Weak()
    agent2 = Agent2StrongEditor()

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("выход", "exit", "quit"):
            print("👋 Пока!")
            break

        # Agent1
        while True:
            try:
                result = agent1.run(user_input)
            except RuntimeError as e:
                print(e)
                break

            if result["mode"] == "ask":
                print(f"\n[Agent1 • уточнение] {result['message']}")
                user_input = input("> ").strip()
                if user_input.lower() in ("выход", "exit", "quit"):
                    print("👋 Пока!")
                    return
                continue

            if result["mode"] == "answer":
                print(f"\n[Agent1 • ответ]\nВопрос: {result['question']}\nОтвет: {result['answer']}\n")
                try:
                    improved = agent2.improve(result['question'], result['answer'])
                    print("\n[Agent2 • улучшенный ответ]\n")
                    print(improved.strip())
                    print("\n— — —\n")
                except RuntimeError as e:
                    print(e)
                break

if __name__ == "__main__":
    main()
