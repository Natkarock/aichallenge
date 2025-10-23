#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses, json, os, sys, threading, time
from typing import Optional
from urllib import request, error

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o-2024-08-06"

# ─────────── JSON Schema для Structured Outputs ─────────── #
JSON_SCHEMA = {
    "type": "json_schema",
    "name": "MadHatterReply",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "persona": {
                "type": "string",
                "description": "Персона, всегда 'Безумный Шляпник'."
            },
            "joke": {
                "type": "string",
                "description": "Короткая, игривая шутка в стиле Безумного Шляпника (по-русски)."
            },
            "riddle": {
                "type": "string",
                "description": "Одна-две строки загадки в стиле Безумного Шляпника (по-русски)."
            },
            "answer": {
                "type": "string",
                "description": "Ясный и полезный ответ на вопрос пользователя (по-русски)."
            }
        },
        "required": ["persona", "joke", "riddle", "answer"]
    },
    "strict": True
}

# ─────────── Класс ответа ─────────── #
@dataclasses.dataclass
class MadHatterReply:
    persona: Optional[str]
    joke: str
    riddle: str
    answer: str

    @staticmethod
    def from_dict(d: dict) -> "MadHatterReply":
        return MadHatterReply(
            persona=d.get("persona"),
            joke=(d.get("joke") or "").strip(),
            riddle=(d.get("riddle") or "").strip(),
            answer=(d.get("answer") or "").strip(),
        )

    def pretty_print(self) -> str:
        lines = []
        if self.persona:
            lines.append(f"({self.persona})")
        lines += [f"Шутка: {self.joke}", f"Загадка: {self.riddle}", f"Ответ: {self.answer}"]
        return "\n".join(lines)

    def as_history_text(self) -> str:
        p = f"[{self.persona}] " if self.persona else ""
        return f"{p}Шутка: {self.joke}\nЗагадка: {self.riddle}\nОтвет: {self.answer}"

# ─────────── Временный спиннер ─────────── #
class Spinner:
    def __init__(self, text="Минутку, я думаю"):
        self.text = text
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while not self.stop_flag.is_set():
            sys.stdout.write(f"\r{frames[i % len(frames)]} {self.text}…")
            sys.stdout.flush(); i += 1; time.sleep(0.08)
    def start(self): self.thread.start()
    def stop_and_clear(self):
        self.stop_flag.set(); self.thread.join(timeout=1)
        sys.stdout.write("\r\033[2K"); sys.stdout.flush()

# ─────────── Вызов OpenAI API ─────────── #
def call_openai(messages) -> MadHatterReply:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")

    payload = {
        "model": MODEL,
        "input": messages,
        "text": {
            "format": JSON_SCHEMA,
            "verbosity": "medium"
        }
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(API_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except Exception:
        # скрываем подробности, выводим только “Попробуй ещё раз!”
        raise RuntimeError("Попробуй ещё раз!")

    # Извлекаем текст модели
    text = obj.get("output_text")
    if not text:
        chunks = []
        for item in obj.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        text = "".join(chunks).strip() if chunks else ""

    if not text:
        raise RuntimeError("Попробуй ещё раз!")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("Попробуй ещё раз!")

    return MadHatterReply.from_dict(data)

# ─────────── Главный цикл ─────────── #
def main():
    # 🎩 Фантазийное приветствие Шляпника
    print("🎩 Безумный Шляпник шепчет:\n«Добро пожаловать на чаепитие разума! Спроси меня что угодно — но берегись, ответ может быть загадкой…»\n")

    history = [{
        "role": "system",
        "content": (
            "Ты безумный шляпник из Алисы и говоришь загадками. "
            "В каждом ответе строго соблюдай структуру: сперва короткая шутка, "
            "затем загадка (1–2 строки), затем ясный ответ на вопрос пользователя."
        )
    }]

    try:
        while True:
            try:
                user = input("> ").strip()
            except EOFError:
                break

            if not user:
                continue
            if user.lower() in ("выход", "exit", "quit"):
                print("🎩 Безумный Шляпник машет шляпой: до скорых встреч!")
                break

            history.append({"role": "user", "content": user})
            spin = Spinner("Минутку, я думаю")
            spin.start()

            try:
                reply = call_openai(history)
            except Exception as e:
                spin.stop_and_clear()
                print(str(e))  # просто “Попробуй ещё раз!”
                continue

            spin.stop_and_clear()
            print(reply.pretty_print())
            history.append({"role": "assistant", "content": reply.as_history_text()})

    except KeyboardInterrupt:
        print("\n🎩 Безумный Шляпник исчезает в облаке чая! Пока!")

if __name__ == "__main__":
    main()
