#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from urllib import request
import sys, time, threading
from typing import Dict, Any

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o-mini-2024-07-18"  # лёгкая модель для заметной разницы

# ───────────── Простой спиннер ─────────────
class Spinner:
    def __init__(self, text="Минутку, думаю"):
        self.text = text
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{frames[i % len(frames)]} {self.text}…")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def start(self): self._t.start()
    def stop_and_clear(self):
        self._stop.set(); self._t.join(timeout=1)
        sys.stdout.write("\r\033[2K"); sys.stdout.flush()

# ───────────── Вызов API ─────────────
def ask_model(question: str) -> str:
    """Отправляет вопрос модели и возвращает её ответ в виде текста"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")

    payload = {
        "model": MODEL,
        "input": [{"role": "user", "content": question}],
        "temperature": 0.7
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with request.urlopen(req, timeout=90) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Ошибка сети или API: {e}")

    # достаём текст
    text = obj.get("output_text")
    if text:
        return text.strip()

    # fallback
    chunks = []
    for item in obj.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text","text") and isinstance(c.get("text"), str):
                chunks.append(c["text"])
    return "".join(chunks).strip()

# ───────────── Основной цикл ─────────────
def main():
    print("🧩 Эксперимент с рассуждениями (прямой ответ vs «решай пошагово»)")
    print("Введите логическую или арифметическую задачу (или 'выход' для завершения):\n")

    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit"):
            print("👋 Пока!")
            break

        # Прямой ответ
        spin = Spinner("Минутку, думаю")
        spin.start()
        try:
            answer_direct = ask_model(question)
        except Exception as e:
            spin.stop_and_clear()
            print(str(e))
            continue
        spin.stop_and_clear()

        print("\n" + "="*80)
        print("▶ Режим A: прямой ответ")
        print("="*80)
        print(answer_direct.strip() or "(пусто)")

        # Ответ с инструкцией "решай пошагово"
        spin = Spinner("Минутку, думаю")
        spin.start()
        try:
            answer_steps = ask_model(f"{question}\n\nИнструкция: решай пошагово.")
        except Exception as e:
            spin.stop_and_clear()
            print(str(e))
            continue
        spin.stop_and_clear()

        print("\n" + "="*80)
        print("▶ Режим B: с инструкцией «решай пошагово»")
        print("="*80)
        print(answer_steps.strip() or "(пусто)")
        print("\n")

if __name__ == "__main__":
    main()
