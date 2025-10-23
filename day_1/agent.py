
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, sys
from urllib import request, error

API_URL = "https://api.openai.com/v1/responses"
MODEL   = "gpt-4o-mini"   # при желании смените модель

def call_openai(messages):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Переменная окружения OPENAI_API_KEY не установлена.")

    payload = {
        "model": MODEL,
        # передаём ПОЛНУЮ историю диалога
        "input": messages
    }

    req = request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}")
    except error.URLError as e:
        raise RuntimeError(f"Network error: {e}")

    # 1) Пытаемся взять удобное поле
    txt = obj.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 2) Fallback: собрать из output[].content[].text при type output_text|text
    chunks = []
    for item in obj.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                chunks.append(c["text"])
    if chunks:
        return "".join(chunks).strip()

    # 3) Ещё один резерв
    if isinstance(obj.get("message"), dict) and isinstance(obj["message"].get("content"), str):
        return obj["message"]["content"].strip()

    return "[Не удалось разобрать текст ответа из content]"

def main():
    print("Чем я могу помочь?")
    history = [{"role": "system", "content": "Ты — дружелюбный и лаконичный помощник."}]

    try:
        while True:
            try:
                user = input("> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in ("выход", "exit", "quit"):
                print("Пока!")
                break

            history.append({"role": "user", "content": user})
            print("Минутку, я думаю")

            try:
                answer = call_openai(history)
            except Exception as e:
                print(f"[Ошибка] {e}")
                # не добавляем неудачный ответ в историю
                continue

            print(answer)
            history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt:
        print("\nПока!")

if __name__ == "__main__":
    main()
