#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses, json, os, sys, time, threading
from typing import Optional
from urllib import request, error

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o-2024-08-06"

# ─────────── JSON Schema одного хода ассистента ─────────── #
TURN_SCHEMA = {
    "type": "json_schema",
    "name": "InteriorDesignerTurn",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["ask", "final"],
                "description": "ask — дизайнер задаёт 1–2 вопроса; final — выдаёт итоговый бриф."
            },
            "message": {
                "type": "string",
                "description": "Текст пользователю. Если ask — 1–2 вопроса; если final — краткое вступление перед брифом."
            },
            "brief": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "client_name": {"type": "string"},
                    "apartment_area": {"type": "string"},
                    "rooms": {"type": "string"},
                    "style": {"type": "string"},
                    "color_palette": {"type": "string"},
                    "materials": {"type": "string"},
                    "furniture": {"type": "string"},
                    "lighting": {"type": "string"},
                    "budget": {"type": "string"},
                    "special_requests": {"type": "string"},
                    "final_design_brief": {"type": "string"}
                },
                "required": [
                    "client_name",
                    "apartment_area",
                    "rooms",
                    "style",
                    "color_palette",
                    "materials",
                    "furniture",
                    "lighting",
                    "budget",
                    "special_requests",
                    "final_design_brief"
                ]
            }
        },
        "required": ["mode", "message", "brief"]
    },
    "strict": True
}

# ─────────── Data classes ─────────── #
@dataclasses.dataclass
class DesignBrief:
    client_name: str
    apartment_area: str
    rooms: str
    style: str
    color_palette: str
    materials: str
    furniture: str
    lighting: str
    budget: str
    special_requests: str
    final_design_brief: str

    @staticmethod
    def from_dict(d: dict) -> "DesignBrief":
        return DesignBrief(
            client_name=d.get("client_name", ""),
            apartment_area=d.get("apartment_area", ""),
            rooms=d.get("rooms", ""),
            style=d.get("style", ""),
            color_palette=d.get("color_palette", ""),
            materials=d.get("materials", ""),
            furniture=d.get("furniture", ""),
            lighting=d.get("lighting", ""),
            budget=d.get("budget", ""),
            special_requests=d.get("special_requests", "нет"),
            final_design_brief=d.get("final_design_brief", "")
        )

    def pretty_print(self) -> None:
        print("\n📐 Итоговое техническое задание (дизайн-бриф):\n")
        rows = [
            ("Имя клиента", self.client_name),
            ("Площадь", self.apartment_area),
            ("Комнаты и назначение", self.rooms),
            ("Стиль", self.style),
            ("Цветовая палитра", self.color_palette),
            ("Материалы и фактуры", self.materials),
            ("Мебель и декор", self.furniture),
            ("Освещение", self.lighting),
            ("Бюджет", self.budget),
            ("Особые пожелания", self.special_requests or "нет"),
        ]
        for k, v in rows:
            print(f"{k}: {v}")
        print("\n— — —\n")
        print(self.final_design_brief)
        print()

@dataclasses.dataclass
class DesignerTurn:
    mode: str
    message: str
    brief: Optional[DesignBrief]

    @staticmethod
    def from_dict(d: dict) -> "DesignerTurn":
        b = d.get("brief")
        return DesignerTurn(
            mode=d.get("mode", "ask"),
            message=(d.get("message") or "").strip(),
            brief=DesignBrief.from_dict(b) if isinstance(b, dict) else None
        )

# ─────────── Спиннер ─────────── #
class Spinner:
    def __init__(self, text="Минутку, подбираю идеи"):
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

# ─────────── Вызов API ─────────── #
def call_openai(messages) -> DesignerTurn:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")

    payload = {
        "model": MODEL,
        "input": messages,
        "text": {
            "format": TURN_SCHEMA,
            "verbosity": "medium"
        }
    }
    req = request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST"
    )
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with request.urlopen(req, timeout=90) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except Exception:
        raise RuntimeError("Попробуй ещё раз!")

    text = obj.get("output_text")
    if not text:
        chunks = []
        for item in obj.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text","text") and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        text = "".join(chunks).strip() if chunks else ""

    if not text:
        raise RuntimeError("Попробуй ещё раз!")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("Попробуй ещё раз!")

    return DesignerTurn.from_dict(data)

# ─────────── Главный цикл ─────────── #
def main():
    system_prompt = (
        "Ты — профессиональный дизайнер интерьеров с большим опытом в жилых проектах. "
        "Ты проводишь консультацию и создаёшь детальное Техническое Задание (ТЗ) "
        "на дизайн-проект квартиры.\n\n"
        "ФОРМАТ ВСЕГДА: JSON по схеме TURN_SCHEMA — {mode, message, brief}. "
        "brief обязателен на каждом ходу. В режиме 'ask' он может быть частично заполнен. "
        "В режиме 'final' — полностью и детально.\n\n"
        "Если клиент не отвечает или отвечает неопределённо, повтори свой вопрос в другой формулировке — "
        "вежливо, без раздражения. Например: «Хочу уточнить ещё раз…», «Кажется, мы это не обсудили…». "
        "Продолжай задавать этот вопрос, пока не получишь ясный ответ.\n\n"
        "Собери у клиента: площадь, комнаты, стиль, цвет, материалы, мебель, освещение, бюджет, сроки, особые запросы. "
        "Если особых пожеланий нет, ставь строку 'нет' в special_requests.\n\n"
        "Когда информации достаточно, выдай mode='final' с полностью заполненным brief и подробным 'final_design_brief', "
        "где ты описываешь не только ответы клиента, но и своё профессиональное видение: "
        "концепцию, зонирование, цвет и материалы, сценарии освещения, мебель и хранение, "
        "технические допущения и этапность реализации. "
        "Пиши как проектную записку, без поэзии и воды. Вне JSON не добавляй комментариев."
    )

    print("🎨 Дизайнер интерьеров:\n"
          "«Здравствуйте! Я помогу создать интерьер вашей мечты. "
          "Расскажите, пожалуйста, о вашей квартире и ожиданиях — начнём с основных пожеланий.»\n")

    history = [{"role": "system", "content": system_prompt}]

    # ждём первый ответ пользователя — только потом делаем запрос
    while True:
        try:
            first = input("> ").strip()
        except EOFError:
            return
        if first.lower() in ("выход", "exit", "quit"):
            print("🪞 Дизайнер интерьеров кивает: «До встречи!»")
            return
        if first:
            history.append({"role": "user", "content": first})
            break

    assistant_ask_turns = 0
    MAX_ASK_TURNS = 10

    try:
        while True:
            spin = Spinner()
            spin.start()
            try:
                turn = call_openai(history)
            except Exception as e:
                spin.stop_and_clear()
                print(str(e))
                user = input("> ").strip()
                if user.lower() in ("выход","exit","quit"):
                    print("🪞 Дизайнер интерьеров кивает: «До встречи!»")
                    break
                if user:
                    history.append({"role": "user", "content": user})
                continue
            spin.stop_and_clear()

            print(turn.message or "(сообщение)")
            history.append({"role": "assistant", "content": turn.message})

            if turn.mode == "final":
                if turn.brief:
                    turn.brief.pretty_print()
                else:
                    print("Попробуй ещё раз!")
                break

            if turn.mode == "ask":
                assistant_ask_turns += 1
                user = input("> ").strip()
                if not user:
                    # если клиент ничего не сказал — добавляем "нет ответа", чтобы модель переспрашивала
                    history.append({"role": "user", "content": "(пользователь не ответил)"})
                elif user.lower() in ("выход","exit","quit"):
                    print("🪞 Дизайнер интерьеров кивает: «До встречи!»")
                    break
                else:
                    history.append({"role": "user", "content": user})

                if assistant_ask_turns >= MAX_ASK_TURNS:
                    history.append({
                        "role": "system",
                        "content": (
                            "Больше уточнений не требуется. "
                            "Сформируй следующий ход обязательно с mode='final' и полностью заполни brief."
                        )
                    })
    except KeyboardInterrupt:
        print("\n🪞 Дизайнер интерьеров прощается. Пока!")

if __name__ == "__main__":
    main()
