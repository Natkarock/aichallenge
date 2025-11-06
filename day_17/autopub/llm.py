from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

CHANGESET_SCHEMA = {
  "name": "changeset",
  "schema": {
    "type": "object",
    "properties": {
      "release_notes": {"type": "string"},
      "changes": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
          },
          "required": ["path","content"],
          "additionalProperties": False
        }
      }
    },
    "required": ["release_notes","changes"],
    "additionalProperties": False
  }
}

SYSTEM = (
    "Вы — помощник релиз‑инженера. На основе краткого ТЗ и выдержек из документации "
    "предложите минимально‑рисковые изменения в указанных файлах (targets) и релиз‑ноты.\n"
    "Возвращайте строго JSON по заданной схеме. Не добавляйте комментариев."
)

def summarize_tor(client: OpenAI, model: str, text: str) -> str:
    res = client.responses.create(
        model=model,
        input=[{"role":"system","content":"Суммаризатор: выдели цели, требования, ограничения в 8–12 маркеров на русском."},
               {"role":"user","content":text}],
    )
    return res.output_text

def propose_changes(client: OpenAI, model: str, tor_summary: str, docs_context: str, targets: List[str]) -> Dict:
    user = (
        f"Краткое ТЗ:\n{tor_summary}\n\n"
        f"Фрагменты документации:\n{docs_context}\n\n"
        f"Доступные для редактирования файлы (targets):\n" + "\n".join(targets) + "\n\n"
        "Сформируй изменения: перезапись целых файлов (не diff). Если файл отсутствует — создай. "
        "Сфокусируйся на авто‑заполнении релизной информации, инструкциях публикации, метаданных."
    )
    res = client.responses.create(
        model=model,
        response_format={"type":"json_schema","json_schema":CHANGESET_SCHEMA},
        input=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
    )
    out = res.output_text
    return json.loads(out)
