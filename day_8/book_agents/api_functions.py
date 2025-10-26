# -*- coding: utf-8 -*-
import json
from urllib import request, error
from typing import Dict, Any, List
import os
from .const import API_URL

def _post(payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        try:
            data = e.read().decode("utf-8")
        except Exception:
            data = ""
        raise RuntimeError(f"HTTPError {e.code}: {data or e.reason}")
    except error.URLError as e:
        raise RuntimeError(f"URLError: {e.reason}")

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

def _get_usage(resp_obj: Dict[str, Any]) -> Dict[str, int]:
    usage = resp_obj.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "reasoning_tokens": int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
    }

def rough_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)

def tokens_to_chars(budget_tokens: int) -> int:
    return max(1, budget_tokens * 4)

def estimate_messages_tokens(system_text: str, user_text: str) -> int:
    return rough_token_estimate(system_text) + rough_token_estimate(user_text) + 50
