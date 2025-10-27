# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
import json
from .const import (
    WEAK_MODEL, STRONG_MODEL, SUMMARIZER_MODEL,
    AGENT1_SYSTEM, AGENT2_SYSTEM, AGENT1_SCHEMA,
    MAX_CONTEXT_TOKENS, SOFT_LIMIT_RATIO
)
from .api_functions import _post, _extract_text, _get_usage, rough_token_estimate, tokens_to_chars, estimate_messages_tokens, build_openlibrary_mcp_tool

class Agent1BookFinderGUI:
    def __init__(self, model: str = WEAK_MODEL):
        self.model = model
    def run(self, user_text: str) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        payload = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': AGENT1_SYSTEM},
                {'role': 'user', 'content': user_text}
            ],
            'text': {'format': AGENT1_SCHEMA, 'verbosity': 'medium'},
            'temperature': 0.5
        }
        resp = _post(payload)
        usage = _get_usage(resp)
        txt = _extract_text(resp)
        if not txt:
            raise RuntimeError('Попробуй ещё раз! (пустой ответ Agent1)')
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            raise RuntimeError('Agent1 вернул некорректный JSON.')
        normalized = {
            'message': data.get('message', ''),
            'keywords': data.get('keywords', ''),
            'books': data.get('books', []) or []
        }
        return normalized, usage, resp

class Agent2BookSummarizerGUI:
    def __init__(self, model: str = STRONG_MODEL):
        self.model = model

    def improve_list(self, keywords: str, books: List[Dict[str, str]]) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        lines = []
        lines.append('Ключевые слова пользователя:')
        lines.append(keywords.strip() or '(не указаны)')
        lines.append('\nСписок книг для аннотаций:')
        for i, b in enumerate(books, 1):
            title = b.get('title', '').strip()
            author = b.get('author', '').strip()
            reason = b.get('reason', '').strip()
            lines.append(f"{i}. {title} — {author}. Причина выбора: {reason}")
        user_content = '\n'.join(lines)
        payload = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': AGENT2_SYSTEM},
                {'role': 'user', 'content': user_content}
            ],
            'temperature': 0.6
        }
        resp = _post(payload)
        usage = _get_usage(resp)
        txt = _extract_text(resp) or 'Agent2 не вернул текст аннотаций.'
        return txt.strip(), usage, resp

    def _fetch_covers_via_mcp(self, books: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        schema = {
            'type': 'json_schema',
            'name': 'CoversResult',
            'schema': {
                'type': 'object',
                'properties': {
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'title': {'type': 'string'},
                                'author': {'type': 'string'},
                                'cover_url': {'type': 'string'}
                            },
                            'required': ['title', 'author', 'cover_url'],
                            'additionalProperties': False
                        }
                    }
                },
                'required': ['items'],
                'additionalProperties': False
            },
            'strict': True
        }
        books_en = [{'title': b.get('title',''), 'author': b.get('author','')} for b in books]
        sys = (
            'You are a retrieval agent that MUST use the provided MCP server (Open Library) to find book cover URLs. '
            'Work in English when talking to tools. For each (title, author), do:\n'
            '1) Try get_book_by_title to resolve identifiers (OLID/ID/ISBN).\n'
            '2) Then use get_book_cover with the best id available to retrieve a `cover_url`.\n'
            'Return only the specified JSON schema.'
        )
        usr = 'Find cover images (cover_url) for these books: ' + json.dumps(books_en, ensure_ascii=False)
        payload = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': sys},
                {'role': 'user', 'content': usr}
            ],
            'text': {'format': schema, 'verbosity': 'medium'},
            'tools': [ build_openlibrary_mcp_tool(['get_book_by_title','get_book_cover','get_book_by_id']) ],
            'tool_choice': 'auto',
            'temperature': 0.2,
            'max_output_tokens': 1000
        }
        resp = _post(payload)
        usage = _get_usage(resp)
        out_text = resp.get('output_text') or ''
        try:
            data = json.loads(out_text) if out_text else {}
        except Exception:
            data = {}
        items = (data.get('items') or []) if isinstance(data, dict) else []
        if not items:
            txt = _extract_text(resp)
            items = []
            if txt:
                for b in books_en:
                    items.append({'title': b['title'], 'author': b['author'], 'cover_url': ''})
        return items, usage

    def improve_list_with_covers(self, keywords: str, books: List[Dict[str, str]]):
        annotations, usage_text, raw1 = self.improve_list(keywords, books)
        covers, usage_covers = self._fetch_covers_via_mcp(books)
        merged = {
            'input_tokens': usage_text['input_tokens'] + usage_covers['input_tokens'],
            'output_tokens': usage_text['output_tokens'] + usage_covers['output_tokens'],
            'total_tokens': usage_text['total_tokens'] + usage_covers['total_tokens'],
            'reasoning_tokens': usage_text['reasoning_tokens'] + usage_covers['reasoning_tokens'],
        }
        for i, b in enumerate(books):
            if i < len(covers):
                covers[i].setdefault('title', b.get('title',''))
                covers[i].setdefault('author', b.get('author',''))
        return annotations, merged, covers

class Agent3SummarizerGUI:
    def __init__(self, model: str = SUMMARIZER_MODEL):
        self.model = model
        self.soft_limit = int(MAX_CONTEXT_TOKENS * SOFT_LIMIT_RATIO)

    def summarize_text(self, raw_text: str, budget_tokens: int) -> Tuple[str, Dict[str, int]]:
        system = (
            'Ты — компрессор контента. Сожми входной текст в 5–10 кратких тезисов '
            'и 3–7 тегов по темам/настроению (без лишней воды), сохрани язык оригинала. '
            'Никаких вступлений — только список тезисов и теги.'
        )
        prompt = f'Ограничение: уложиться примерно в {budget_tokens} входных токенов дальше по пайплайну.\n\nТекст:\n{raw_text}'
        est = estimate_messages_tokens(system, prompt)
        if est <= self.soft_limit:
            payload = {
                'model': self.model,
                'input': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.2,
                'truncation': 'auto'
            }
            resp = _post(payload)
            txt = _extract_text(resp) or ''
            usage = _get_usage(resp)
            return txt.strip(), usage
        return self._summarize_map_reduce(raw_text, budget_tokens)

    def _safe_chunks(self, text: str, chunk_token_budget: int) -> List[str]:
        parts = [p for p in text.split('\n') if p.strip()]
        chunks = []
        cur = []
        cur_tokens = 0
        for p in parts:
            t = rough_token_estimate(p)
            if t > chunk_token_budget:
                piece_chars = tokens_to_chars(chunk_token_budget)
                s = p
                while s:
                    piece = s[:piece_chars]; s = s[piece_chars:]
                    if cur:
                        chunks.append('\n'.join(cur)); cur, cur_tokens = [], 0
                    chunks.append(piece)
                continue
            if cur_tokens + t > chunk_token_budget and cur:
                chunks.append('\n'.join(cur)); cur, cur_tokens = [], 0
            cur.append(p); cur_tokens += t
        if cur: chunks.append('\n'.join(cur))
        return chunks

    def _summarize_chunk(self, chunk_text: str) -> Tuple[str, Dict[str, int]]:
        system = (
            'Ты — компрессор контента. Сожми входной текст в 5–10 кратких тезисов '
            'и 3–7 тегов по темам/настроению (без лишней воды), сохрани язык оригинала. '
            'Никаких вступлений — только список тезисов и теги.'
        )
        prompt = f'Сожми кусковой фрагмент. Дальше будет объединение. Вот текст:\n{chunk_text}'
        payload = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2,
            'truncation': 'auto'
        }
        resp = _post(payload)
        txt = _extract_text(resp) or ''
        usage = _get_usage(resp)
        return txt.strip(), usage

    def _summarize_map_reduce(self, raw_text: str, target_budget_tokens: int) -> Tuple[str, Dict[str, int]]:
        per_chunk_budget = max(512, min(4096, target_budget_tokens // 3))
        chunks = self._safe_chunks(raw_text, per_chunk_budget)
        total_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'reasoning_tokens': 0}
        partial_summaries = []
        for i, ch in enumerate(chunks, 1):
            s, u = self._summarize_chunk(ch)
            partial_summaries.append(f'[КУСОК {i}]\n{s}')
            for k in total_usage: total_usage[k] += u[k]
        combined = '\n\n'.join(partial_summaries)
        system = 'Ты — редактор. Объедини тезисы из всех кусков в цельное краткое саммари (8–12 bullets + 5–10 тегов). Без повтора одно и того же.'
        payload = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': combined}
            ],
            'temperature': 0.2,
            'truncation': 'auto'
        }
        resp = _post(payload)
        final_text = _extract_text(resp) or ''
        u = _get_usage(resp)
        for k in total_usage: total_usage[k] += u[k]
        return final_text.strip(), total_usage

def hard_truncate(text: str, max_chars: int = 8000, keep_head: int = 4000) -> str:
    if len(text) <= max_chars: return text
    tail = max_chars - keep_head
    if tail <= 0: return text[:max_chars]
    return text[:keep_head] + '\n…\n' + text[-tail:]
