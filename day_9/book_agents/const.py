# -*- coding: utf-8 -*-
import os
API_URL = os.getenv('API_URL', 'https://api.openai.com/v1/responses')
WEAK_MODEL = os.getenv('WEAK_MODEL', 'gpt-4o')
STRONG_MODEL = os.getenv('STRONG_MODEL', 'gpt-4o')
SUMMARIZER_MODEL = os.getenv('SUMMARIZER_MODEL', 'gpt-4o-mini')
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:6277')
MCP_PROXY_AUTH_TOKEN = os.getenv('MCP_PROXY_AUTH_TOKEN', '')
MCP_AUTH_HEADER_NAME = os.getenv('MCP_AUTH_HEADER_NAME', 'Authorization')
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', '120000'))
SOFT_LIMIT_RATIO = float(os.getenv('SOFT_LIMIT_RATIO', '0.85'))
AGENT1_SCHEMA = {
    'type': 'json_schema',
    'name': 'BookFinderTurn',
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'message': {'type': 'string'},
            'keywords': {'type': 'string'},
            'books': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'title': {'type': 'string'},
                        'author': {'type': 'string'},
                        'reason': {'type': 'string'}
                    },
                    'required': ['title', 'author', 'reason']
                }
            }
        },
        'required': ['message', 'keywords', 'books']
    },
    'strict': True
}
AGENT1_SYSTEM = (
    "You are a book recommendation assistant. Return EXACTLY three fiction books "
    "that match the user's genres/mood/themes. Output strict JSON via the BookFinderTurn schema."
)
AGENT2_SYSTEM = (
    "You are a skilled literary editor. Given a list of books (title, author, reason), "
    "write a short 2–4 sentence annotation for each: what's it about, key themes/mood, and who will like it. "
    "Respond in the same language as the input."
)
