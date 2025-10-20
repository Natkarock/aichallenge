import json, os
from urllib import request
from .schemas import API_URL, MODEL, TURN_SCHEMA, BRIEF_SCHEMA
from .models import DesignerTurn, DesignBrief

def _post(payload:dict,timeout:int=90)->dict:
    api_key=os.getenv('OPENAI_API_KEY');
    if not api_key: raise RuntimeError('OPENAI_API_KEY не установлен')
    req=request.Request(API_URL,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),method='POST')
    req.add_header('Content-Type','application/json; charset=utf-8')
    req.add_header('Authorization',f'Bearer {api_key}')
    with request.urlopen(req,timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))

def _fallback_text(obj:dict)->str:
    chunks=[]
    for item in obj.get('output',[]) or []:
        for c in item.get('content',[]) or []:
            if c.get('type') in ('output_text','text') and isinstance(c.get('text'),str):
                chunks.append(c['text'])
    return ''.join(chunks).strip() if chunks else ''

def call_turn(messages)->DesignerTurn:
    try:
        obj=_post({'model':MODEL,'input':messages,'text':{'format':TURN_SCHEMA,'verbosity':'medium'}})
    except Exception:
        raise RuntimeError('Попробуй ещё раз!')
    text=obj.get('output_text') or _fallback_text(obj)
    if not text: raise RuntimeError('Попробуй ещё раз!')
    try:
        data=json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError('Попробуй ещё раз!')
    return DesignerTurn.from_dict(data)

def call_brief(system_prompt:str,user_prompt:str,temperature:float)->DesignBrief:
    try:
        obj=_post({'model':MODEL,'input':[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}], 'text':{'format':BRIEF_SCHEMA,'verbosity':'medium'}, 'temperature':temperature})
    except Exception:
        raise RuntimeError('Попробуй ещё раз!')
    text=obj.get('output_text') or _fallback_text(obj)
    if not text: raise RuntimeError('Попробуй ещё раз!')
    try:
        data=json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError('Попробуй ещё раз!')
    return DesignBrief.from_dict(data)
