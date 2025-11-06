from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List
from bs4 import BeautifulSoup

RU_STOP = {"и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было","вот","от","меня","еще","нет","о","из"}
EN_STOP = {"the","and","a","an","is","it","to","in","of","for","on","at","by","with","as","that","this","from","or","be","are","was","were"}

def clean_html(text: str) -> str:
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_stopwords(text: str) -> str:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9_+-]+", text)
    out = []
    for t in tokens:
        tl = t.lower()
        if tl in RU_STOP or tl in EN_STOP:
            continue
        out.append(t)
    return " ".join(out)

def read_text_safe(p: Path, limit_chars: int = 20000) -> str:
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
        return s[:limit_chars]
    except Exception:
        return ""

def glob_many(root: Path, patterns: Iterable[str]) -> List[Path]:
    res: List[Path] = []
    for pat in patterns:
        for p in root.rglob(pat.strip()):
            if p.is_file():
                res.append(p)
    return res
