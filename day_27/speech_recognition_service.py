from __future__ import annotations

import io
from typing import Optional

import speech_recognition as sr


def _map_language(lang: Optional[str]) -> str:
    """
    Маппинг простых языковых кодов на форматы,
    которые ожидает Google Web Speech API.
    """
    if not lang:
        return "ru-RU"  # по умолчанию русский

    lang = lang.lower()
    if lang == "ru":
        return "ru-RU"
    if lang == "en":
        return "en-US"
    # если пользователь передал уже полный код (типа 'fr-FR') — используем как есть
    return lang


def transcribe_audio_bytes(
    audio_bytes: bytes,
    language: Optional[str] = "ru",
) -> str:
    """
    Преобразует байты аудио (из streamlit_mic_recorder) в текст
    с помощью библиотеки SpeechRecognition + Google Web Speech API.

    :param audio_bytes: байты аудио (WAV/OGG и т.п.)
    :param language: 'ru', 'en' или полный код локали ('ru-RU', 'en-US' и т.п.)
    :return: распознанный текст или пустая строка при неудаче
    """
    if not audio_bytes:
        return ""

    recognizer = sr.Recognizer()
    lang_code = _map_language(language)

    # Оборачиваем bytes в file-like объект
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=lang_code)
        return text.strip()
    except sr.UnknownValueError:
        # Ничего не распознано
        return ""
    except sr.RequestError as e:
        # Проблема с запросом к Google
        # Можно логировать/показывать в UI, но здесь просто вернём пустую строку
        return ""
