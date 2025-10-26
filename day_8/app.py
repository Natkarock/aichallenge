# -*- coding: utf-8 -*-
import streamlit as st

from book_agents.const import WEAK_MODEL, STRONG_MODEL, SUMMARIZER_MODEL
from book_agents.api_functions import rough_token_estimate
from book_agents.agent_gui import (
    Agent1BookFinderGUI, Agent2BookSummarizerGUI, Agent3SummarizerGUI, hard_truncate
)

st.set_page_config(page_title="Книжный мультиагент (GUI)", page_icon="📚", layout="centered")

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str}]
if "target_budget" not in st.session_state:
    st.session_state.target_budget = 4000
if "allow_summary" not in st.session_state:
    st.session_state.allow_summary = True
if "hard_truncate" not in st.session_state:
    st.session_state.hard_truncate = False

# ---------------- Sidebar ----------------
st.sidebar.header("Настройки")
st.sidebar.write(f"Модели:\n\n- Agent1: `{WEAK_MODEL}`\n- Agent2: `{STRONG_MODEL}`\n- Summarizer: `{SUMMARIZER_MODEL}`")
st.session_state.target_budget = st.sidebar.number_input("Целевой бюджет токенов на ввод", 512, 32000, st.session_state.target_budget, step=256)
st.session_state.allow_summary = st.sidebar.checkbox("Включить суммаризацию длинного ввода", True)
st.session_state.hard_truncate = st.sidebar.checkbox("Жёсткая обрезка вместо суммаризации", False, help="Если включено — длинный ввод будет обрезан.")

demo = st.sidebar.button("▶ Демо 3 запросов разной длины") 
oversize_demo = st.sidebar.button("▶ Демо запроса сверхлимита") 
clear = st.sidebar.button("🗑 Очистить")

if clear:
    st.session_state.messages = []

st.title("📚 Книжный мультиагент")
st.caption("Agent1 (подбор) → Agent2 (аннотации) + Agent3 (саммаризация). Баблы, лоадер и usage.")

# --------------- Helpers ---------------
def add_bubble(role: str, text: str):
    """Persist + render a bubble."""
    st.session_state.messages.append({"role": role, "content": text})
    with st.chat_message(role):
        st.markdown(text)

def add_user_bubble(text: str): add_bubble("user", text)
def add_assistant_bubble(text: str): add_bubble("assistant", text)

def show_history():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# Render existing history first
show_history()

def run_once(user_text: str):
    # 1) length estimate
    est = rough_token_estimate(user_text)
    add_assistant_bubble(f"_Грубая оценка длины ввода: ~{est} токенов_")

    preprocessed = user_text
    pre_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    # 2) preprocess
    if est > st.session_state.target_budget and (st.session_state.allow_summary or st.session_state.hard_truncate):
        if not st.session_state.hard_truncate:
            add_assistant_bubble("⏳ **Суммаризирую длинный ввод…**")
            summarizer = Agent3SummarizerGUI()
            preprocessed, pre_usage = summarizer.summarize_text(user_text, st.session_state.target_budget)
            add_assistant_bubble("✅ **Готово. Суммаризация выполнена.**")
            add_assistant_bubble(f"**usage (summarizer):**\n\n```json\n{pre_usage}\n```")
        else:
            preprocessed = hard_truncate(user_text)
            add_assistant_bubble("✂️ **Длинный ввод обрезан (hard truncate).**")

    # 3) Agent1
    add_assistant_bubble("🤖 **Agent1 подбирает книги…**")
    agent1 = Agent1BookFinderGUI()
    result1, usage1, _ = agent1.run(preprocessed)

    msg_lines = [
        f"**Ключевые слова:** {result1['keywords']}",
        "**Три подходящие книги:**",
        *[f"- {b.get('title','')} — {b.get('author','')}" for b in result1['books']],
        f"\n**usage (Agent1):**\n\n```json\n{usage1}\n```"
    ]
    add_assistant_bubble("\n".join(msg_lines))

    # 4) Agent2
    add_assistant_bubble("📝 **Agent2 пишет аннотации…**")
    agent2 = Agent2BookSummarizerGUI()
    annotations, usage2, _ = agent2.improve_list(result1["keywords"], result1["books"])
    add_assistant_bubble("**Аннотации:**\n\n" + annotations)
    add_assistant_bubble(f"**usage (Agent2):**\n\n```json\n{usage2}\n```")

    # 5) totals
    total_usage = {
        "input_tokens": pre_usage["input_tokens"] + usage1["input_tokens"] + usage2["input_tokens"],
        "output_tokens": pre_usage["output_tokens"] + usage1["output_tokens"] + usage2["output_tokens"],
        "total_tokens": pre_usage["total_tokens"] + usage1["total_tokens"] + usage2["total_tokens"],
        "reasoning_tokens": pre_usage["reasoning_tokens"] + usage1["reasoning_tokens"] + usage2["reasoning_tokens"],
    }
    add_assistant_bubble(f"**Итоговая статистика usage:**\n\n```json\n{total_usage}\n```")

# --------------- Input ---------------
prompt = st.chat_input("Опишите, что хотите почитать…")
if prompt:
    add_user_bubble(prompt)
    run_once(prompt)

# --------------- Demo ---------------
if demo:
    short_query = "Хочу тёплую атмосферную прозу про осень, меланхолию и маленький европейский город."
    long_query = (
        "Нужны романы и рассказы с неторопливым ритмом, атмосферой старых улочек, "
        "кафе, дождливых парков. Пусть будет лиричность, размышления о семье, "
        "о памяти и взрослении, но без чернухи. Интересны авторы, где важны детали быта, "
        "вкус к чтению, уют и мягкая ирония. Допустимы переводные вещи (европейские или японские), "
        "но чтобы язык в русском переводе был живой и не «сухой». Можно немного магического реализма."
        "\n— Также добавлю большой фрагмент дневников/заметок на несколько тысяч символов, "
        "который сильно раздувает ввод, и который нам не нужен целиком, а только как ориентир для тем/настроений. "
        * 20
    )
    oversized = ("Очень-длинный-ввод " * 20000) +                 "\nТема: тёплая проза о городе, осени, семейной памяти, без мрачняка."

    cases = [
        ("КОРОТКИЙ ЗАПРОС", short_query),
        ("ДЛИННЫЙ ЗАПРОС", long_query),
        ("СВЕРХЛИМИТНЫЙ ЗАПРОС", oversized),
    ]

    for title, text in cases:
        add_user_bubble(title)
        run_once(text)
        
 # --------------- Demo Oversize---------------
if oversize_demo:
    oversized = ("Очень-длинный-ввод " * 20000) +                 "\nТема: тёплая проза о городе, осени, семейной памяти, без мрачняка." 
    
    add_user_bubble("СВЕРХЛИМИТНЫЙ ЗАПРОС")
    run_once(oversized )
