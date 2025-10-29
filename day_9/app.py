# -*- coding: utf-8 -*-
import streamlit as st
from book_agents.agent_gui import Agent1BookFinderGUI, Agent2BookSummarizerGUI, Agent3SummarizerGUI, hard_truncate
from book_agents.api_functions import rough_token_estimate, _extract_text
from book_agents.const import WEAK_MODEL, STRONG_MODEL, SUMMARIZER_MODEL

st.set_page_config(page_title='Book Assistant GUI (MCP)', page_icon='📚', layout='centered')

if 'messages' not in st.session_state: st.session_state.messages = []
if 'target_budget' not in st.session_state: st.session_state.target_budget = 4000
if 'allow_summary' not in st.session_state: st.session_state.allow_summary = True
if 'hard_truncate' not in st.session_state: st.session_state.hard_truncate = False

st.sidebar.header('Settings')
st.sidebar.write(f'Models:\n- Agent1: `{WEAK_MODEL}`\n- Agent2: `{STRONG_MODEL}`\n- Summarizer: `{SUMMARIZER_MODEL}`')
st.session_state.target_budget = st.sidebar.number_input('Token budget (input)', 512, 32000, st.session_state.target_budget, step=256)
st.session_state.allow_summary = st.sidebar.checkbox('Allow summarization', True)
st.session_state.hard_truncate = st.sidebar.checkbox('Hard truncate instead of summarize', False)
if st.sidebar.button('🗑 Clear'):
    st.session_state.messages = []

def add(role, text):
    st.session_state.messages.append({'role': role, 'content': text})
    with st.chat_message(role):
        st.markdown(text)

def render_history():
    for m in st.session_state.messages:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

render_history()

def run_once(user_text: str):
    est = rough_token_estimate(user_text)
    add('assistant', f'_Estimated length: ~{est} tokens_')

    pre = user_text
    pre_usage = {'input_tokens':0,'output_tokens':0,'total_tokens':0,'reasoning_tokens':0}
    if est > st.session_state.target_budget:
        if st.session_state.allow_summary and not st.session_state.hard_truncate:
            add('assistant', '⏳ Summarizing long input…')
            s = Agent3SummarizerGUI()
            pre, pre_usage = s.summarize_text(user_text, st.session_state.target_budget)
            add('assistant', '✅ Summarization done.')
            add('assistant', f'**usage (summarizer):**\n\n```json\n{pre_usage}\n```')
        else:
            pre = hard_truncate(user_text)
            add('assistant', '✂️ Hard truncated input.')

    add('assistant', '🤖 Agent1 is picking books…')
    a1 = Agent1BookFinderGUI()
    res1, u1, _ = a1.run(pre)

    lines = [f'**Keywords:** {res1["keywords"]}', '**Top 3 books:**']
    for b in res1['books']:
        lines.append(f'- {b.get("title","")} — {b.get("author","")}')
    lines.append(f'\n**usage (Agent1):**\n\n```json\n{u1}\n```')
    add('assistant', '\n'.join(lines))

    add('assistant', '📝 Agent2 is writing annotations + fetching covers via MCP…')
    a2 = Agent2BookSummarizerGUI()
    annotations, u2, covers = a2.improve_list_with_covers(res1['keywords'], res1['books'])
    add('assistant', '**Annotations:**\n\n' + annotations)
    add('assistant', f'**usage (Agent2 total incl. MCP):**\n\n```json\n{u2}\n```')

    if covers:
        add('assistant', '**Covers:**')
        for c in covers:
            url = c.get('cover_url') or ''
            if url:
                st.image(url, width=180, caption=f"{c.get('title','')}")

    total = {
        'input_tokens': pre_usage['input_tokens']+u1['input_tokens']+u2['input_tokens'],
        'output_tokens': pre_usage['output_tokens']+u1['output_tokens']+u2['output_tokens'],
        'total_tokens': pre_usage['total_tokens']+u1['total_tokens']+u2['total_tokens'],
        'reasoning_tokens': pre_usage['reasoning_tokens']+u1['reasoning_tokens']+u2['reasoning_tokens'],
    }
    add('assistant', f'**Total usage:**\n\n```json\n{total}\n```')

prompt = st.chat_input('What do you want to read…')
if prompt:
    add('user', prompt)
    run_once(prompt)
    
# app.py
with st.sidebar.expander("MCP diagnostics"):
    import os, json
    from book_agents.api_functions import _post, _get_usage, build_openlibrary_mcp_tool
    st.write("MCP_SERVER_URL:", os.getenv("MCP_SERVER_URL"))
    st.write("MCP_PROXY_AUTH_TOKEN set:", bool(os.getenv("MCP_PROXY_AUTH_TOKEN")))
    if st.button("Test MCP"):
        payload = {
            "model": "gpt-4o",
            "input": [
                {"role": "system", "content": "Use MCP to list tools."},
                {"role": "user", "content": "List available tools."}
            ],
            "tools": [ build_openlibrary_mcp_tool(["get_book_by_title"]) ],
            "tool_choice": "auto",
        }
        try:
            resp = _post(payload)
            st.code(json.dumps(resp.get("usage", {}), indent=2, ensure_ascii=False))
            text = _extract_text(resp)
            print(text)
            st.text_area("output_text", text)
        except Exception as e:
            st.error(str(e))

