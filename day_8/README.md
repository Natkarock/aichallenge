# 📚 Книжный мультиагент — GUI (Streamlit)

- Баблы теперь **персистят** в `st.session_state.messages` и не пропадают после демо.
- Кнопка "🗑 Очистить" — очищает историю.

## Запуск (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```
