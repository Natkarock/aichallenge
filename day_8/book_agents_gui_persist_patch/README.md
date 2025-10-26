# 📚 Книжный мультиагент — GUI (Streamlit)

## Почему Streamlit

- Кроссплатформенно, работает на macOS
- Лёгкий старт: `pip install -r requirements.txt && streamlit run app.py`

## Установка и запуск (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Экспортируйте ключ:
export OPENAI_API_KEY="sk-..."

# Запуск
streamlit run app.py
```

Сайдбар: настройка бюджета токенов, суммаризация/обрезка, кнопка демо.
Главное окно: чат, баблы агентов, usage
