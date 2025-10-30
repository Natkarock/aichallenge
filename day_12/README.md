
# ⛅ Weather → Image → PDF (MCP Agent)

Streamlit-чат, который запускает цепочку через MCP-инструменты:
1) получает погоду (MCP `weather_mcp`),  
2) генерирует иллюстрацию (MCP `image-gen` / Replicate),  
3) собирает короткий Markdown-отчёт и конвертирует его в PDF (MCP `markdown2pdf`).  

В чате показывается текстовое описание и **картинки с “шиммером”** на время загрузки. PDF сохраняется на диск.

---

## ✨ Возможности

- 🗣️ Простое общение: вводите «Погода в <городе>» — агент сам решает, какие инструменты вызывать.  
- 🌤️ MCP-погода → краткая сводка.  
- 🖼️ MCP-генерация изображений → список URL, картинки подтягиваются в UI с анимированным **shimmer-плейсхолдером**.  
- 📄 MCP-конвертация Markdown→PDF → файл сохраняется в указанную папку.  
- 🧩 Работает поверх LangGraph/LCEL + LangChain + `langchain_mcp_adapters`.

---

## 📁 Структура (минимум)

```
project/
├─ main.py           # Streamlit UI (шимер, парсинг финального JSON)
├─ agent.py          # Граф агента: weather → image → markdown2pdf (MCP)
├─ requirements.txt  # Зависимости Python
└─ .env              # Ключи и параметры (см. ниже)
```

---

## 🔧 Установка

```bash
# 1) Python окружение
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Node.js (для markdown2pdf и image-gen через npx)
# macOS: установите Node 18+ (или новее). Проверьте:
node -v
npx -v
```

---

## 🔑 Переменные окружения

Создайте `.env` рядом с проектом (или выставляйте в системе/IDE):

### Обязательные / часто используемые
```
OPENAI_API_KEY=sk-...                # ключ провайдера OpenAI (для ChatOpenAI)
OPENAI_MODEL=gpt-4.1-mini            # можно менять в UI (sidebar)

REPLICATE_API_TOKEN=r8_...           # токен Replicate для image-gen MCP
REPLICATE_MODEL=google/nano-banana   # можно менять в UI (sidebar)
```

### MCP Weather
```
# В agent.py уже указан публичный SSE URL, но при необходимости замените:
# WEATHER_MCP_URL=https://your-weather-mcp.example/mcp
```

### MCP Markdown→PDF
```
# Папка для сохранения pdf; должна существовать и быть доступна на запись
M2P_OUTPUT_DIR=/abs/path/to/output/dir

## ▶️ Запуск

```bash
streamlit run main.py
```

Откроется UI. Введите, например:  
`Погода в Краснодаре`  

Агент вызовет MCP-погоду → сгенерирует картинку → сформирует Markdown и сохранит PDF.

---

## 🧠 Как это работает

- `main.py`
  - передаёт запрос в `agent.run_with_message(prompt)`;
  - берёт **последнее сообщение** из `result["messages"]` — там лежит **валидный JSON**:
    ```json
    {
      "description": "Погода в Краснодаре: ...",
      "images": [
        "https://replicate.delivery/.../out-0.webp"
      ]
    }
    ```
  - отображает `description` и **подгружает картинки с shimmer**.

- `agent.py`
  - собирает MCP-инструменты через `MultiServerMCPClient`;
  - строит граф (`StateGraph`): узел агента → `ToolNode` → агент … до финального JSON;
  - инструменты:
    - `weather_mcp` (HTTP SSE),
    - `image-gen` (stdio, `npx @gongrzhe/image-gen-server`),
    - `markdown2pdf` (stdio, `npx markdown2pdf-mcp@latest` **или** `node build/index.js`).

---

## 🧪 Пример финального ответа

```json
{
  "description": "Погода в Краснодаре: сейчас ясно, около 10°C. В ближайшие три дня — преимущественно пасмурно, 14–18°C днём.",
  "images": ["https://replicate.delivery/.../out-0.webp"]
}
```

