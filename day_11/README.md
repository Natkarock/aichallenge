# Telegram Reminder Bot (Weather + Poetry) with MCP tool

Этот бот каждые 15 минут присылает поэтическую сводку погоды в выбранном городе и умеет отправлять сводку по кнопке. Он использует ваш `agent.py` (MCP-enabled) для получения погоды через tools и формирует ответ в стихах.

## Что умеет

- По умолчанию город — **Москва** (меняется пользователем).
- Кнопки:
  - **Погода сейчас** — моментальный запрос.
  - **Сменить город** — введите название.
  - **Включить/Остановить рассылку** — контроль напоминаний.
- Каждые 15 минут отправляет сводку (можно настроить `REMINDER_INTERVAL_MINUTES`).

## Установка

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка

Скопируйте `.env.example` в `.env` и заполните:

```env
TELEGRAM_BOT_TOKEN=123456789:ABCDEFYourTelegramToken
OPENAI_API_KEY=sk-xxx
MCP_SERVER_URL=https://weathermcp-natkarock.amvera.io/mcp  # если требуется
```

## Запуск

```bash
python bot.py
```

## Как это работает

- `bot.py` импортирует `run_agent` из `agent.py` и формирует запрос:
  «Скажи текущую погоду в городе {город}… Отвечай строго в стихотворной форме… Используй инструменты».
- `agent.py` уже сконфигурирован на работу с OpenAI Responses API и MCP weather tool.
- Внутри Telegram используется `JobQueue` для периодической рассылки раз в 15 минут. Бот работает 24/7 при запущенном процессе.

## Подсказки

- Для деплоя в прод окружение используйте systemd/pm2/supervisor/Docker, чтобы процесс перезапускался автоматически.
- Если у вас уже есть `requirements.txt`, добавьте:
  ```text
  python-telegram-bot==21.6
  python-dotenv==1.0.1
  ```
  и переустановите зависимости.

Удачной погоды и хороших стихов! 🌦️📜
