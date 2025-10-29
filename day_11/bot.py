# bot.py — Telegram bot with 15-minute weather "reminder" using MCP-enabled agent
# Requires:
#   pip install 'python-telegram-bot[job-queue]==21.6' python-dotenv==1.0.1
#
# ENV (в .env или в окружении):
#   TELEGRAM_BOT_TOKEN   — токен телеграм-бота
#   OPENAI_API_KEY       — ключ для agent.py (OpenAI)
#   (опционально) MCP_SERVER_URL, OPENAI_MODEL, DEFAULT_CITY, REMINDER_INTERVAL_MINUTES
#
# Функциональность:
# - /start спрашивает город (по умолчанию — Москва) и показывает кнопки
# - Каждые N минут (по умолчанию 15) шлёт поэтическую сводку погоды
# - Кнопки: «Погода сейчас», «Сменить город», «Включить/Остановить рассылку»
# - Ответ формируется через run_agent() из agent.py (MCP weather tool)
# - NEW: при нажатии «Погода сейчас» сначала показывается сообщение «⏳ Получаю поэтическую сводку…»

import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    JobQueue,
    filters,
)

# === Импорт вашего MCP-агента ================================================
try:
    from agent import run_agent
except Exception as e:
    raise RuntimeError("Не удалось импортировать run_agent из agent.py") from e

# === Конфиг / окружение ======================================================
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")

REMINDER_INTERVAL_MIN = int(os.getenv("REMINDER_INTERVAL_MINUTES", "15"))
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Москва")

# === Логирование =============================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("tg-weather-reminder")

# === Вспомогательные функции =================================================
def kb_main(is_subscribed: bool) -> InlineKeyboardMarkup:
    """Основная inline-клавиатура."""
    rows = [
        [
            InlineKeyboardButton("🌤 Погода сейчас", callback_data="weather_now"),
            InlineKeyboardButton("🏙 Сменить город", callback_data="change_city"),
        ],
        [
            InlineKeyboardButton(
                "⏸ Остановить рассылку" if is_subscribed else "▶️ Включить рассылку",
                callback_data="toggle_sub",
            )
        ],
    ]
    return InlineKeyboardMarkup(rows)


def ensure_defaults(chat_data: Dict[str, Any]) -> None:
    """Гарантирует обязательные поля в chat_data."""
    if "city" not in chat_data:
        chat_data["city"] = DEFAULT_CITY
    if "subscribed" not in chat_data:
        chat_data["subscribed"] = True


async def fetch_poetic_weather(city: str) -> str:
    """Получить поэтическую сводку погоды через MCP-агента."""
    user_prompt = (
        f"Скажи текущую погоду в городе {city}. "
        f"Отвечай коротко, в 4–6 строк, строго в стихотворной форме. "
        f"Используй доступные инструменты, чтобы получить свежие данные."
    )
    # Минимальная история — агент сам делает summary при необходимости
    result = run_agent(user_prompt, chat_history=[{"role": "user", "content": user_prompt}])
    text = result.get("text", "Не удалось получить ответ")
    return text


def schedule_job_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Запланировать/перезапланировать периодическую отправку сводок для чата."""
    jq = context.job_queue
    if jq is None:
        logger.warning("JobQueue отсутствует — рассылка не запланирована")
        return

    job_name = f"reminder_{chat_id}"

    # Удаляем прошлые задания с тем же именем
    for job in jq.get_jobs_by_name(job_name):
        job.schedule_removal()

    # Создаём новое повторяющееся задание
    jq.run_repeating(
        callback=reminder_job,
        interval=REMINDER_INTERVAL_MIN * 60,
        first=5,  # первая отправка через 5 секунд для проверки
        chat_id=chat_id,
        name=job_name,
    )

# === Хендлеры ================================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    chat_data = context.chat_data
    ensure_defaults(chat_data)

    await context.bot.send_message(
        chat_id=chat.id,
        text=(
            "Привет! Я бот-напоминалка о погоде ☔️\n\n"
            f"Каждые {REMINDER_INTERVAL_MIN} минут пришлю краткую поэтическую сводку.\n"
            f"Город по умолчанию: *{chat_data['city']}*.\n\n"
            "Напишите новый город одним сообщением — или используйте кнопки ниже."
        ),
        parse_mode="Markdown",
        reply_markup=kb_main(is_subscribed=chat_data["subscribed"]),
    )

    if chat_data["subscribed"]:
        schedule_job_for_chat(context, chat.id)


async def reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Периодическая задача: отправка поэтической сводки в чат."""
    job = context.job
    chat_id = job.chat_id  # type: ignore[attr-defined]
    chat_data = context.chat_data
    ensure_defaults(chat_data)

    if not chat_data.get("subscribed", True):
        return

    city = chat_data.get("city", DEFAULT_CITY)
    try:
        text = await fetch_poetic_weather(city)
    except Exception as e:
        logger.exception("Ошибка в reminder_job: %s", e)
        text = "Не удалось получить сводку погоды."
    await context.bot.send_message(chat_id=chat_id, text=text)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Любой текст трактуем как установку/смену города."""
    chat_data = context.chat_data
    ensure_defaults(chat_data)

    city = (update.effective_message.text or "").strip()
    if city:
        chat_data["city"] = city
        await update.effective_message.reply_text(
            f"Город обновлён: *{city}*.",
            parse_mode="Markdown",
            reply_markup=kb_main(is_subscribed=chat_data["subscribed"]),
        )
        if chat_data["subscribed"]:
            schedule_job_for_chat(context, update.effective_chat.id)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    chat_data = context.chat_data
    ensure_defaults(chat_data)

    data = query.data
    if data == "weather_now":
        city = chat_data["city"]

        # Убираем инлайн-кнопки с исходного сообщения, чтобы избежать повторных нажатий
        await query.edit_message_reply_markup(reply_markup=None)

        # Показываем "печатает..." и временное сообщение "загрузка"
        await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.TYPING)
        temp_msg = await query.message.reply_text("⏳ Получаю поэтическую сводку…")

        try:
            text = await fetch_poetic_weather(city)
        except Exception as e:
            logger.exception("Ошибка в weather_now: %s", e)
            text = "Не удалось получить сводку погоды."

        # Обновляем временное сообщение на итоговый ответ и возвращаем клавиатуру
        await temp_msg.edit_text(text, reply_markup=kb_main(is_subscribed=chat_data["subscribed"]))

    elif data == "change_city":
        await query.message.reply_text("Введите название города одним сообщением. По умолчанию — Москва.")

    elif data == "toggle_sub":
        chat_data["subscribed"] = not chat_data["subscribed"]
        if chat_data["subscribed"]:
            schedule_job_for_chat(context, update.effective_chat.id)
            await query.message.reply_text("Рассылка включена ✅", reply_markup=kb_main(True))
        else:
            jq = context.job_queue
            if jq is not None:
                job_name = f"reminder_{update.effective_chat.id}"
                for job in jq.get_jobs_by_name(job_name):
                    job.schedule_removal()
            await query.message.reply_text("Рассылка остановлена ⏸", reply_markup=kb_main(False))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/start — начать и выбрать город\n"
        "/help — помощь\n\n"
        "Кнопки:\n"
        "• «Погода сейчас» — прислать поэтическую сводку\n"
        "• «Сменить город» — задать новый город\n"
        "• «Включить/Остановить рассылку» — раз в 15 минут"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок, чтобы не падать молча."""
    logger.exception("Unhandled error", exc_info=context.error)

# === Сборка и запуск приложения =============================================
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Страховка: на случай, если PTB установлен без extras и job_queue == None
    if app.job_queue is None:
        jq = JobQueue()
        jq.set_application(app)
        app.job_queue = jq

    # Хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_error_handler(on_error)

    return app


def main() -> None:
    app = build_app()
    logger.info(
        "Bot started. Interval: %s minutes. Default city: %s. JobQueue ready: %s",
        REMINDER_INTERVAL_MIN, DEFAULT_CITY, app.job_queue is not None
    )
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
