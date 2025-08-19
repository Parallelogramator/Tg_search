import asyncio
import logging
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from dotenv import load_dotenv

from config import TELEGRAM_BOT_TOKEN, LOG_LEVEL, DEFAULT_SITE
from rag_core import RAGCore

load_dotenv()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN не найден в .env")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Инициализация RAG
logging.info("Инициализация RAG-системы...")
try:
    rag_system = RAGCore()
    logging.info("RAG-система успешно инициализирована.")
except Exception as e:
    logging.exception("Критическая ошибка при инициализации RAG-системы")
    rag_system = None

HELP_TEXT = (
    "*Команды:*\n"
    "/start — приветствие\n"
    "/help — справка\n"
    "/update <url> [max] — обновить базу знаний с сайта (по карте сайта)\n"
    "/stats — статистика базы (док-ов/чанков)\n"
    "/ping — проверка доступности\n\n"
    f"По умолчанию источник: {DEFAULT_SITE}\n"
    "Просто задайте вопрос текстом — я найду ответ в базе."
)


@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.answer(
        "Здравствуйте! Я ИИ-ассистент для продуктовых исследований.\n\n"
        "Задайте вопрос — найду ответ в базе знаний (delprof.ru и др.) и верну структурированный ответ со ссылками.\n\n"
        "Команды: /help"
    )


@dp.message(Command("help"))
async def help_handler(message: types.Message):
    await message.answer(HELP_TEXT, disable_web_page_preview=True)


@dp.message(Command("ping"))
async def ping_handler(message: types.Message):
    ok = rag_system is not None
    await message.answer("✅ Готов к работе." if ok else "❌ Сервис недоступен.")


@dp.message(Command("stats"))
async def stats_handler(message: types.Message):
    if not rag_system:
        await message.answer("❌ Сервис недоступен.")
        return
    stats = rag_system.get_stats()
    await message.answer(
        f"*Статистика базы:*\n"
        f"- Документов: {stats['documents']}\n"
        f"- Чанков: {stats['chunks']}\n"
        f"- Последнее обновление: {stats['last_updated']}"
    )


@dp.message(Command("update"))
async def update_command_handler(message: types.Message):
    if not rag_system:
        await message.answer("Сервис недоступен. Невозможно выполнить обновление.")
        return

    command_args = message.text.split(maxsplit=2)
    url = None
    max_links = None

    if len(command_args) >= 2:
        url = command_args[1].strip()
    if len(command_args) == 3 and command_args[2].strip().isdigit():
        max_links = int(command_args[2].strip())

    if not url:
        url = DEFAULT_SITE

    parsed_url = urlparse(url)
    if not (parsed_url.scheme and parsed_url.netloc):
        await message.answer("Неверный URL. Убедитесь, что он начинается с http:// или https://")
        return

    thinking_message = await message.answer(
        f"Начинаю обновление базы знаний с сайта {url}... Это может занять несколько минут. 🚀"
    )
    try:
        status = await asyncio.to_thread(rag_system.update_knowledge_base, url, max_links=max_links)
        await thinking_message.edit_text(status)
    except ValueError as e:
        logging.error("Ошибка обновления: %s", e)
        await thinking_message.edit_text(f"Ошибка: {e}")
    except Exception as e:
        logging.exception("Критическая ошибка обновления")
        await thinking_message.edit_text("Произошла непредвиденная ошибка. Попробуйте позже.")


@dp.message(F.text)
async def handle_query(message: types.Message):
    if not rag_system:
        await message.answer("Сервис временно недоступен. Пожалуйста, попробуйте позже.")
        return

    user_query = message.text.strip()
    thinking_message = await message.answer("Думаю... 🧠")

    try:
        messages, sources = await asyncio.to_thread(rag_system.get_answer, user_query)
        # красивое оформление ссылок
        '''if sources:
            src_lines = "\n".join([f"• [{s['title']}]({s['url']})" for s in sources])
            messages[-1] += f"\n\n*Источники:*\n{src_lines}"'''

        await thinking_message.edit_text(messages[0], disable_web_page_preview=True)

        for msg in messages[1:]:
            await message.answer(msg, disable_web_page_preview=True)
    except Exception:
        logging.exception("Ошибка при обработке запроса")
        await thinking_message.edit_text("Внутренняя ошибка. Попробуйте ещё раз.")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
