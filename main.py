import asyncio
import logging
import os
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from urllib.parse import urlparse

from rag_core import RAGCore


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


logging.info("Инициализация RAG-системы...")
try:
    rag_system = RAGCore()
    logging.info("RAG-система успешно инициализирована.")
except Exception as e:
    logging.error(f"Критическая ошибка при инициализации RAG-системы: {e}")
    rag_system = None


@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.answer(
        "Здравствуйте! Я ваш ИИ-ассистент для продуктовых исследований.\n\n"
        "Задайте мне вопрос, и я постараюсь найти на него ответ в базе знаний, "
        "сформированной на основе сайта delprof.ru.\n\n"
        "Например: 'Какие налоговые льготы доступны для резидентов технопарков?'"
    )


@dp.message(Command("update"))
async def update_command_handler(message: types.Message):
    if not rag_system:
        await message.answer("Извините, сервис временно недоступен. Невозможно выполнить обновление.")
        return

    command_args = message.text.split(maxsplit=2)
    if len(command_args) < 2:
        await message.answer(
            "Пожалуйста, укажите URL сайта для обновления.\n"
            "Пример: `/update https://new-knowledge-source.com`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    max_links = 50
    if len(command_args) == 3 and command_args[2].strip().isdigit():
        max_links = int(command_args[2].strip())
    url = command_args[1].strip()
    parsed_url = urlparse(url)
    if not (parsed_url.scheme and parsed_url.netloc):
        await message.answer(
            "Вы ввели некорректный URL. Пожалуйста, убедитесь, что он начинается с http:// или https://")
        return

    logging.info(f"Пользователь {message.from_user.id} запустил обновление с URL: {url}")
    thinking_message = await message.answer(
        f"Начинаю процесс обновления с сайта {url}... 🚀\nЭто может занять несколько минут.")

    try:
        status = await asyncio.to_thread(rag_system.update_knowledge_base, url, max_links=max_links)
        await thinking_message.edit_text(status)
    except ValueError as e:
        logging.error(f"Ошибка обновления: {e}")
        await thinking_message.edit_text(f"Ошибка: {e}")
    except Exception as e:
        logging.error(f"Критическая ошибка во время обновления: {e}", exc_info=True)
        await thinking_message.edit_text("Произошла непредвиденная ошибка во время обновления. Попробуйте позже.")


@dp.message(F.text)
async def handle_query(message: types.Message):
    if not rag_system:
        await message.answer(
            "Извините, сервис временно недоступен из-за ошибки инициализации. Пожалуйста, попробуйте позже.")
        return

    user_query = message.text
    logging.info(f"Получен запрос от пользователя {message.from_user.id}: {user_query}")

    thinking_message = await message.answer("Думаю... 🧠 Пожалуйста, подождите.")

    try:
        answer, sources = await asyncio.to_thread(rag_system.get_answer, user_query)

        if sources:
            response_text = f"{answer}\n\n"
            sources_text = "\n".join(
                [f"• [{os.path.basename(s).replace('-', ' ').capitalize()}]({s})" for s in sources])
            response_text += f"*Источники:*\n{sources_text}"
        else:
            response_text = answer

        await thinking_message.edit_text(response_text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    except ValueError as e:
        logging.error(f"Ошибка при обработке запроса '{user_query}': {e}", exc_info=True)
        await thinking_message.edit_text("Не удалось собрать документы для создания базы знаний.\n Попробуйте добавить файлы в векторную БД используя /update")
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса '{user_query}': {e}", exc_info=True)
        await thinking_message.edit_text(
            "Произошла внутренняя ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз позже.")


async def main():
    """Основная функция для запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logging.critical("Токен Telegram-бота не найден. Убедитесь, что он указан в файле .env")
        return

    logging.info("Запуск бота...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())