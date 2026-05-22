from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def build_answer_keyboard(options: list[str], q_id: str) -> InlineKeyboardMarkup:
    """Кнопки с вариантами ответа для вопроса q_id.

    callback_data хранит индекс варианта, а не текст - у Telegram лимит 64 байта.
    """
    builder = InlineKeyboardBuilder()
    for idx, option in enumerate(options):
        builder.button(
            text=option,
            callback_data=f"ans:{q_id}:{idx}",
        )
    builder.adjust(2)
    return builder.as_markup()


def build_user_choice_keyboard(pelmeni: dict) -> InlineKeyboardMarkup:
    """
    Кнопки «Согласен» и все пельмени на случай если пользователь
    хочет указать другой правильный ответ для датасета.

    pelmeni - словарь PELMENI из quiz_data.py
    """
    builder = InlineKeyboardBuilder()
    builder.button(text="Да, это я!", callback_data="choice:agree")
    for key, data in pelmeni.items():
        builder.button(
            text=data['name'],
            callback_data=f"choice:{key}"
        )
    # Первая строка — кнопка согласия на всю ширину, остальные по 2
    builder.adjust(1, 2)
    return builder.as_markup()


def build_restart_keyboard() -> InlineKeyboardMarkup:
    """Кнопка для повторного прохождения теста."""
    builder = InlineKeyboardBuilder()
    builder.button(text="Пройти ещё раз", callback_data="restart")
    return builder.as_markup()
