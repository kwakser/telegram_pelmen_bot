import logging

from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext

from handlers.states import QuizStates
from keyboards.inline import (
    build_answer_keyboard,
    build_user_choice_keyboard,
    build_restart_keyboard,
)
from config import resolve_path
from data.quiz_data import QUESTIONS, PELMENI
from data.dataset import save_result
from ml.predictor import predictor

router = Router()
logger = logging.getLogger(__name__)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "*Добро пожаловать в викторину Уральских Пельменей!*\n\n"
        "Ответь на несколько вопросов — и узнаешь, "
        "кто из участников тебе ближе всего.\n\n"
        "Нажми /quiz чтобы начать!",
        parse_mode="Markdown"
    )


@router.message(Command("quiz"))
async def cmd_quiz(message: Message, state: FSMContext):
    await start_quiz(message, state)


async def start_quiz(event: Message | CallbackQuery, state: FSMContext):
    await state.set_state(QuizStates.answering)
    await state.update_data(answers={}, current_q=0)
    target = event.message if isinstance(event, CallbackQuery) else event
    await ask_question(target, state)


@router.callback_query(F.data == "restart")
async def cb_restart(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await start_quiz(callback, state)


async def ask_question(message: Message, state: FSMContext):
    data = await state.get_data()
    idx: int = data.get("current_q", 0)

    if idx >= len(QUESTIONS):
        await show_model_result(message, state)
        return

    q = QUESTIONS[idx]
    progress = f"Вопрос {idx + 1} из {len(QUESTIONS)}"
    text = f"*{progress}*\n\n{q['text']}"

    if q.get("options") is None:
        await message.answer(text, parse_mode="Markdown")
    else:
        await message.answer(
            text,
            reply_markup=build_answer_keyboard(q["options"], q["id"]),
            parse_mode="Markdown"
        )


@router.callback_query(QuizStates.answering, F.data.startswith("ans:"))
async def cb_answer(callback: CallbackQuery, state: FSMContext):
    await callback.answer()

    _, q_id, idx_str = callback.data.split(":", 2)
    try:
        option_idx = int(idx_str)
    except ValueError:
        return

    q = next((item for item in QUESTIONS if item["id"] == q_id), None)
    if not q or not q.get("options") or not (0 <= option_idx < len(q["options"])):
        return

    answer = q["options"][option_idx]

    data = await state.get_data()
    answers: dict = data.get("answers", {})
    answers[q_id] = answer
    current_q: int = data.get("current_q", 0) + 1
    await state.update_data(answers=answers, current_q=current_q)

    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await ask_question(callback.message, state)


@router.message(QuizStates.answering)
async def handle_text_answer(message: Message, state: FSMContext):
    data = await state.get_data()
    idx: int = data.get("current_q", 0)

    if idx >= len(QUESTIONS):
        return

    q = QUESTIONS[idx]

    if q.get("options") is not None:
        await message.answer("Выбери вариант из кнопок выше")
        return

    if q["id"] == "q_age":
        if not message.text.strip().isdigit():
            await message.answer("Пожалуйста, введи число")
            return

    answers: dict = data.get("answers", {})
    answers[q["id"]] = message.text.strip()
    await state.update_data(answers=answers, current_q=idx + 1)
    await ask_question(message, state)


async def show_model_result(message: Message, state: FSMContext):
    await state.set_state(QuizStates.user_choice)

    data = await state.get_data()
    answers: dict = data.get("answers", {})

    pelmen_key = predictor.predict(answers)
    pelmen = PELMENI.get(pelmen_key, list(PELMENI.values())[0])
    await state.update_data(model_prediction=pelmen_key)

    caption = (
        f"*Твой Уральский Пельмень - {pelmen['name']}!*\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*А ты согласен с результатом?*\n"
        "_Если нет - выбери своего пельменя ниже.\n"
        "Это поможет сделать модель точнее!_"
    )

    keyboard = build_user_choice_keyboard(PELMENI)
    image_path = pelmen.get("image_path", "")

    if image_path.startswith("http"):
        await message.answer_photo(
            photo=image_path, caption=caption,
            reply_markup=keyboard, parse_mode="Markdown"
        )
    else:
        local_image = resolve_path(image_path)
        if local_image.is_file():
            await message.answer_photo(
                photo=FSInputFile(local_image), caption=caption,
                reply_markup=keyboard, parse_mode="Markdown"
            )
        else:
            logger.warning("Image not found: %s", local_image)
            await message.answer(caption, reply_markup=keyboard, parse_mode="Markdown")


@router.callback_query(QuizStates.user_choice, F.data.startswith("choice:"))
async def cb_user_choice(callback: CallbackQuery, state: FSMContext):
    await callback.answer()

    user_answer = callback.data.split(":", 1)[1]
    data = await state.get_data()
    answers: dict = data.get("answers", {})
    model_prediction: str = data.get("model_prediction", "")
    user_choice = model_prediction if user_answer == "agree" else user_answer

    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    user = callback.from_user
    save_result(
        user_id=user.id,
        username=user.username or user.full_name,
        answers=answers,
        model_prediction=model_prediction,
        user_choice=user_choice,
    )

    await state.set_state(QuizStates.result)

    if user_answer == "agree":
        text = "Данные сохранены! Спасибо за прохождение теста."
    else:
        chosen = PELMENI.get(user_choice, {})
        name = chosen.get("name", user_choice)
        text = (
            f"Понял! Записали *{name}* как твоего любимого пельменя.\n"
            "Это поможет модели стать точнее. Спасибо!"
        )

    await callback.message.answer(
        text, reply_markup=build_restart_keyboard(), parse_mode="Markdown"
    )


@router.message(QuizStates.user_choice)
async def guard_choice(message: Message):
    await message.answer("Нажми одну из кнопок выше")


@router.message(QuizStates.result)
async def guard_result(message: Message):
    await message.answer(
        "Нажми /quiz чтобы пройти тест ещё раз!",
        reply_markup=build_restart_keyboard()
    )
