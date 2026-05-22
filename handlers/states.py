from aiogram.fsm.state import State, StatesGroup


class QuizStates(StatesGroup):
    answering   = State()   # Пользователь отвечает на вопросы
    user_choice = State()   # Ожидаем подтверждение / выбор своего пельменя
    result      = State()   # Показан финальный результат
