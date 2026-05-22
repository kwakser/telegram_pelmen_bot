"""
Обёртка над ML-моделью.

Как подключить свою модель:
1. Положи файл в ml/model.pkl (sklearn) или ml/model.pt (PyTorch)
2. Убедись, что _encode_answers() кодирует признаки так же, как при обучении
3. Заполни LABEL_MAP: числовой выход модели → ключ из PELMENI

Ключи PELMENI:
  rozhkov, sokolov, brekotkin, myasnikov, isaev,
  yaritsa, yuryeva, korneva, popov, postovalov, mikhalkova
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path
import joblib
logger = logging.getLogger(__name__)

# Числовые метки модели → ключи PELMENI
LABEL_MAP: Dict[Any, str] = {
    0:  "rozhkov",
    1:  "sokolov",
    2:  "brekotkin",
    3:  "myasnikov",
    4:  "isaev",
    5:  "yaritsa",
    6:  "yuryeva",
    7:  "korneva",
    8:  "popov",
    9:  "postovalov",
    10: "mikhalkova",
}
_ML_DIR = Path(__file__).resolve().parent
# Кодировка вариантов ответа для каждого вопроса
# Вопросы с options=None (свободный ввод — возраст) кодируются отдельно
_OPTIONS_MAP: Dict[str, Dict[str, int]] = {}

def _build_options_map():
    from data.quiz_data import QUESTIONS
    for q in QUESTIONS:
        if q.get("options"):
            _OPTIONS_MAP[q["id"]] = {opt: i for i, opt in enumerate(q["options"])}

_build_options_map()


class PelmenPredictor:

    def __init__(self, model_path: str = "model.pkl"):
        self.model = None
        self.model_path = str(_ML_DIR / model_path)
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Файл модели '{self.model_path}' не найден. "
                "Используется заглушка."
            )
            return
        try:
            import pickle
            with open(self.model_path, "rb") as f:
                self.model = joblib.load(self.model_path)
            logger.info(f"Модель загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

    def _encode_answers(self, answers: Dict[str, str]) -> list:
        """
        Преобразует ответы в числовой вектор.

        - Варианты ответа → индекс варианта (0, 1, 2...)
        - Возраст → число напрямую (или 0 если не введён)

        Замени логику под свою модель если нужно другое кодирование!
        """
        from data.quiz_data import QUESTIONS
        features = []
        for q in QUESTIONS:
            q_id = q["id"]
            answer = answers.get(q_id, "")

            if q_id == "q_age":
                # Числовой признак
                try:
                    features.append(int(answer))
                except (ValueError, TypeError):
                    features.append(0)
            else:
                # Категориальный → индекс варианта
                opts = _OPTIONS_MAP.get(q_id, {})
                features.append(opts.get(answer, 0))

        return features

    def predict(self, answers: Dict[str, str]) -> str:
        features = self._encode_answers(answers)

        if self.model is None:
            # Заглушка: детерминированный выбор по сумме фич
            from data.quiz_data import PELMENI
            keys = list(PELMENI.keys())
            return keys[sum(features) % len(keys)]

        try:
            # sklearn
            raw = self.model.predict([features])[0]
            return LABEL_MAP.get(raw, "rozhkov")

            # PyTorch (раскомментируй если нужно):
            # import torch
            # with torch.no_grad():
            #     out = self.model(torch.tensor([features], dtype=torch.float32))
            #     raw = out.argmax().item()
            # return LABEL_MAP.get(raw, "rozhkov")

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            from data.quiz_data import PELMENI
            return list(PELMENI.keys())[0]


predictor = PelmenPredictor()
