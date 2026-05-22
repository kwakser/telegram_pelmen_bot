"""
Модуль для записи результатов опросов в CSV-датасет.

Колонки:
  timestamp, user_id, username,
  q_gender, q_age, q_personality, q_show_type, q_episodes,
  q_humor_style, q_humor_type, q_stage_style, q_emotions,
  q_rest, q_social, q_decisions, q_chronotype, q_food_philosophy,
  model_prediction, user_choice, is_match
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from config import BASE_DIR
from data.quiz_data import QUESTIONS

logger = logging.getLogger(__name__)

DATASET_PATH = BASE_DIR / "data" / "dataset.csv"

_Q_COLS = [q["id"] for q in QUESTIONS]
FIELDNAMES = (
    ["timestamp", "user_id", "username"]
    + _Q_COLS
    + ["model_prediction", "user_choice", "is_match"]
)


def _ensure_file():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        logger.info(f"Создан новый датасет: {DATASET_PATH}")


def save_result(
    user_id: int,
    username: str,
    answers: Dict[str, str],
    model_prediction: str,
    user_choice: str,
):
    _ensure_file()

    row: Dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "username": username,
        "model_prediction": model_prediction,
        "user_choice": user_choice,
        "is_match": int(model_prediction == user_choice),
    }
    for q_id in _Q_COLS:
        row[q_id] = answers.get(q_id, "")

    try:
        with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        logger.info(
            f"Датасет обновлён | user={user_id} "
            f"model={model_prediction} choice={user_choice} "
            f"match={row['is_match']}"
        )
    except Exception as e:
        logger.error(f"Ошибка записи в датасет: {e}")
