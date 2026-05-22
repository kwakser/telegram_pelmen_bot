import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report
)
from sklearn.dummy import DummyClassifier


df = pd.read_csv("survey2.csv")


TARGET = "Ваш любимый Уральский Пельмень"


if "Отметка времени" in df.columns:
    df.drop(columns=["Отметка времени"], inplace=True)


age_column = "Ваш возраст(число)"


def age_to_group(age):
    try:
        age = float(age)

        if age < 18:
            return "до_18"
        elif age <= 22:
            return "18_22"
        elif age <= 30:
            return "23_30"
        elif age <= 40:
            return "31_40"
        else:
            return "40_plus"

    except:
        return "unknown"


df[age_column] = df[age_column].apply(age_to_group)


df = df.fillna("missing")

class_counts = df[TARGET].value_counts()

rare_classes = class_counts[
    class_counts < 4
].index

df[TARGET] = df[TARGET].replace(
    rare_classes,
    "Другое"
)

print(df[TARGET].value_counts())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

model_lr = Pipeline([
    ("prep", preprocessor),

    ("clf",
     LogisticRegression(
         max_iter=3000,
         class_weight="balanced",
         C=1.0,
         random_state=42
     ))
])

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores_lr = []

for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y), 1):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model_lr.fit(X_train, y_train)

    preds = model_lr.predict(X_val)

    score = f1_score(
        y_val,
        preds,
        average="macro",
        zero_division=0
    )

    scores_lr.append(score)

    print(
        f"Fold {fold}: "
        f"F1_macro = {score:.4f}"
    )

print("\n")
print("Mean F1_macro:", np.mean(scores_lr))
print("Std:", np.std(scores_lr))


dummy_scores = []

for train_idx, val_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(X_train, y_train)

    preds = dummy.predict(X_val)

    score = f1_score(
        y_val,
        preds,
        average="macro",
        zero_division=0
    )

    dummy_scores.append(score)

print("\n")
print(
    "Dummy Mean F1_macro:",
    np.mean(dummy_scores)
)
print(
    "Dummy Std:",
    np.std(dummy_scores)
)


model_lr.fit(X, y)


train_preds = model_lr.predict(X)

print("\n========== TRAIN REPORT ==========")

print(
    classification_report(
        y,
        train_preds,
        zero_division=0
    )
)