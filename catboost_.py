import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report
)

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

rare_classes = class_counts[class_counts < 4].index

df[TARGET] = df[TARGET].replace(
    rare_classes,
    "Другое"
)

print("Распределение классов после объединения:\n")
print(df[TARGET].value_counts())


X = df.drop(columns=[TARGET])
y = df[TARGET]


cat_features = list(range(X.shape[1]))


# ==========================
# 9. Модель
# ==========================
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=5,

    loss_function="MultiClass",

    auto_class_weights="Balanced",

    random_seed=42,
    verbose=0
)


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    preds = model.predict(X_val)

    score = f1_score(
        y_val,
        preds,
        average="macro",
        zero_division=0
    )

    scores.append(score)

    print(f"Fold {fold}: F1_macro = {score:.4f}")

print("\n")
print("Mean F1_macro:", np.mean(scores))
print("Std:", np.std(scores))


from sklearn.dummy import DummyClassifier

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
print("Dummy Mean F1_macro:", np.mean(dummy_scores))
print("Dummy Std:", np.std(dummy_scores))


model.fit(
    X,
    y,
    cat_features=cat_features
)

feature_importance = model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n")
print(importance_df)


top_k_values = [5, 7, 10]

results = {}

best_score = -1
best_features = None

for k in top_k_values:

    selected_features = (
        importance_df.head(k)["feature"]
        .tolist()
    )

    X_selected = X[selected_features]

    cat_features_selected = list(
        range(X_selected.shape[1])
    )

    scores_selected = []

    for train_idx, val_idx in cv.split(
            X_selected, y):

        X_train = X_selected.iloc[train_idx]
        X_val = X_selected.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features_selected
        )

        preds = model.predict(X_val)

        score = f1_score(
            y_val,
            preds,
            average="macro",
            zero_division=0
        )

        scores_selected.append(score)

    mean_score = np.mean(scores_selected)

    results[k] = mean_score

    print(
        f"Top-{k}: "
        f"F1_macro = {mean_score:.4f}"
    )

    if mean_score > best_score:
        best_score = mean_score
        best_features = selected_features


print("\n")
print(results)
print(best_features)

print("\nBest score:", best_score)


X_selected = X[best_features]

cat_features_selected = list(
    range(X_selected.shape[1])
)

model.fit(
    X_selected,
    y,
    cat_features=cat_features_selected
)



train_preds = model.predict(X_selected)

print("\n========== TRAIN REPORT ==========")

print(
    classification_report(
        y,
        train_preds,
        zero_division=0
    )
)