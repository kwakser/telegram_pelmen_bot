import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("Пельменный_лор.xlsx")

print("\nSHAPE:")
print(df.shape)

print("\nКОЛОНКИ:\n")
print(df.columns)

df = df.drop(columns=[df.columns[0]])

# Последняя колонка = любимый участник
target_column = df.columns[-1]

print("\nTARGET COLUMN:")
print(target_column)

X = df.drop(columns=[target_column])
y = df[target_column]

# Превращаем текстовые признаки в числа
X = pd.get_dummies(X)

print("\nРАЗМЕР ПОСЛЕ ENCODING:")
print(X.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nБАЛАНС КЛАССОВ:\n")
print(df[target_column].value_counts())

counts = df[target_column].value_counts()

rare_classes = counts[counts < 3].index

df = df[~df[target_column].isin(rare_classes)]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN SHAPE:")
print(X_train.shape)

print("\nTEST SHAPE:")
print(X_test.shape)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print(f"ACCURACY: {accuracy:.2f}")
print("===================================\n")

print("CLASSIFICATION REPORT:\n")

print(classification_report(
    y_test,
    predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n===================================")
print("TOP 15 ВАЖНЫХ ПРИЗНАКОВ")
print("===================================\n")

print(importance_df.head(15))

print("\n===================================")
print("ТЕСТ ПРЕДСКАЗАНИЯ")
print("===================================\n")

# Пример нового пользователя
new_person = {
    "Ваш пол": "Мужской",
    "Ваш возраст(число)": 20,
    "К какому типу личности вы себя относите?": "Интроверт",
    "Любимый тип номера": "Пародии",
    "Какие выпуски вам нравятся больше": "Старые",
    "Что Вам ближе": "Тихий сарказм",
    "Какой тип юмора Вам ближе?": "Ирония",
    "Вам больше нравится": "Хаос и импровизация на сцене",
    "Вы выражаете эмоции:": "Открыто",
    "Вы предпочитаете ": "Спокойное времяпрепровождение",
    "Вы легко знакомитесь с новыми людьми?": "Нет",
    "При принятии решений Вам ближе:": "Логика",
    "Вы скорее:": "Сова",
    "???": "Вы едите, чтобы жить?"
}

# DataFrame из одного человека
new_df = pd.DataFrame([new_person])

# One-hot encoding
new_df = pd.get_dummies(new_df)

# Добавляем отсутствующие колонки
for col in X.columns:
    if col not in new_df.columns:
        new_df[col] = 0

# Правильный порядок колонок
new_df = new_df[X.columns]

# Предсказание
prediction = model.predict(new_df)

# Вероятности
probabilities = model.predict_proba(new_df)

# Перевод обратно в имя
predicted_member = encoder.inverse_transform(prediction)

print("ПРЕДСКАЗАННЫЙ ЛЮБИМЫЙ УЧАСТНИК:\n")
print(predicted_member[0])

print("\nТОП-3 ВЕРОЯТНОСТИ:\n")

# Индексы топ-3 вероятностей
top3_idx = probabilities[0].argsort()[-3:][::-1]

for idx in top3_idx:
    name = encoder.inverse_transform([idx])[0]
    prob = probabilities[0][idx]

    print(f"{name}: {prob:.2%}")