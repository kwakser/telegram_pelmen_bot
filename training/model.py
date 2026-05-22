import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

df = pd.read_excel("Пельменный_лор.xlsx")
df = df.drop(columns=[df.columns[0]]) 
target_column = df.columns[-1]
print("TARGET COLUMN:", target_column)

counts = df[target_column].value_counts()
top_n = 5
top_classes = counts.head(top_n).index.tolist()
print(f"\nОсновные классы: {top_classes}")
df[target_column] = df[target_column].apply(lambda x: x if x in top_classes else "Другой")

# Удаляем классы с частотой < 2
counts_new = df[target_column].value_counts()
df = df[df[target_column].map(counts_new) >= 2]
print("\nБаланс после объединения:\n", df[target_column].value_counts())

def bin_age(age):
    if age < 25:
        return "до 25"
    elif age < 35:
        return "25-35"
    else:
        return "35+"

df["Ваш возраст(число)"] = df["Ваш возраст(число)"].apply(bin_age)

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nРазмер X после one-hot:", X.shape)
print("Уникальных классов:", len(encoder.classes_))
print("Распределение классов:", np.bincount(y))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"После PCA: {X_pca.shape[1]} компонент (95% дисперсии)")

models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVC': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

n_splits = 6
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {} 

for name, model in models.items():
    print(f"\n--- Тестируем {name} ---")
    macro_f1_scores = []
    weighted_f1_scores = []
    
    for train_idx, val_idx in skf.split(X_pca, y):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        
        # Обучение
        model_clone = model
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_val)
        
        macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)
    
    results[name] = {
        'macro_f1_mean': np.mean(macro_f1_scores),
        'macro_f1_std': np.std(macro_f1_scores),
        'weighted_f1_mean': np.mean(weighted_f1_scores),
        'weighted_f1_std': np.std(weighted_f1_scores),
        'macro_scores': macro_f1_scores
    }
    print(f"Macro F1: {results[name]['macro_f1_mean']:.3f} ± {results[name]['macro_f1_std']:.3f}")
    print(f"Weighted F1: {results[name]['weighted_f1_mean']:.3f} ± {results[name]['weighted_f1_std']:.3f}")

print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ (Macro F1)")
print("="*60)
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Macro F1 (mean)': [results[m]['macro_f1_mean'] for m in results],
    'Macro F1 (std)': [results[m]['macro_f1_std'] for m in results],
    'Weighted F1 (mean)': [results[m]['weighted_f1_mean'] for m in results]
})
comparison = comparison.sort_values('Macro F1 (mean)', ascending=False)
print(comparison.to_string(index=False))

best_model_name = comparison.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nЛучшая модель: {best_model_name} (Macro F1 = {comparison.iloc[0]['Macro F1 (mean)']:.3f})")

# Обучаем на всех данных с oversampling
ros_full = RandomOverSampler(random_state=42)
X_full_res, y_full_res = ros_full.fit_resample(X_pca, y)
best_model.fit(X_full_res, y_full_res)

# Сохраняем лучшую модель и трансформеры
joblib.dump(best_model, "best_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("Модель и трансформеры сохранены.")

def predict_new_person(person_dict, model, encoder, scaler, pca, feature_columns):
    new_df = pd.DataFrame([person_dict])
    new_df["Ваш возраст(число)"] = new_df["Ваш возраст(число)"].apply(bin_age)
    new_df = pd.get_dummies(new_df)
    for col in feature_columns:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[feature_columns]
    new_scaled = scaler.transform(new_df)
    new_pca = pca.transform(new_scaled)
    pred = model.predict(new_pca)[0]
    prob = model.predict_proba(new_pca)[0]
    pred_name = encoder.inverse_transform([pred])[0]
    return pred_name, prob

# Пример использования
new_person = {
    "Ваш пол": "Женский",
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

name, probs = predict_new_person(new_person, best_model, encoder, scaler, pca, X.columns)
print(f"\nПредсказанный участник: {name}")
print("Вероятности по классам:")
for i, cls in enumerate(encoder.classes_):
    print(f"  {cls}: {probs[i]:.2%}")