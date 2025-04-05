import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка данных
file_path = "features.csv"  # Укажи правильный путь к файлу
df = pd.read_csv(file_path)


# 3. Разделение данных на признаки и метки
X = df.iloc[:, 3:]  # Числовые признаки
y = df["accent"]  # Целевая переменная

# 4. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Предсказания и оценка
y_pred = model.predict(X_test)

# Вычисление accuracy в процентах
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracy:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
