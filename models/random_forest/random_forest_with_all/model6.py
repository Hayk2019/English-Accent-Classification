import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Загрузка данных
file_path = "result.csv"
df = pd.read_csv(file_path)

# 2. Признаки и целевая переменная
X = df.iloc[:, 2:]
y = df["accent"]

# 3. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Кросс-валидация
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Average CV Accuracy: {cv_scores.mean():.2f}")
print(f"Loss (1 - accuracy): {(1 - cv_scores.mean()):.2f}")

# Визуализация кросс-валидации
plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_scores * 100, marker='o', label="Accuracy")
plt.axhline(cv_scores.mean() * 100, color='r', linestyle='--', label="Average Accuracy")
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Финальное обучение и предсказания
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Accuracy и отчёт
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

