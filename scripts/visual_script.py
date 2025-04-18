import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv("../random_forest_with_all/result.csv")  # Замените на путь к вашему файлу

# График 1 — Распределение акцентов
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="accent", order=df["accent"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Распределение акцентов")
plt.tight_layout()
plt.savefig("accent_distribution.png")
plt.close()

# График 2 — Количество аудиофайлов по акценту
accent_counts = df['accent'].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=accent_counts.index, y=accent_counts.values)
plt.xticks(rotation=45)
plt.title("Number of audiofiles by accent")
plt.ylabel("Count")
plt.xlabel("Accent")
plt.tight_layout()
plt.savefig("accent_file_counts.png")
plt.close()

