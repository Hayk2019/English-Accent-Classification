import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === Настройки ===
csv_path = "../dataset/all_data.csv"
sr = 16000                  # частота дискретизации
n_mels = 128                # количество мел-фильтров
max_len = 500               # длина спектрограммы (в кадрах)

# === Функция спектрограммы фиксированной длины ===
def audio_to_mel(file_path, sr=16000, n_mels=128, max_len=500):
    y, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Приводим к фиксированной длине
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]

    return mel_db

# === Загрузка и обработка ===
df = pd.read_csv(csv_path)
X = []
y = []

for idx, row in df.iterrows():
    label = row["accent"]
    if label == "armenian":
        path = os.path.join("../dataset/armenian_dataset/all/",  row["path"])
    elif label == "Spanish":
        path = os.path.join("../dataset/spanish_dataset/filtered_file/",  row["path"])
    else:
        path = os.path.join("../dataset/indian_german_us_dataset/all_dataset_exept_arm_spnish_whit_newUS/",  row["path"])
    try:
        mel = audio_to_mel(path, sr=sr, n_mels=n_mels, max_len=max_len)
        X.append(mel)
        y.append(label)
    except Exception as e:
        print(f"❌ Ошибка с {path}: {e}")

# === Преобразуем в массивы ===
X = np.array(X)[..., np.newaxis]  # добавляем канал для CNN
print("✅ X shape:", X.shape)

# === Кодируем метки ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)
print("✅ y shape:", y_cat.shape)

# === Сохраняем ===
np.save("X_melspec.npy", X)
np.save("y_onehot.npy", y_cat)
np.save("label_classes.npy", encoder.classes_)
print(np.load("label_classes.npy"))
print("✅ Датасет сохранён!")

