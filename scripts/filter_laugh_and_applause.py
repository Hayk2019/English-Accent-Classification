import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from tqdm import tqdm

# Пути к директориям
OUTPUT_DIR = "output_directory"
INVALID_DIR = "invalid_directory"

# Загружаем YAMNet (Google аудио-классификатор)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Метки классов, относящиеся к смеху и аплодисментам
LAUGH_CLASSES = {"Laughter", "Giggle"}
APPLAUSE_CLASSES = {"Applause", "Cheering"}

def load_wav(filepath):
    """Загружает аудиофайл и возвращает его как массив numpy."""
    audio, sr = librosa.load(filepath, sr=16000)
    return audio, sr

def detect_laugh_applause(file_path, segment_duration=0.5):
    """
    Анализирует аудиофайл и проверяет, есть ли в нем смех или аплодисменты.
    Разбивает аудио на фрагменты segment_duration секунд для детального анализа.
    """
    # Загружаем аудио
    audio, sr = load_wav(file_path)

    # Загружаем список классов YAMNet
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = list(map(str.strip, tf.io.gfile.GFile(class_map_path).readlines()))

    # Разбиваем аудио на сегменты (по segment_duration секунд)
    segment_samples = int(segment_duration * sr)
    num_segments = len(audio) // segment_samples

    for i in range(num_segments):
        segment = audio[i * segment_samples : (i + 1) * segment_samples]

        # Пропускаем слишком короткие сегменты
        if len(segment) < segment_samples // 2:
            continue

        # Делаем предсказание через YAMNet
        scores, embeddings, spectrogram = yamnet(segment)

        # Находим вероятные классы для сегмента
        top_classes = np.argmax(scores.numpy(), axis=1)
        detected_classes = {class_names[idx] for idx in top_classes}

        # Проверяем наличие нежелательных звуков
        if detected_classes & (LAUGH_CLASSES | APPLAUSE_CLASSES):
            return True  # Найден смех или аплодисменты
    return False  # Аудио чистое

def process_audio_files():
    """Перемещает файлы со смехом или аплодисментами в INVALID_DIR."""
    if not os.path.exists(INVALID_DIR):
        os.makedirs(INVALID_DIR)

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]

    for filename in tqdm(files, desc="Анализ аудио"):
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Проверяем, есть ли в аудиофайле смех или аплодисменты
        if detect_laugh_applause(file_path):
            shutil.move(file_path, os.path.join(INVALID_DIR, filename))
            print(f"❌ {filename} перемещен в {INVALID_DIR} (обнаружен смех/аплодисменты)")
        else:
            print(f"✅ {filename} чистый, оставлен в {OUTPUT_DIR}")

# Запускаем обработку
process_audio_files()

