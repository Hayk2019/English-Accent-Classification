import os
import shutil
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# === Папки ===
INPUT_FOLDER = "audio_files"       # Исходные файлы
OUTPUT_FOLDER = "filtered_file"   # Обработанные файлы
INVALID_FOLDER = "invalid_audio"    # Проблемные файлы

# === Создаем выходные папки, если их нет ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)


# === Функция обработки одного аудиофайла ===
def process_audio(file_path):
    try:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".mp3", ".wav"))

        print(f"[INFO] Начало обработки: {file_name}")

        # === Конвертация в WAV, если это MP3 ===
        if file_path.endswith(".mp3"):
            print(f"[INFO] Конвертация в WAV: {file_name}")
            audio = AudioSegment.from_mp3(file_path)
            file_path = output_path
            audio.export(file_path, format="wav")

        # === Загрузка аудиофайла ===
        print(f"[INFO] Загрузка аудио: {file_name}")
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # === Нормализация громкости ===
        print(f"[INFO] Нормализация громкости: {file_name}")
        y = y / np.max(np.abs(y))

        # === Обрезка тишины ===
        print(f"[INFO] Обрезка тишины: {file_name}")
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # === Шумоподавление ===
        print(f"[INFO] Шумоподавление: {file_name}")
        y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr, prop_decrease=1.0)

        # === Сохранение обработанного файла ===
        print(f"[INFO] Сохранение обработанного файла: {file_name}")
        sf.write(output_path, y_denoised, sr)

        print(f"[INFO] Обработка завершена: {file_name}\n")
        return output_path  # Возвращаем путь к обработанному файлу

    except Exception as e:
        print(f"[ERROR] Ошибка обработки {file_path}: {e}")
        return None


# === Функция проверки качества ===
def check_audio_quality(file_path):
    try:
        file_name = os.path.basename(file_path)
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # === Проверка длительности ===
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1.0:
            print(f"[WARNING] Файл {file_name} слишком короткий ({duration:.2f} сек.), перемещаем в {INVALID_FOLDER}")
            shutil.move(file_path, os.path.join(INVALID_FOLDER, file_name))
            return None

        # === Проверка громкости (энергии) ===
        energy = np.sum(y ** 2) / len(y)
        if energy < 1e-6:
            print(f"[WARNING] Файл {file_name} слишком тихий (энергия {energy:.8f}), перемещаем в {INVALID_FOLDER}")
            shutil.move(file_path, os.path.join(INVALID_FOLDER, file_name))
            return None

        return file_path  # Если всё в порядке, возвращаем файл

    except Exception as e:
        print(f"[ERROR] Ошибка при проверке {file_path}: {e}")
        return None


# === Главная функция ===
def main():
    # === Получаем список файлов ===
    audio_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith((".wav", ".mp3"))]

    print(f"[INFO] Найдено {len(audio_files)} файлов для обработки.\n")

    # === Многопоточная обработка файлов ===
    processed_files = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed_files = list(tqdm(executor.map(process_audio, audio_files), total=len(audio_files)))

    # Удаляем None (ошибки обработки)
    processed_files = [f for f in processed_files if f is not None]

    print("\n[INFO] Все файлы обработаны. Начинаем проверку качества...\n")
    # === Однопоточная проверка качества файлов ===
    valid_files = []
    for file_path in tqdm(processed_files):
        if file_path:
            result = check_audio_quality(file_path)
            if result:
                valid_files.append(result)

    print(f"\n[INFO] Проверка завершена. Принято файлов: {len(valid_files)}, перемещено в {INVALID_FOLDER}: {len(processed_files) - len(valid_files)}\n")

    # === Визуализация первых 5 хороших файлов ===
    print("[INFO] Начинаем визуализацию...\n")
    for i, file_path in enumerate(valid_files[:5]):
        file_name = os.path.basename(file_path)
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform: {file_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    print("[INFO] Визуализация завершена.")

if __name__ == "__main__":
    main()
