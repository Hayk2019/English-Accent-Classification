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

INPUT_FOLDER = "spliced"
OUTPUT_FOLDER = "home/hayk/Documents/English-Accent-Classification/armenian_dataset/telegram_capture/processed"
INVALID_FOLDER = "home/hayk/Documents/English-Accent-Classification/armenian_dataset/telegram_capture/invalid"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)


def process_audio(file_path):
    try:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".mp3", ".wav"))

        print(f"[INFO] First stage: {file_name}")

        if file_path.endswith(".mp3"):
            print(f"[INFO] Collecting in WAV: {file_name}")
            audio = AudioSegment.from_mp3(file_path)
            file_path = output_path
            audio.export(file_path, format="wav")

        print(f"[INFO] Load audio: {file_name}")
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        print(f"[INFO] Volume normalizing: {file_name}")
        y = y / np.max(np.abs(y))

        print(f"[INFO] silence: {file_name}")
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        print(f"[INFO] noise cancelation: {file_name}")
        y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr, prop_decrease=1.0)

        print(f"[INFO] Saving files: {file_name}")
        sf.write(output_path, y_denoised, sr)

        print(f"[INFO] processing done: {file_name}\n")
        return output_path 

    except Exception as e:
        print(f"[ERROR] Processing error{file_path}: {e}")
        return None


def check_audio_quality(file_path):
    try:
        file_name = os.path.basename(file_path)
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1.0:
            print(f"[WARNING] file {file_name} is to short ({duration:.2f} сек.), moved to{INVALID_FOLDER}")
            shutil.move(file_path, os.path.join(INVALID_FOLDER, file_name))
            return None

        energy = np.sum(y ** 2) / len(y)
        if energy < 1e-6:
            print(f"[WARNING] file {file_name} is to silenced ({energy :.8f}), moved to {INVALID_FOLDER}")
            shutil.move(file_path, os.path.join(INVALID_FOLDER, file_name))
            return None

        return file_path

    except Exception as e:
        print(f"[ERROR] ERROR during checking {file_path}: {e}")
        return None


def main():
    audio_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith((".wav", ".mp3"))]

    print(f"[INFO] Fing {len(audio_files)} files fro processing.\n")

    processed_files = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        processed_files = list(tqdm(executor.map(process_audio, audio_files), total=len(audio_files)))

    processed_files = [f for f in processed_files if f is not None]

    print("\n[INFO] All files processed. Testing quality...\n")
    valid_files = []
    for file_path in tqdm(processed_files):
        if file_path:
            result = check_audio_quality(file_path)
            if result:
                valid_files.append(result)

    print(f"\n[INFO] Testing Done. Tested files: {len(valid_files)}, move to {INVALID_FOLDER}: {len(processed_files) - len(valid_files)}\n")

if __name__ == "__main__":
    main()
