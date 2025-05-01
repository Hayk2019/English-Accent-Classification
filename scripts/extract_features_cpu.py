import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# === НАСТРОЙКИ ===
CSV_PATH = "./dataset/all_data.csv"         # CSV с колонками path, accent
FEATURE_DIR = "features"              # Папка для сохранения .npy
NUM_WORKERS = 8                       # Кол-во параллельных процессов

# === ЗАГРУЗКА МОДЕЛИ И ПРОЦЕССОРА ===
print("Загрузка модели...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === ПОДГОТОВКА CSV ===
df = pd.read_csv(CSV_PATH)
os.makedirs(FEATURE_DIR, exist_ok=True)
assert "path" in df.columns and "accent" in df.columns, "CSV должен содержать path и accent"

# === ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ===
def process_file(row, processor, model, device):
    path, label = row["path"], row["accent"]
    if label == "armenian":
        path = "./dataset/armenian_dataset/all/" + path
    elif label == "Spanish":
        path = "./dataset/spanish_dataset/filtered_file/" + path
    else:
        path = "./dataset/indian_german_us_dataset/all_dataset_exept_arm_spnish_whit_newUS/" + path
    try:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(FEATURE_DIR, f"{filename}.npy")
        np.save(out_path, feature)
        return {"filename": f"{filename}.npy", "label": label}
    except Exception as e:
        print(f"Ошибка с файлом {path}: {e}")
        return None

# === ОБЁРТКА ДЛЯ ЗАПУСКА В ПРОЦЕССЕ ===
def worker(row_dict):
    return process_file(row_dict, processor, model, device)

# === ЗАПУСК ПАРАЛЛЕЛЬНО ===
print(f"Извлечение признаков с {NUM_WORKERS} процессами...")
rows = df.to_dict(orient="records")

results = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for result in tqdm(executor.map(worker, rows), total=len(rows)):
        if result:
            results.append(result)

# === СОХРАНЕНИЕ ИТОГОВОГО CSV ===
pd.DataFrame(results).to_csv(os.path.join(FEATURE_DIR, "features.csv"), index=False)
print("Готово! Признаки сохранены.")

