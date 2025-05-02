import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Extract audio features from WAV files.")
parser.add_argument("--audio_folder", type=str, required=True, help="Folder with processed audio files")
parser.add_argument("--csv_file", type=str, required=True, help="CSV file with metadata (columns: path, accent )")
parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output features CSV")

args = parser.parse_args()

AUDIO_FOLDER = args.audio_folder
CSV_FILE = args.csv_file
OUTPUT_CSV = args.output_csv

df = pd.read_csv(CSV_FILE)

existing_files = set(os.listdir(AUDIO_FOLDER))
df = df[df["path"].isin(existing_files)]  # Оставляем только файлы, которые есть в папке

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel)
        mel_std = np.std(mel)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Combine features
        features = np.hstack([mfcc_mean, mfcc_std, mel_mean, mel_std, chroma_mean, chroma_std, zcr_mean, zcr_std, rms_mean, rms_std])
        return features

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return None

feature_columns = [f"mfcc_{i}_mean" for i in range(13)] + \
                  [f"mfcc_{i}_std" for i in range(13)] + \
                  ["mel_mean", "mel_std", "chroma_mean", "chroma_std", "zcr_mean", "zcr_std", "rms_mean", "rms_std"]

feature_data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    file_path = os.path.join(AUDIO_FOLDER, row["path"])
    features = extract_features(file_path)
    if features is not None:
        feature_data.append([row["path"], row["accent"]] + list(features))

df_features = pd.DataFrame(feature_data, columns=["path", "accent"] + feature_columns)
df_features.to_csv(OUTPUT_CSV, index=False)

print(f"[INFO] Feature extraction completed! File saved: {OUTPUT_CSV}")

