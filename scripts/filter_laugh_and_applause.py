import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from tqdm import tqdm

OUTPUT_DIR = "output_directory"
INVALID_DIR = "invalid_directory"

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

LAUGH_CLASSES = {"Laughter", "Giggle"}
APPLAUSE_CLASSES = {"Applause", "Cheering"}

def load_wav(filepath):
    audio, sr = librosa.load(filepath, sr=16000)
    return audio, sr

def detect_laugh_applause(file_path, segment_duration=0.5):
    audio, sr = load_wav(file_path)

    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = list(map(str.strip, tf.io.gfile.GFile(class_map_path).readlines()))

    segment_samples = int(segment_duration * sr)
    num_segments = len(audio) // segment_samples

    for i in range(num_segments):
        segment = audio[i * segment_samples : (i + 1) * segment_samples]

        if len(segment) < segment_samples // 2:
            continue

        scores, embeddings, spectrogram = yamnet(segment)

        top_classes = np.argmax(scores.numpy(), axis=1)
        detected_classes = {class_names[idx] for idx in top_classes}

        if detected_classes & (LAUGH_CLASSES | APPLAUSE_CLASSES):
            return True
    return False 

def process_audio_files():
    if not os.path.exists(INVALID_DIR):
        os.makedirs(INVALID_DIR)

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]

    for filename in tqdm(files, desc="Analyze audio"):
        file_path = os.path.join(OUTPUT_DIR, filename)

        if detect_laugh_applause(file_path):
            shutil.move(file_path, os.path.join(INVALID_DIR, filename))
            print(f" {filename} moved to {INVALID_DIR} (have an applauses)")
        else:
            print(f" {filename} filtered, saved from {OUTPUT_DIR}")

process_audio_files()

