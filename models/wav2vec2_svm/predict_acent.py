import os
import glob
import librosa
import numpy as np
import joblib
import soundfile as sf
import torchaudio
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
from sklearn.svm import SVC

AUDIO_DIR = "input_audio"
MODEL_PATH = "svc_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
SAMPLE_RATE = 16000
TEMP_WAV_DIR = "converted_wav"

os.makedirs(TEMP_WAV_DIR, exist_ok=True)

clf: SVC = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()

def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")
        return None

def extract_features(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    inputs = feature_extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1)
    return embedding.numpy()

audio_paths = glob.glob(os.path.join(AUDIO_DIR, "*.*"))
if not audio_paths:
    print("No audio files found in", AUDIO_DIR)
    exit()

print(f"Found {len(audio_paths)} file(s). Converting, extracting and predicting...\n")

for path in audio_paths:
    filename = os.path.splitext(os.path.basename(path))[0]
    temp_wav = os.path.join(TEMP_WAV_DIR, filename + ".wav")
    converted = convert_to_wav(path, temp_wav)
    if converted is None:
        continue
    try:
        features = extract_features(converted)
        pred = clf.predict(features)
        label = le.inverse_transform(pred)[0]
        print(f"{os.path.basename(path)} → {label}")
    except Exception as e:
        print(f"Error processing {path}: {e}")

