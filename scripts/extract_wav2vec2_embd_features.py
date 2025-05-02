import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

CSV_PATH = "british_new.csv"  
FEATURE_DIR = "features"      
os.makedirs(FEATURE_DIR, exist_ok=True)


df = pd.read_csv(CSV_PATH)
assert "path" in df.columns and "accent" in df.columns, "CSV must be consist of path и accent columns"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    path = row["path"]
    label = row["accent"]
    if label == "armenian":
        path = "./dataset/armenian_dataset/all/" + path
    elif label == "Spanish":
        path = "./dataset/spanish_dataset/filtered_file/" + path
    else:
        path = "./welsh_english_male/" + path
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
        np.save(os.path.join(FEATURE_DIR, f"{filename}.npy"), feature)
        data.append({"filename": f"{filename}.npy", "label": label})

    except Exception as e:
        print(f"Error with file: {path}: {e}")

pd.DataFrame(data).to_csv(os.path.join(FEATURE_DIR, "features.csv"), index=False)
print("Done.")

