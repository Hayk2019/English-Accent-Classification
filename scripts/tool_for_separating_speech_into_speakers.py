import os
import json
import torchaudio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Audio
from pyannote.core import Segment
from huggingface_hub import login


# Инициализация модели диаризации
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.0")

# Папка с аудиофайлами
input_folder = "./"
output_folder = "diarized_output"
os.makedirs(output_folder, exist_ok=True)

# Функция для разделения аудио по смене говорящего
def split_audio(audio_path, diarization, output_folder):
    waveform, sample_rate = torchaudio.load(audio_path)
    audio = Audio(sample_rate=sample_rate)
    
    segments = []
    prev_speaker = None
    segment_start = None
    
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker != prev_speaker:
            if prev_speaker is not None:
                # Сохраняем предыдущий фрагмент
                segment_end = segment.start
                segment_waveform = waveform[:, int(segment_start * sample_rate) : int(segment_end * sample_rate)]
                output_path = os.path.join(output_folder, f"{prev_speaker}_{segment_start:.2f}_{segment_end:.2f}.wav")
                torchaudio.save(output_path, segment_waveform, sample_rate)
                segments.append({"speaker": prev_speaker, "start": segment_start, "end": segment_end, "file": output_path})
            segment_start = segment.start
        prev_speaker = speaker
    
    # Сохранение разметки
    json_path = os.path.join(output_folder, os.path.basename(audio_path) + "_diarization.json")
    with open(json_path, "w") as f:
        json.dump(segments, f, indent=4)

# Обработка всех файлов
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_folder, filename)
        print(f"Processing {audio_path}...")
        diarization = pipeline(audio_path)
        file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(file_output_folder, exist_ok=True)
        split_audio(audio_path, diarization, file_output_folder)

print("Диаризация и разделение завершены!")

