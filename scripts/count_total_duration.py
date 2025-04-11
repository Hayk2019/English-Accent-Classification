import wave
import contextlib
import os

directory = "../spanish_dataset/filtered_file"
total_duration = 0.0

for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(directory, filename)
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            total_duration += duration

print(f"Total duration of spanish_dataset: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")

