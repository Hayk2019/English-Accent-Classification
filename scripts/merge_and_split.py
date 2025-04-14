from pydub import AudioSegment
from pathlib import Path
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

def merge_wav_files(wav_files):
    combined = AudioSegment.empty()
    for file_path in wav_files:
        audio = AudioSegment.from_file(file_path)
        combined += audio
    return combined

def export_chunk(index, chunk, output_dir):
    output_file = output_dir / f"chunk_{index:04d}.wav"
    chunk.export(output_file, format="wav")
    return f"✅ Saved: {output_file}"

def split_and_export(audio, chunk_length_ms, output_dir, workers):
    total_chunks = math.ceil(len(audio) / chunk_length_ms)
    print(f"🔪 Splitting into {total_chunks} chunks...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(total_chunks):
            start = i * chunk_length_ms
            end = min(start + chunk_length_ms, len(audio))
            chunk = audio[start:end]
            futures.append(executor.submit(export_chunk, i + 1, chunk, output_dir))

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge WAV files and split into chunks (multithreaded)")
    parser.add_argument("input_dir", help="Directory with input .wav files")
    parser.add_argument("output_dir", help="Directory to save output chunks")
    parser.add_argument("--chunk_length", type=int, default=5000, help="Chunk length in milliseconds (default: 5000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 4)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print("❌ No .wav files found.")
        exit(1)

    print(f"🔄 Merging {len(wav_files)} files...")
    merged_audio = merge_wav_files(wav_files)

    split_and_export(merged_audio, args.chunk_length, output_dir, args.workers)

    print("🎉 Done!")

