from pydub import AudioSegment, silence
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def remove_silence_file(file_path_str, output_dir_str, threshold, min_len, keep):
    try:
        file_path = Path(file_path_str)
        output_dir = Path(output_dir_str)
        output_path = output_dir / (file_path.stem + "_cleaned.wav")

        audio = AudioSegment.from_file(file_path)

        non_silent_chunks = silence.split_on_silence(
            audio,
            min_silence_len=min_len,
            silence_thresh=threshold,
            keep_silence=keep
        )

        processed_audio = AudioSegment.empty()
        for chunk in non_silent_chunks:
            processed_audio += chunk

        processed_audio.export(output_path, format="wav")
        return f"Saved: {output_path}"
    except Exception as e:
        return f"Error processing {file_path_str}: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove silence from audio files using multiprocessing.")
    parser.add_argument("input_dir", help="Directory with input audio files")
    parser.add_argument("output_dir", help="Directory to save output audio files")
    parser.add_argument("--threshold", type=int, default=-40, help="Silence threshold in dBFS (default: -40)")
    parser.add_argument("--min_len", type=int, default=500, help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--keep", type=int, default=0, help="Milliseconds of silence to keep around chunks (default: 0)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes (default: 4)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = [".wav", ".mp3", ".ogg", ".aac", ".flac"]
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_exts]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(remove_silence_file, str(f), str(output_dir), args.threshold, args.min_len, args.keep)
            for f in files
        ]
        for future in as_completed(futures):
            print(future.result())

