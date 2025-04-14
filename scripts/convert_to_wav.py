import os
import subprocess
from pathlib import Path
import argparse

def convert_ogg_to_wav(input_folder, output_folder=None,input_format="m4a"):
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve() if output_folder else input_folder

    output_folder.mkdir(parents=True, exist_ok=True)
    input_format = "*."+input_format;
    print(input_format)
    for ogg_file in input_folder.glob(input_format):
        wav_file = output_folder / (ogg_file.stem + ".wav")
        print(f"🔁 Converting:\n  📥 {ogg_file.resolve()}\n  📤 {wav_file.resolve()}")
        
        command = [
            "ffmpeg",
            "-y",
            "-i", str(ogg_file),
            str(wav_file)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ All files have been converted to .wav.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ogg files to .wav using ffmpeg.")
    parser.add_argument("input_folder", help="Path to the folder containing .ogg files")
    parser.add_argument("output_folder", nargs="?", default=None, help="Optional path to save .wav files")
    parser.add_argument("input_format", nargs="?", default="m4a", help="Optional input files format that be converted")
    args = parser.parse_args()
    print(args) 
    convert_ogg_to_wav(args.input_folder, args.output_folder,args.input_format)

