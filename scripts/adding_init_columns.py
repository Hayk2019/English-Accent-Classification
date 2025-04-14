import os
import pandas as pd
from pathlib import Path
import argparse

def generate_csv_from_files(input_dir, output_csv, accent_label="armenian"):
    input_dir = Path(input_dir)
    file_list = [f.name for f in input_dir.iterdir() if f.is_file()]

    data = {
        "path": file_list,
        "accent": [accent_label] * len(file_list)
    }

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved: {output_csv} ({len(file_list)} files)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV with file names and accent label")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_csv", help="Output CSV file name")
    parser.add_argument("--accent", default="armenian", help="Accent label to use (default: armenian)")
    args = parser.parse_args()

    generate_csv_from_files(args.input_dir, args.output_csv, args.accent)

