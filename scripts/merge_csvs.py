import pandas as pd
import argparse
from pathlib import Path

def merge_csv(csv1_path, csv2_path, output_path, how="inner", on=None):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if on:
        merged_df = pd.merge(df1, df2, how=how, on=on)
    else:
        merged_df = pd.concat([df1, df2], ignore_index=True)

    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files")
    parser.add_argument("csv1", help="Path to the first CSV file")
    parser.add_argument("csv2", help="Path to the second CSV file")
    parser.add_argument("output", help="Path to save merged CSV")
    parser.add_argument("--how", choices=["inner", "outer", "left", "right"], default="inner",
                        help="Type of merge if using key (default: inner)")
    parser.add_argument("--on", help="Column name to merge on (if merging by key)")

    args = parser.parse_args()

    merge_csv(args.csv1, args.csv2, args.output, args.how, args.on)

