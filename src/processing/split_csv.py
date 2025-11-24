# src/processing/split_dataset.py

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


def show_distribution(label_series, name):
    counter = Counter(label_series)
    total = sum(counter.values())
    print(f"\nDistribution for {name}:")
    for k, v in sorted(counter.items()):
        print(f"  {k}: {v} ({v/total:.4f})")


def stratified_split(df, n_samples, seed=42):
    y = df["answer"]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=len(df) - n_samples,
        random_state=seed
    )

    for idx_small, idx_large in splitter.split(df, y):
        df_small = df.iloc[idx_small].reset_index(drop=True)
        df_large = df.iloc[idx_large].reset_index(drop=True)
        return df_small, df_large

    raise RuntimeError("Stratified split failed unexpectedly.")


def main():
    parser = argparse.ArgumentParser(description="Stratified quantity-based CSV splitter")
    parser.add_argument("--input", type=str, help="Path to input CSV file",default=r'data/raw/6k/6000_train_examples.csv')
    parser.add_argument("--n", type=int, help="Number of rows for the first split",default= 800)
    parser.add_argument("--outdir", type=str, help="Directory to save outputs",default=r'data/raw/6k')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    if args.n <= 0 or args.n >= total_rows:
        raise ValueError(f"Invalid n={args.n}. Must be between 1 and {total_rows-1}")

    print("\nOriginal answer distribution:")
    show_distribution(df["answer"], "Original")

    # Perform stratified split
    print(f"\nPerforming stratified split: first={args.n}, second={total_rows - args.n}")
    df_small, df_large = stratified_split(df, args.n)

    print("\nSplit completed.")
    show_distribution(df_small["answer"], "Split 1")
    show_distribution(df_large["answer"], "Split 2")

    # Save files
    out1 = output_dir / "part_1.csv"
    out2 = output_dir / "part_2.csv"

    df_small.to_csv(out1, index=False)
    df_large.to_csv(out2, index=False)

    print(f"\nSaved:")
    print(f"  {out1}  ({len(df_small)} rows)")
    print(f"  {out2}  ({len(df_large)} rows)")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
