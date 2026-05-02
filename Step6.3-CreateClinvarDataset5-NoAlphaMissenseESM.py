#!/usr/bin/env python3
"""
Create Clinvar_Dataset5 for No AlphaMissense / No ESM ablation.

Input:
    Clinvar_Dataset2/Fold_*/X_train_processed.csv
    Clinvar_Dataset2/Fold_*/X_test_processed.csv
    Clinvar_Dataset2/Fold_*/Y_train.csv
    Clinvar_Dataset2/Fold_*/Y_test.csv

Output:
    Clinvar_Dataset5/Fold_*/X_train_processed.csv
    Clinvar_Dataset5/Fold_*/X_test_processed.csv
    Clinvar_Dataset5/Fold_*/Y_train.csv
    Clinvar_Dataset5/Fold_*/Y_test.csv

Purpose:
    Remove AlphaMissense and ESM-derived features while keeping the same fold structure.
"""

from pathlib import Path
import pandas as pd
import shutil
import json


INPUT_DIR = Path("Clinvar_Dataset2")
OUTPUT_DIR = Path("Clinvar_Dataset5")

DROP_KEYWORDS = [
    # AlphaMissense
    "alphamissense",
    "alpha_missense",
    "alpha-missense",
    "alpha missense",
    "am_pathogenicity",
    "am_class",
    "am_score",

    # ESM
    "esm",
    "esm1v",
    "esm_1v",
    "esm2",
    "esm_2",
    "esm_score",
    "esm-score",
]


def should_drop_column(column_name: str) -> bool:
    col = str(column_name).lower()
    return any(keyword in col for keyword in DROP_KEYWORDS)


def process_fold(fold_dir: Path, output_fold_dir: Path):
    fold_name = fold_dir.name
    print(f"\nProcessing {fold_name}")

    x_train_path = fold_dir / "X_train_processed.csv"
    x_test_path = fold_dir / "X_test_processed.csv"
    y_train_path = fold_dir / "Y_train.csv"
    y_test_path = fold_dir / "Y_test.csv"

    required_files = [
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
    ]

    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        print(f"Skipping {fold_name}; missing files:")
        for p in missing_files:
            print(f"  - {p}")
        return None

    output_fold_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    common_features = list(set(X_train.columns) & set(X_test.columns))

    drop_cols = [
        col for col in common_features
        if should_drop_column(col)
    ]

    keep_cols = [
        col for col in common_features
        if col not in drop_cols
    ]

    X_train_filtered = X_train[keep_cols].copy()
    X_test_filtered = X_test[keep_cols].copy()

    X_train_filtered.to_csv(output_fold_dir / "X_train_processed.csv", index=False)
    X_test_filtered.to_csv(output_fold_dir / "X_test_processed.csv", index=False)

    shutil.copy2(y_train_path, output_fold_dir / "Y_train.csv")
    shutil.copy2(y_test_path, output_fold_dir / "Y_test.csv")

    print(f"  Original train shape: {X_train.shape}")
    print(f"  Original test shape:  {X_test.shape}")
    print(f"  Dropped columns:      {len(drop_cols)}")
    print(f"  New train shape:      {X_train_filtered.shape}")
    print(f"  New test shape:       {X_test_filtered.shape}")
    print(f"  Saved to:             {output_fold_dir}")

    if drop_cols:
        print("  Example dropped columns:")
        for col in drop_cols[:30]:
            print(f"    - {col}")

    return {
        "fold": fold_name,
        "input_train_shape": list(X_train.shape),
        "input_test_shape": list(X_test.shape),
        "output_train_shape": list(X_train_filtered.shape),
        "output_test_shape": list(X_test_filtered.shape),
        "num_original_features": len(common_features),
        "num_dropped_columns": len(drop_cols),
        "num_kept_columns": len(keep_cols),
        "dropped_columns": drop_cols,
        "kept_columns": keep_cols,
    }


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_dirs = [
        d for d in INPUT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("Fold_")
    ]

    fold_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

    if not fold_dirs:
        raise FileNotFoundError(f"No Fold_* directories found in {INPUT_DIR}")

    print("=" * 80)
    print("Creating Clinvar_Dataset5: No AlphaMissense / No ESM")
    print("=" * 80)
    print(f"Input dataset:  {INPUT_DIR}")
    print(f"Output dataset: {OUTPUT_DIR}")
    print(f"Folds found:    {[d.name for d in fold_dirs]}")

    report = {}

    for fold_dir in fold_dirs:
        output_fold_dir = OUTPUT_DIR / fold_dir.name
        fold_report = process_fold(fold_dir, output_fold_dir)

        if fold_report is not None:
            report[fold_dir.name] = fold_report

    report_path = OUTPUT_DIR / "NoAlphaMissenseESM_feature_filter_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Filtered dataset saved in: {OUTPUT_DIR}")
    print(f"Filter report saved to:    {report_path}")

    total_dropped = {
        fold: data["num_dropped_columns"]
        for fold, data in report.items()
    }

    print("\nDropped-column count by fold:")
    for fold, count in total_dropped.items():
        print(f"  {fold}: {count}")


if __name__ == "__main__":
    main()