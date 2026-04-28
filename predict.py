from __future__ import annotations

"""CLI convenience script: run predictions on a file and print results."""

import argparse
import sys

import pandas as pd

from sap.features import build_term_features, clean_data
from sap.io import load_table, prepare
from sap.model import infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions on a student data file"
    )
    parser.add_argument("--input",  required=True, help="CSV or Excel file")
    parser.add_argument("--output", default="",    help="Optional output CSV path")
    parser.add_argument("--threshold-low",    type=float, default=0.75)
    parser.add_argument("--threshold-medium", type=float, default=0.50)
    args = parser.parse_args()

    df      = load_table(args.input)
    prepared = prepare(df, mode="predict")
    clean   = clean_data(prepared)
    term_df = build_term_features(clean)

    thresholds = {"low": args.threshold_low, "medium": args.threshold_medium}
    preds = infer(term_df, thresholds=thresholds)

    print(preds[["Student ID", "Student Name", "Term", "Class",
                 "predicted_outcome", "risk_tier",
                 "pass_probability"]].to_string(index=False))

    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
