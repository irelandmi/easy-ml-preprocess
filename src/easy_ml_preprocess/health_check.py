#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Perform basic health checks on a CSV dataset.")
    parser.add_argument(
        "--input_csv", 
        required=True, 
        help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--output_report",
        required=False,
        help="Path to the output report file. If not specified, prints to console."
    )
    args = parser.parse_args()

    # 1. Ensure input CSV exists
    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Could not find file: {args.input_csv}")

    # 2. Load the CSV into a pandas DataFrame
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}")

    # 3. Generate the health checks
    report = []

    # -------------------------------------------------------------------------
    # A) Basic Info
    # -------------------------------------------------------------------------
    report.append("========== DATA HEALTH REPORT ==========\n")
    report.append(f"FILE: {args.input_csv}\n")
    report.append(f"NUMBER OF ROWS: {df.shape[0]}\n")
    report.append(f"NUMBER OF COLUMNS: {df.shape[1]}\n\n")

    # -------------------------------------------------------------------------
    # B) Missing Values
    # -------------------------------------------------------------------------
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    report.append("MISSING VALUES (Count | Percentage):\n")
    for col in df.columns:
        if missing_counts[col] > 0:
            msg = f"  {col}: {missing_counts[col]} ({missing_percentages[col]:.2f}%)"
        else:
            msg = f"  {col}: 0"
        report.append(msg)
    report.append("")

    # -------------------------------------------------------------------------
    # C) Duplicate Rows
    # -------------------------------------------------------------------------
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report.append(f"DUPLICATE ROWS: {duplicate_count} (Consider dropping them)\n")
    else:
        report.append("DUPLICATE ROWS: None detected\n")

    # -------------------------------------------------------------------------
    # D) Data Types Overview
    # -------------------------------------------------------------------------
    report.append("\nDATA TYPES:\n")
    dtype_info = df.dtypes.astype(str)
    for col in df.columns:
        report.append(f"  {col}: {dtype_info[col]}")
    report.append("")

    # -------------------------------------------------------------------------
    # E) Numeric Checks (Outliers, Basic Stats)
    # -------------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report.append("NUMERIC COLUMNS STATISTICS:\n")
        describe_numeric = df[numeric_cols].describe()
        report.append(str(describe_numeric))
        report.append("")

        # (Optional) Quick outlier check using IQR
        # (You might refine or add thresholds for “severe outliers”.)
        outlier_report = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            if len(outliers) > 0:
                outlier_report.append(f"  {col}: {len(outliers)} outliers detected")

        if outlier_report:
            report.append("OUTLIER DETECTION (IQR method):\n" + "\n".join(outlier_report) + "\n")
        else:
            report.append("OUTLIER DETECTION: None detected by IQR method\n")

    else:
        report.append("NO NUMERIC COLUMNS DETECTED.\n")

    # -------------------------------------------------------------------------
    # F) Categorical Checks (Cardinality, Value Counts)
    # -------------------------------------------------------------------------
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        report.append("CATEGORICAL COLUMNS SUMMARY:\n")
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            card = len(unique_vals)
            report.append(f"  {col} -> cardinality: {card}")
            # For debugging, you might also want top 5 frequent categories:
            # freq = df[col].value_counts().head(5).to_dict()
            # report.append(f"     top categories: {freq}")
        report.append("")
    else:
        report.append("NO CATEGORICAL COLUMNS DETECTED.\n")

    # -------------------------------------------------------------------------
    # G) Constant or Near-Constant Columns
    # -------------------------------------------------------------------------
    # Sometimes columns with zero variance can be dropped
    near_constant_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(dropna=False).max()
        ratio = top_freq / len(df)
        if ratio > 0.99:  # e.g., if >99% of values are the same
            near_constant_cols.append(col)

    if near_constant_cols:
        report.append("NEAR-CONSTANT / LOW-VARIANCE COLUMNS (over 99% identical values):\n")
        for col in near_constant_cols:
            report.append(f"  {col}")
        report.append("")
    else:
        report.append("NO NEAR-CONSTANT COLUMNS DETECTED.\n")

    # -------------------------------------------------------------------------
    # H) Correlation Check (Optional)
    # -------------------------------------------------------------------------
    # Simple approach: only for numeric columns. In practice, you might want a more advanced check.
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        report.append("CORRELATION MATRIX (Numeric Columns):\n")
        report.append(str(corr_matrix))
        report.append("")
    else:
        report.append("NOT ENOUGH NUMERIC COLUMNS FOR CORRELATION ANALYSIS.\n")

    # -------------------------------------------------------------------------
    # Print or Save Report
    # -------------------------------------------------------------------------
    final_report = "\n".join(report)

    if args.output_report:
        try:
            with open(args.output_report, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"Health check report saved to: {args.output_report}")
        except Exception as e:
            print(f"Could not write to report file {args.output_report}: {e}")
            print("Printing report to console instead:\n")
            print(final_report)
    else:
        print(final_report)

if __name__ == "__main__":
    main()
