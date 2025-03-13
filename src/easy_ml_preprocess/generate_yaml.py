import argparse
import yaml
import pandas as pd
import numpy as np
import os

def main(args):
    
    # -------------------------------------------------------------------------
    # 1. Load the CSV data to infer numeric vs. categorical columns
    # -------------------------------------------------------------------------
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"]).columns.tolist()

    # -------------------------------------------------------------------------
    # 2. Load a base YAML if provided; otherwise, start with an empty config
    # -------------------------------------------------------------------------
    config = {}
    if args.base_yaml and os.path.exists(args.base_yaml):
        with open(args.base_yaml, "r") as f:
            base_config = yaml.safe_load(f)
            if base_config is not None:
                config = base_config

    if "preprocessing" not in config:
        config["preprocessing"] = {}

    # -------------------------------------------------------------------------
    # 3. Build column-specific blocks for numeric and categorical
    # -------------------------------------------------------------------------
    # We will create a structure like:
    #
    #  preprocessing:
    #    numeric:
    #      columns:
    #        age:
    #          imputer_strategy: mean
    #          scaler: StandardScaler
    #        income:
    #          imputer_strategy: mean
    #          scaler: MinMaxScaler
    #    categorical:
    #      columns:
    #        gender:
    #          imputer_strategy: most_frequent
    #          encoder: OneHotEncoder
    #        city:
    #          imputer_strategy: most_frequent
    #          encoder: OrdinalEncoder
    #
    # You can adjust the default strategies as you see fit.

    # Numeric sub-block
    if "numeric" not in config["preprocessing"]:
        config["preprocessing"]["numeric"] = {}
    if "columns" not in config["preprocessing"]["numeric"]:
        config["preprocessing"]["numeric"]["columns"] = {}

    for col in numeric_columns:
        # Only set if not already defined (so we don't overwrite user-defined settings in base YAML).
        if col not in config["preprocessing"]["numeric"]["columns"]:
            config["preprocessing"]["numeric"]["columns"][col] = {
                "imputer_strategy": "mean",
                "scaler": "StandardScaler"  # or "MinMaxScaler", "RobustScaler", etc.
            }

    # Categorical sub-block
    if "categorical" not in config["preprocessing"]:
        config["preprocessing"]["categorical"] = {}
    if "columns" not in config["preprocessing"]["categorical"]:
        config["preprocessing"]["categorical"]["columns"] = {}

    cat_columns_config = config["preprocessing"]["categorical"]["columns"]
    for col in categorical_columns:
        if col not in cat_columns_config:
            cat_columns_config[col] = {
                "imputer_strategy": "most_frequent",
                "encoder": "OneHotEncoder"  # or "OrdinalEncoder", etc.
            }

    # -------------------------------------------------------------------------
    # 4. Set the output_type in the config
    # -------------------------------------------------------------------------
    config["output_type"] = args.output_type

    # -------------------------------------------------------------------------
    # 5. Write the final config to the specified output YAML
    # -------------------------------------------------------------------------
    with open(args.output_yaml, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"YAML config file generated at: {args.output_yaml}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML config with column-specific encoders.")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the CSV file containing raw features."
    )
    parser.add_argument(
        "--output_yaml",
        required=True,
        help="Path to the output YAML file to be generated."
    )
    parser.add_argument(
        "--base_yaml",
        required=False,
        help="Optional path to a base YAML file. If provided, this script will load "
             "that YAML first, then override or add settings based on the CSV."
    )
    parser.add_argument(
        "--output_type",
        choices=["pandas", "torch"],
        default="pandas",
        help="Desired output data format in the generated config (default: pandas)."
    )
    # Optionally, you could also let users provide defaults for numeric/categorical
    # transformations, but here we hardcode typical defaults.
    args = parser.parse_args()
    main(args)
