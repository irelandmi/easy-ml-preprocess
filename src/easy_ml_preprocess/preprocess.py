from easy_ml_preprocess.preprocessing import PreprocessingPipeline
import pandas as pd
import argparse


def main(args):
    df = pd.read_csv(args.input_csv)
    pipeline = PreprocessingPipeline(config_path=args.yaml_config)
    X_processed = pipeline.fit_transform(df)
    print(type(X_processed))
    print(X_processed.head() if hasattr(X_processed, "head") else X_processed)
    X_processed.to_csv(args.output_csv, index=False)
    print(X_processed.columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML config with column-specific encoders.")
    parser.add_argument("--input_csv", required=True, help="Path to the CSV file containing raw features.")
    parser.add_argument("--yaml_config", required=True, help="Path to the CSV file containing raw features.")
    parser.add_argument("--output_csv", type=str, help="Output filepath and name for the data")
    args = parser.parse_args()
    main(args)