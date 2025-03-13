from preprocessing import PreprocessingPipeline
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate YAML config with column-specific encoders.")
parser.add_argument("--input_csv", required=True, help="Path to the CSV file containing raw features.")
parser.add_argument("--yaml_config", required=True, help="Path to the CSV file containing raw features.")

args = parser.parse_args()
df = pd.read_csv(args.input_csv)

pipeline = PreprocessingPipeline(config_path=args.yaml_config)
X_processed = pipeline.fit_transform(df, y="BID")
print(type(X_processed))
print(X_processed.head() if hasattr(X_processed, "head") else X_processed)
X_processed.to_csv("processed_data.csv", index=False)
print(X_processed.columns)