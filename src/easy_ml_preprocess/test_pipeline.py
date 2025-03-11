from preprocessing import PreprocessingPipeline
import pandas as pd

df = pd.read_csv("../data/btc_usd_training.csv")

pipeline = PreprocessingPipeline(config_path="../config/test_config.yaml")
X_processed = pipeline.fit_transform(df)
print(type(X_processed))
print(X_processed.head() if hasattr(X_processed, "head") else X_processed)
