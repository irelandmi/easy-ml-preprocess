import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# 1. Custom wrapper for LabelEncoder
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """Wraps LabelEncoder so it can be used on a single column within a Pipeline."""
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        self.encoder.fit(X.astype(str))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        encoded = self.encoder.transform(X.astype(str))
        return encoded.reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature name(s)."""
        # If it's only one column in → one column out, you can safely return the same name.
        # Alternatively, append '_encoded' or something to clarify that it's transformed.
        if input_features is None:
            return [f"label_encoded"]
        else:
            return [f"{col}_encoded" for col in input_features]

class PreprocessingPipeline:
    def __init__(self, config_path: str):
        """
        Reads a YAML config (with column-level transformations) and initializes a pipeline.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # All transformations will eventually be stored in self.column_transformer
        self.column_transformer = None
        self.output_type = self.config.get("output_type", "pandas")
        
    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Build and fit the ColumnTransformer based on config, then transform X.
        """
        self.column_transformer = self._build_column_transformer()
        transformed = self.column_transformer.fit_transform(X, y)
        return self._format_output(transformed, X)
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Use the already-fitted ColumnTransformer to transform new data.
        """
        if not self.column_transformer:
            raise RuntimeError("Pipeline has not been fit yet. Call fit_transform first.")
        
        transformed = self.column_transformer.transform(X)
        return self._format_output(transformed, X)

    def _build_column_transformer(self) -> ColumnTransformer:
        """
        Builds a ColumnTransformer with a separate pipeline for each column, 
        based on the YAML config.
        """
        transformers = []

        # --- Numeric columns ---
        numeric_config = self.config["preprocessing"].get("numeric", {})
        numeric_cols = numeric_config.get("columns", {})
        # numeric_cols is a dict: { "ASK": {"imputer_strategy":..., "scaler":...}, ... }

        for col_name, col_settings in numeric_cols.items():
            # e.g. col_settings might be {"imputer_strategy": "mean", "scaler": "StandardScaler"}
            imputer_strategy = col_settings.get("imputer_strategy", "mean")
            scaler_name = col_settings.get("scaler", "StandardScaler")

            # Build the pipeline for this single numeric column
            col_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=imputer_strategy)),
                ("scaler", self._get_transformer(scaler_name))
            ])
            # Add to the ColumnTransformer as (name, pipeline, [column_name])
            transformers.append((f"num_{col_name}", col_pipeline, [col_name]))

        # --- Categorical columns ---
        categorical_config = self.config["preprocessing"].get("categorical", {})
        categorical_cols = categorical_config.get("columns", {})
        # cat_cols is a dict: { "TIMESTAMP": {"imputer_strategy": ..., "encoder": ...}, ... }

        for col_name, col_settings in categorical_cols.items():
            # e.g. {"imputer_strategy": "most_frequent", "encoder": "LabelEncoder"}
            imputer_strategy = col_settings.get("imputer_strategy", "most_frequent")
            encoder_name = col_settings.get("encoder", "OneHotEncoder")

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=imputer_strategy)),
                ("encoder", self._get_transformer(encoder_name, is_categorical=True))
            ])
            transformers.append((f"cat_{col_name}", cat_pipeline, [col_name]))

        # Build the final ColumnTransformer
        column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        return column_transformer

    def _get_transformer(self, transformer_name: str, is_categorical=False):
        """
        Dynamically create a transformer class by name.
        For numeric: StandardScaler, MinMaxScaler, etc.
        For categorical: OneHotEncoder, LabelEncoder, etc.
        """
        # You can maintain a dictionary or do if-else logic
        if not is_categorical:
            # Numeric
            if transformer_name == "StandardScaler":
                return StandardScaler()
            elif transformer_name == "MinMaxScaler":
                return MinMaxScaler()
            else:
                raise ValueError(f"Unknown numeric scaler: '{transformer_name}'")
        else:
            # Categorical
            if transformer_name == "OneHotEncoder":
                return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            elif transformer_name == "LabelEncoder":
                return LabelEncoderTransformer()
            else:
                # Optionally handle "OrdinalEncoder", etc.
                raise ValueError(f"Unknown categorical encoder: '{transformer_name}'")

    def _format_output(self, transformed, reference_df):
        if self.output_type == "pandas":
            # Reconstruct feature names from the ColumnTransformer
            feature_names = self.column_transformer.get_feature_names_out(
                input_features=reference_df.columns
            )
            print(feature_names)
            return pd.DataFrame(transformed, columns=feature_names)

        elif self.output_type == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed, but 'torch' output was requested.")
            return torch.tensor(transformed, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported output_type: {self.output_type}")
