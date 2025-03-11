import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PreprocessingPipeline:
    def __init__(self, config_path):
        """
        Reads a YAML config and initializes a pipeline accordingly.
        """
        # Load the YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Parse pre-processing config
        self.num_config = self.config['preprocessing']['numeric']
        self.cat_config = self.config['preprocessing']['categorical']
        self.output_type = self.config.get('output_type', 'pandas')  # default to 'pandas'
        
        # Prepare transformations
        # We create a numeric pipeline and a categorical pipeline based on the config.
        self.numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.num_config['imputer_strategy'])),
            # e.g. 'StandardScaler', 'MinMaxScaler', etc.
            ('scaler', self._get_transformer(self.num_config['scaler']))
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.cat_config['imputer_strategy'])),
            # e.g. 'OneHotEncoder', 'OrdinalEncoder', etc.
            ('encoder', self._get_transformer(self.cat_config['encoder']))
        ])
        
        # 4. The ColumnTransformer ties the numeric and categorical pipelines together.
        self.column_transformer = None
        
        # 5. We store the final fitted pipeline
        self.fitted_pipeline = None
    
    
    def _get_transformer(self, transformer_name):
        """
        Dynamically fetch transformer classes by name. Extend this as needed.
        """
        # Map string names to actual scikit-learn classes
        TRANSFORMER_MAP = {
            'StandardScaler': StandardScaler,
            'OneHotEncoder': OneHotEncoder,
            'MinMaxScaler':  lambda: __import__('sklearn.preprocessing').preprocessing.MinMaxScaler(),
            # You can add more transforms as needed
        }
        
        if transformer_name in TRANSFORMER_MAP:
            # If it's a class, instantiate it
            mapped = TRANSFORMER_MAP[transformer_name]
            # For some transforms, you might want to pass parameters
            if callable(mapped):
                return mapped()
            else:
                return mapped
        else:
            raise ValueError(f"Unknown transformer {transformer_name}. Please update TRANSFORMER_MAP.")
    
    
    def _build_column_transformer(self, X: pd.DataFrame):
        """
        Dynamically build a ColumnTransformer using the numeric_pipeline and
        categorical_pipeline. Uses data types to find numeric vs. categorical columns.
        """
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # If you prefer to specify columns in the YAML, you could skip this detection
        # and read the column names from the config instead.
        
        # Build the ColumnTransformer
        column_transformer = ColumnTransformer(transformers=[
            ('num', self.numeric_pipeline, numeric_features),
            ('cat', self.categorical_pipeline, categorical_features)
        ])
        
        return column_transformer
    
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit the pipeline to training data and then transform it.
        """
        # Build ColumnTransformer
        self.column_transformer = self._build_column_transformer(X)
        
        # Fit/Transform the data
        transformed = self.column_transformer.fit_transform(X, y)
        
        # Convert to final format (DataFrame or Torch)
        return self._format_output(transformed, X)
    
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Use the fitted pipeline to transform new data.
        """
        if not self.column_transformer:
            raise RuntimeError("Pipeline has not been fit yet. Call fit_transform first.")
        
        transformed = self.column_transformer.transform(X)
        return self._format_output(transformed, X)
    
    
    def _format_output(self, transformed, reference_df: pd.DataFrame):
        """
        Convert the transformed numpy array to either a Pandas DataFrame or PyTorch Tensor.
        """
        if self.output_type == 'pandas':
            # Reconstruct feature names (optional, but nice to have).
            # With OneHotEncoder, you’ll have multiple columns for each category,
            # so building robust column names can be more involved.
            # For demonstration, we’ll just use numeric column indices.
            col_count = transformed.shape[1]
            columns = [f"feature_{i}" for i in range(col_count)]
            return pd.DataFrame(transformed, columns=columns)
        
        elif self.output_type == 'torch':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not installed but 'torch' output was requested.")
            return torch.tensor(transformed, dtype=torch.float32)
        
        else:
            # Extend to other output formats as needed
            raise ValueError(f"Unknown output_type '{self.output_type}'. Valid options: 'pandas', 'torch'.")
