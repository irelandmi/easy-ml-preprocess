import pandas as pd
import random
import numpy as np

def generate_mixed_type_dataset(file_path: str, num_rows: int = 100):
    data = {
        "ID": range(1, num_rows + 1),
        "MixedColumn": [random.choice([random.randint(1, 100), random.uniform(1, 100), random.choice(["A", "B", "C"])]) for _ in range(num_rows)],
        "NumericColumn": [random.uniform(10, 100) for _ in range(num_rows)],
        "StringColumn": [random.choice(["apple", "banana", "cherry", np.nan]) for _ in range(num_rows)],
        "BooleanColumn": [random.choice([True, False, np.nan]) for _ in range(num_rows)],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

if __name__ == "__main__":
    generate_mixed_type_dataset("mixed_types_dataset.csv")
