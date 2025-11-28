import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_and_split

def test_preprocess_and_split():
    """
    Unit test for the preprocessing module.

    Verifies that:
    1. The output types are correct (DataFrames/Series)
    2. The split ratio is respected (e.g., 80 rows for train, 20 for test on dummy data)
    3. Categorical variables are correctly One-Hot Encoded (column count increases)
    4. StandardScaler is correctly applied (resulting mean is approx 0, std approx 1)
    """
    # Create a dummy dataset
    data = {
        'Feature1': np.random.rand(100) * 100,  # Random data 0-100
        'Feature2': np.random.rand(100) * 50,   # Random data 0-50
        'Category': np.random.choice(['A', 'B'], 100), # Categorical
        'Target': np.random.choice([0, 1], 100) # Binary Target
    }
    df = pd.DataFrame(data)

    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocess_and_split(df, target_col='Target', test_size=0.2)

    # 1. Check output types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    # 2. Check split sizes (80 train / 20 test)
    assert len(X_train) == 80
    assert len(X_test) == 20

    # 3. Check One-Hot Encoding
    assert 'Category' not in X_train.columns
    assert 'Category_B' in X_train.columns

    # 4. Check Scaling (StandardScaler)
    # The mean of the training set features should be very close to 0
    mean_val = X_train['Feature1'].mean()
    std_val = X_train['Feature1'].std()
    
    assert abs(mean_val) < 0.1, "Feature should be scaled (mean approx 0)"
    assert abs(std_val - 1.0) < 0.1, "Feature should be scaled (std approx 1)"