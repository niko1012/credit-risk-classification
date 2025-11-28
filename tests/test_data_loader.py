import pytest
import pandas as pd
import os
from src.data_loader import load_german_credit_data

def test_load_german_credit_data():
    """
    Unit test for the data loader module.

    Verifies that:
    1. The returned object is a pandas DataFrame
    2. The DataFrame is not empty
    3. The shape corresponds to the expected dataset size (1000 rows)
    4. Essential columns (including the Target) are present
    5. The Target variable contains only expected binary values (0 and 1)
    """
    
    df = load_german_credit_data()
    
    # 1. Check object type
    assert isinstance(df, pd.DataFrame), "Returned object should be a pandas DataFrame"
    
    # 2. Check if not empty
    assert not df.empty, "DataFrame should not be empty"
    
    # 3. Check shape (1000 rows expected)
    assert df.shape[0] == 1000, f"Expected 1000 rows, got {df.shape[0]}"
    
    # 4. Check specific columns exist
    expected_cols = ["Age", "Credit_amount", "Target"]
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from DataFrame"

    # 5. Check Target values (should be 0 or 1)
    unique_targets = df['Target'].unique()
    assert set(unique_targets).issubset({0, 1}), "Target should only contain 0 and 1"