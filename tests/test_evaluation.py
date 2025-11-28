import pytest
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_models

def test_evaluate_models():
    """
    Integration test for the evaluation module

    Verifies that:
    1. The function runs without errors on dummy data and models
    2. It returns a summary DataFrame with the correct columns ('Model', 'ROC-AUC')
    3. The generated ROC-AUC score is a valid float between 0.0 and 1.0
    4. It handles the saving of results to the disk without crashing
    """
    # 1. Setup: Create dummy test data
    X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"col_{i}" for i in range(5)])
    y_test = pd.Series(np.random.choice([0, 1], 20))

    # 2. Setup: Create a dummy trained model
    model = LogisticRegression()
    model.fit(X_test, y_test)
    
    models = {'Test Model': model}

    # 3. Ensure 'results' directory exists 
    os.makedirs("results", exist_ok=True)

    # 4. Run the function
    try:
        results_df = evaluate_models(models, X_test, y_test)
    except Exception as e:
        pytest.fail(f"Evaluation function crashed: {e}")

    # 5. Assertions
    # Check return type
    assert isinstance(results_df, pd.DataFrame), "Should return a pandas DataFrame"
    
    # Check content
    assert not results_df.empty, "Results DataFrame should not be empty"
    assert "Model" in results_df.columns
    assert "ROC-AUC" in results_df.columns
    
    # Check if we got exactly 1 row
    assert len(results_df) == 1
    assert results_df.iloc[0]["Model"] == "Test Model"
    
    # Optional: Check if the ROC-AUC score is a valid float between 0 and 1
    auc = results_df.iloc[0]["ROC-AUC"]
    assert 0.0 <= auc <= 1.0