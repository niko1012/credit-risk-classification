import pytest
import pandas as pd
import numpy as np
from src.models import train_models

def test_train_models():
    """
    Tests if the model training pipeline works correctly.
    It checks if the function returns a dictionary containing the 4 expected models.
    """
    # 1. Create dummy training data 
    X_train = pd.DataFrame(np.random.rand(50, 10), columns=[f"col_{i}" for i in range(10)])
    y_train = pd.Series(np.random.choice([0, 1], 50))

    # 2. Run the training function
    models = train_models(X_train, y_train)

    # 3. Assertions
    # Check if we got a dictionary
    assert isinstance(models, dict)
    
    # Check if we have exactly 4 models (LR Balanced, LR SMOTE, RF, XGB)
    assert len(models) == 4, f"Expected 4 models, got {len(models)}"
    
    # Check if specific model keys exist
    expected_models = [
        'Logistic Regression (Balanced)', 
        'Logistic Regression (SMOTE)', 
        'Random Forest', 
        'XGBoost'
    ]
    for name in expected_models:
        assert name in models, f"Model {name} is missing from results"
        
    # Check if the models have been fitted (they should have a 'predict' method)
    for name, model in models.items():
        assert hasattr(model, "predict"), f"Model {name} should have a predict method"