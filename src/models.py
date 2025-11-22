from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def train_models(X_train, y_train):
    """
    Initializes and trains the three required models:
    1. Logistic Regression (Baseline)
    2. Random Forest (Ensemble)
    3. XGBoost (Gradient Boosting)
    
    Handles class imbalance using 'class_weight' parameters
    """
    print("\n[Model Training] Initializing models...")
    
    models = {}

    # --- Model 1: Logistic Regression (Baseline) ---
    # class_weight='balanced': Automatically adjusts weights inversely proportional to class frequencies
    # max_iter=1000: Ensures the solver converges
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # --- Model 2: Random Forest ---
    # class_weight='balanced': Penalizes mistakes on the minority class (defaults) more heavily
    # n_estimators=100: Standard number of trees
    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # --- Model 3: XGBoost ---
    # scale_pos_weight: The XGBoost equivalent of class_weight
    # Calculation: count(negative) / count(positive) ~ 700/300 = 2.33
    # max_depth=3: Limits tree depth to prevent OVERFITTING
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=3,  # Conservative depth to avoid overfitting on small data (1000 rows)
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    print(f"Success! {len(models)} models trained.")
    return models