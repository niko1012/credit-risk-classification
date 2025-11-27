from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

def train_models(X_train, y_train):
    """
    Initializes and trains the models, including handling class imbalance via:
    1. Class Weights (for LR, RF, XGB)
    2. SMOTE (Synthetic Minority Over-sampling Technique)
    """
    print("\n[Model Training] Initializing models...")
    
    models = {}

    # --- Model 1: Logistic Regression (Method A: Class Weights) ---
    print("Training Logistic Regression (Balanced Weights)...")
    lr_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_balanced.fit(X_train, y_train)
    models['Logistic Regression (Balanced)'] = lr_balanced

    # --- Model 2: Logistic Regression (Method B: SMOTE) ---
    print("Training Logistic Regression (SMOTE)...")
    pipeline_smote = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline_smote.fit(X_train, y_train)
    models['Logistic Regression (SMOTE)'] = pipeline_smote

    # --- Model 3: Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # --- Model 4: XGBoost ---
    print("Training XGBoost...")
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    print(f"Success! {len(models)} models trained.")
    return models