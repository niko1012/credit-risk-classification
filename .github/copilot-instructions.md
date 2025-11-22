# AI Agent Instructions: Credit Risk Classification

## Project Overview
**Credit Risk Classification Model** - A machine learning pipeline to predict loan default probability using the UCI German Credit Dataset (1000 observations, 20 features). This is an imbalanced binary classification problem requiring careful evaluation metric selection.

## Architecture & Data Flow

### Key Directories
- **`src/`** - Implementation code (currently empty, to be populated)
- **`tests/`** - Unit and integration tests
- **`data/`** - UCI German Credit Dataset inputs and processed data
- **`examples/`** - Example notebooks/scripts demonstrating usage
- **`results/`** - Model outputs, metrics, visualizations, trained models

### Core Pipeline Pattern
The project implements a standard data science workflow:
1. **Data Loading & EDA** → Explore class balance, feature distributions
2. **Preprocessing** → StandardScaler for normalization, handle missing values
3. **Train/Test Split** → Stratified 80/20 split (critical for imbalanced data)
4. **Model Training** → Three baseline models compared
5. **Cross-Validation** → Stratified 5-Fold CV for hyperparameter tuning
6. **Evaluation** → Recall, F1-Score, ROC-AUC (NOT accuracy)

## Models & Technologies

### ML Models (Compare These Three)
1. **Logistic Regression** - Baseline (scikit-learn)
2. **Random Forest** - Ensemble method (scikit-learn)
3. **XGBoost** - Gradient boosting (xgboost library)

### Key Libraries
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Preprocessing (`StandardScaler`), models, `train_test_split`, metrics
- **imbalanced-learn** - SMOTE for handling class imbalance
- **matplotlib/seaborn** - Visualizations
- **Python 3.10+**

## Critical Design Patterns

### Class Imbalance Handling
This dataset's primary challenge: defaults are the **minority class** (~30% default rate). Naive accuracy is useless.

**Required Approaches:**
- Use `class_weight='balanced'` in model constructors for automatic class weight adjustment
- Apply **SMOTE** (Synthetic Minority Over-sampling) to training data before CV/model fitting
- Compare baseline (class_weight) vs. SMOTE approaches side-by-side

### Evaluation Strategy
**Never use simple accuracy.** Always evaluate using:
- **Recall** - Priority: catch defaults (minimize false negatives)
- **F1-Score** - Balance precision/recall
- **ROC-AUC** - Probability calibration
- **Confusion Matrix** - Visualize actual vs. predicted

### Cross-Validation Pattern
Use **Stratified 5-Fold** to maintain class distribution:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Hyperparameter Tuning
Keep tuning **conservative** to prevent overfitting (only 1000 samples):
- XGBoost: Limit `max_depth` (e.g., 3-6), use regularization (`reg_alpha`, `reg_lambda`)
- Random Forest: Limit `max_depth`, use `max_samples`, `min_samples_leaf`
- Logistic Regression: Regularization strength `C` via CV

## Developer Workflows

### Setup
```bash
pip install -r requirements.txt  # Must include all dependencies
```

### Running
Expected structure: Scripts in `src/` or notebooks in `examples/` that:
1. Load data from `data/` directory
2. Execute pipeline steps
3. Save results (models, metrics, plots) to `results/`

### Testing
```bash
pytest tests/ -v  # Run all tests
```
Tests should verify:
- Data loading/preprocessing correctness
- Model training without errors
- Evaluation metric computation
- SMOTE application doesn't break pipelines

## Conventions & Patterns

### File Organization
- Keep data pipeline code separate from model training
- Notebook examples should be self-contained and reproducible
- Store trained models as `.pkl` or `.joblib` in `results/`

### Naming
- Feature names should preserve original UCI dataset semantics
- Model checkpoints: `{model_type}_{timestamp}.pkl`
- Results/metrics: Include fold number if from CV (e.g., `rf_metrics_fold_1.csv`)

### Reproducibility
- **Always set random seeds:** `random_state=42` in all sklearn/XGBoost calls
- Log hyperparameters, dataset splits, SMOTE parameters in result files
- Document which class weighting/SMOTE approach was used for each model

## Integration Points

### Dependency Chain
- Preprocessing (StandardScaler) must be fit on train data only, then transform test data
- SMOTE applied only to training set, never to test set
- Cross-validation must be stratified to preserve class distribution in each fold

### Model Comparison Framework
All three models should be evaluated identically:
1. Same train/test split and CV strategy
2. Same evaluation metrics computed for each
3. Results aggregated in a summary table (model name × metrics)

## Common Pitfalls to Avoid
- ❌ Using accuracy as the sole metric
- ❌ Fitting StandardScaler on entire dataset (data leakage)
- ❌ Applying SMOTE to test set
- ❌ Not using stratified splits with imbalanced data
- ❌ Using different evaluation metrics across models
- ❌ Overfitting with deep trees on small dataset (1000 samples)
