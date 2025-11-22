import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(df: pd.DataFrame, target_col: str = 'Target', test_size: float = 0.2, random_state: int = 42):
    """
    Performs data preprocessing, encoding, scaling, and splitting
    
    Steps:
    1. Separates Target and Features
    2. One-Hot Encoding for categorical variables
    3. Stratified Train-Test Split
    4. Standard Scaling (Fit on Train, Transform on Test) to fix convergence issues
    """
    print("\n[Preprocessing] Starting data preparation...")

    # 1. Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Encode Categorical Variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Encoded features: {X.shape[1]} original -> {X_encoded.shape[1]} expanded features")

    # 3. Stratified Split
    print(f"Splitting data (Test size: {test_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    # 4. Standard Scaling (Fixes ConvergenceWarning)
    # Important: Fit ONLY on training data to avoid data leakage
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # We keep them as DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Simple test logic
    pass