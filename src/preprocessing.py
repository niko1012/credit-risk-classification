import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(df: pd.DataFrame, target_col: str = 'Target', test_size: float = 0.2, random_state: int = 42):
    """
    Performs the full data preparation pipeline: splitting, encoding, and scaling

    Steps:
    1. Separates features (X) and target (y)
    2. Encodes categorical variables using One-Hot Encoding (pd.get_dummies)
    3. Performs a Stratified Train-Test Split to maintain the class balance 
    4. Applies StandardScaler to normalize features (fit on train, transform on test)
       to ensure convergence of linear models.

    Args:
        df (pd.DataFrame): The full dataset loaded from the data loader
        target_col (str): The name of the target column to predict. Defaults to 'Target'
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2
        random_state (int): Seed used by the random number generator for reproducibility

    Returns:
        tuple: A tuple containing four elements:
               - X_train_scaled (pd.DataFrame): Scaled training features
               - X_test_scaled (pd.DataFrame): Scaled testing features
               - y_train (pd.Series): Training labels
               - y_test (pd.Series): Testing labels
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